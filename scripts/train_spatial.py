#!/usr/bin/env python3
"""Train the small U-Net on frozen training AOIs; select using development pixels."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from terradelta.spatial import BuildingUNet, masked_segmentation_loss
import terradelta.spatial as spatial_module


def open_region(cache: Path, region: str) -> dict:
    root = cache / region
    result = {name: np.load(root / f"{name}.npy", mmap_mode="r")
              for name in ("rgb", "target", "edge", "valid")}
    rgb = result["rgb"]
    if rgb.ndim != 4 or rgb.shape[1] != 3 or rgb.dtype != np.uint8:
        raise ValueError(f"bad RGB cache contract for {region}")
    expected = (rgb.shape[0], *rgb.shape[-2:])
    if any(result[name].shape != expected for name in ("target", "edge", "valid")):
        raise ValueError(f"cache shapes disagree for {region}")
    if result["valid"].dtype != np.bool_:
        raise ValueError(f"validity mask must be boolean for {region}")
    result["region_id"] = region
    result["metadata_sha256"] = hashlib.sha256((root / "metadata.json").read_bytes()).hexdigest()
    return result


def extract(region: dict, frame: int, top: int, left: int, patch_size: int):
    rgb = region["rgb"][frame, :, top:top + patch_size, left:left + patch_size].copy()
    target = np.stack([
        region[name][frame, top:top + patch_size, left:left + patch_size]
        for name in ("target", "edge")
    ]).astype(np.float32)
    valid = region["valid"][frame, top:top + patch_size, left:left + patch_size].copy()
    return rgb.astype(np.float32) / 255, target, valid


def train_batch(regions: list[dict], rng: np.random.Generator, batch_size: int, patch: int):
    images, targets, masks = [], [], []
    for _ in range(batch_size):
        region = regions[int(rng.integers(len(regions)))]
        count, _, height, width = region["rgb"].shape
        if min(height, width) < patch:
            raise ValueError("training image smaller than patch; resizing is not permitted")
        seek_building = rng.random() < 0.5
        for _attempt in range(5):
            frame = int(rng.integers(count))
            top = int(rng.integers(height - patch + 1))
            left = int(rng.integers(width - patch + 1))
            rgb, target, valid = extract(region, frame, top, left, patch)
            if valid.mean() >= 0.5 and (not seek_building or (target[0] * valid).sum() >= 8):
                break
        # Dihedral transformations preserve native resolution and exact labels.
        rotations = int(rng.integers(4))
        rgb, target, valid = [np.rot90(x, rotations, axes=(-2, -1))
                              for x in (rgb, target, valid)]
        if rng.random() < 0.5:
            rgb, target, valid = [np.flip(x, axis=-1) for x in (rgb, target, valid)]
        rgb = np.clip(rgb * rng.uniform(0.85, 1.15, (3, 1, 1))
                      + rng.uniform(-0.05, 0.05), 0, 1).astype(np.float32)
        images.append(rgb.copy())
        targets.append(target.copy())
        masks.append(valid.copy())
    return tuple(torch.from_numpy(np.stack(x)) for x in (images, targets, masks))


def development_batch(regions: list[dict], patch: int):
    """Fixed spatial grid/time sample; no event thresholds or confirmation labels."""
    images, targets, masks = [], [], []
    for region in regions:
        count, _, height, width = region["rgb"].shape
        if min(height, width) < patch:
            raise ValueError("development image smaller than patch")
        for frame in np.linspace(0, count - 1, min(4, count), dtype=int):
            for y_fraction, x_fraction in ((0.2, 0.2), (0.2, 0.8), (0.8, 0.2), (0.8, 0.8)):
                top = int((height - patch) * y_fraction)
                left = int((width - patch) * x_fraction)
                rgb, target, valid = extract(region, int(frame), top, left, patch)
                images.append(rgb)
                targets.append(target)
                masks.append(valid)
    return tuple(torch.from_numpy(np.stack(x)) for x in (images, targets, masks))


@torch.inference_mode()
def validate(model, data, device, batch_size=8):
    model.eval()
    intersection, predicted, actual = [np.zeros(2, dtype=np.float64) for _ in range(3)]
    loss_sum, batches = 0.0, 0
    for start in range(0, len(data[0]), batch_size):
        rgb, truth, valid = [x[start:start + batch_size].to(device) for x in data]
        logits = model(rgb)
        loss_sum += float(masked_segmentation_loss(logits, truth, valid))
        batches += 1
        mask = valid[:, None]
        binary = logits.sigmoid() >= 0.5
        axes = (0, 2, 3)
        intersection += (binary * truth * mask).sum(axes).cpu().numpy()
        predicted += (binary * mask).sum(axes).cpu().numpy()
        actual += (truth * mask).sum(axes).cpu().numpy()
    dice = (2 * intersection + 1) / (predicted + actual + 1)
    return {"footprint_dice": float(dice[0]), "boundary_dice": float(dice[1]),
            "loss": loss_sum / max(1, batches), "selection_score": float(dice.mean()),
            "valid_positive_pixels": actual.astype(int).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    args = parser.parse_args()
    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)
    limits = config["resource_limits"]
    if not 1 <= args.steps <= limits["spatial_training_steps"]:
        raise ValueError("steps exceed frozen resource limit")
    torch.set_num_threads(min(6, int(limits["cpu_threads"])))
    seed = int(config["seed"])
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    torch.use_deterministic_algorithms(True)
    device = args.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    training = [open_region(args.cache, region) for region in config["train"]]
    development = [open_region(args.cache, region) for region in config["development"]]
    model = BuildingUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    batch_size, patch = min(16, limits["batch_size"]), min(128, limits["patch_size"])
    dev_data = development_batch(development, patch)
    positive_weights = torch.tensor([3.0, 6.0], device=device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    trace, best, started = [], -1.0, time.monotonic()
    running_loss, running_count = 0.0, 0
    for step in range(1, args.steps + 1):
        model.train()
        rgb, target, valid = [x.to(device) for x in train_batch(training, rng, batch_size, patch)]
        optimizer.zero_grad(set_to_none=True)
        loss = masked_segmentation_loss(model(rgb), target, valid, positive_weights)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite training loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        running_loss += float(loss.detach())
        running_count += 1
        if step % 100 == 0 or step == 1 or step == args.steps:
            metrics = validate(model, dev_data, device)
            elapsed = time.monotonic() - started
            row = {"step": step, "elapsed_seconds": elapsed,
                   "train_loss": running_loss / running_count,
                   "development": metrics}
            trace.append(row)
            print(json.dumps(row), flush=True)
            running_loss, running_count = 0.0, 0
            if metrics["selection_score"] > best:
                best = metrics["selection_score"]
                checkpoint = {
                    "format": "terradelta-spatial-v1",
                    "architecture": {"name": "BuildingUNet", "base_channels": 16,
                                     "inputs": "RGB uint8 normalized /255", "outputs": ["footprint", "boundary"]},
                    "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                    "training": {"config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                                 "config": config, "train_regions": config["train"],
                                 "development_regions": config["development"], "seed": seed,
                                 "selected_step": step, "requested_steps": args.steps,
                                 "device": device, "pretrained": False,
                                 "parameters": sum(p.numel() for p in model.parameters()),
                                 "cache_metadata_sha256": {
                                     r["region_id"]: r["metadata_sha256"]
                                     for r in training + development
                                 },
                                 "source_sha256": {
                                     "scripts/train_spatial.py": hashlib.sha256(
                                         Path(__file__).read_bytes()
                                     ).hexdigest(),
                                     "src/terradelta/spatial.py": hashlib.sha256(
                                         Path(spatial_module.__file__).read_bytes()
                                     ).hexdigest(),
                                 },
                                 "positive_weights": [3.0, 6.0], "trace": trace.copy(),
                                 "development_pixel_metrics": metrics,
                                 "selection": "maximum mean footprint/boundary Dice on fixed development crops at threshold 0.5"},
                }
                temporary = args.output.with_suffix(args.output.suffix + ".tmp")
                torch.save(checkpoint, temporary)
                os.replace(temporary, args.output)
            trace_path = args.output.with_suffix(".training.json")
            trace_path.write_text(json.dumps({"device": device, "steps": step,
                                            "elapsed_seconds": elapsed, "trace": trace}, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "best_development_score": best,
                      "seconds": time.monotonic() - started}), flush=True)


if __name__ == "__main__":
    main()

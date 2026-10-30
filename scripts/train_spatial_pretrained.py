#!/usr/bin/env python3
"""Fit only the skip decoder on frozen TRAIN AOIs; select on fixed DEV pixels."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch

import terradelta.pretrained_spatial as model_source
import terradelta.spatial as loss_source
from terradelta.pretrained_spatial import (
    BACKBONE_ENUM, BACKBONE_URI, PretrainedBuildingUNet, make_checkpoint, sha256_file,
)
from terradelta.spatial import masked_segmentation_loss
import train_spatial as sampler_source
from train_spatial import development_batch, open_region, train_batch, validate


class TrainingDeadline(RuntimeError):
    """A wall-clock alarm stops updates even between scheduled checks."""


def atomic_json(path: Path, data: dict):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(temporary, path)


def save_checkpoint(path: Path, checkpoint: dict):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


def require_config(config: dict, steps: int, backbone_sha256: str):
    limits = config["resource_limits"]
    if not 1 <= steps <= min(4000, int(limits["spatial_training_steps"])):
        raise ValueError("steps exceed frozen resource limit")
    if not 0 < float(limits["spatial_wall_seconds"]) <= 10800:
        raise ValueError("wall-time budget must be at most three hours")
    expected = {"patch_size": 128, "development_patch_size": 128,
                "batch_size": 4, "input_scale": 2, "cpu_threads": 2}
    if any(limits.get(key) != value for key, value in expected.items()):
        raise ValueError("resource configuration differs from bounded pretrained protocol")
    backbone = config["spatial_backbone"]
    if (backbone["sha256"] != backbone_sha256 or backbone["enum"] != BACKBONE_ENUM
            or backbone["uri"] != BACKBONE_URI):
        raise ValueError("backbone differs from frozen configuration")
    roles = [config[key] for key in ("train", "development", "confirmation")]
    if [len(role) for role in roles] != [9, 4, 3]:
        raise ValueError("this revision requires the frozen nine/four/three geographic split")
    all_regions = [region for role in roles for region in role]
    if len(all_regions) != len(set(all_regions)):
        raise ValueError("geographic roles overlap or duplicate a region")
    audit = config.get("audit_only", config.get("audit_only_regions", []))
    if set(audit) & set(all_regions):
        raise ValueError("old audit-only regions may not enter the new study")


def encoder_fingerprint(model) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.encoder.state_dict().items()}


def assert_frozen(model, before):
    if model.encoder.training or any(p.requires_grad for p in model.encoder.parameters()):
        raise RuntimeError("encoder training/gradient contract violated")
    after = model.encoder.state_dict()
    if any(not torch.equal(value, after[key].detach().cpu()) for key, value in before.items()):
        raise RuntimeError("frozen encoder parameters or BatchNorm statistics changed")
    if any(p.grad is not None for p in model.encoder.parameters()):
        raise RuntimeError("frozen encoder unexpectedly accumulated a gradient")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--backbone-sha256", required=True)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--pilot-only", action="store_true",
                        help="One actual TRAIN masked update and frozen-state audit; no DEV reads")
    args = parser.parse_args()
    started = time.monotonic()
    config = json.loads(args.config.read_bytes())
    require_config(config, args.steps, args.backbone_sha256)
    trace_path = args.output.with_suffix(".training.json")
    pilot_path = args.output.with_suffix(".pilot.json")
    if any(path.exists() for path in (args.output, trace_path, pilot_path)):
        raise FileExistsError("output already exists; use a new immutable version path")
    torch.set_num_threads(2)
    seed = int(config["seed"])
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    torch.use_deterministic_algorithms(True)
    device = args.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; no silent fallback")
    model = PretrainedBuildingUNet(args.backbone, args.backbone_sha256, input_scale=2).to(device)
    encoder_before = encoder_fingerprint(model)
    optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=.001, weight_decay=.0001)
    training = [open_region(args.cache, region) for region in config["train"]]
    positive_weights = torch.tensor([3., 6.], device=device)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def update():
        model.train()
        rgb, truth, valid = [x.to(device) for x in train_batch(training, rng, 4, 128)]
        if not bool(valid.any()):
            raise RuntimeError("TRAIN batch has no valid pixels")
        optimizer.zero_grad(set_to_none=True)
        loss = masked_segmentation_loss(model(rgb), truth, valid, positive_weights)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("non-finite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 5)
        optimizer.step()
        return float(loss.detach()), int(valid.sum())

    if args.pilot_only:
        tick = time.monotonic()
        loss, valid_pixels = update()
        if device == "mps":
            torch.mps.synchronize()
        assert_frozen(model, encoder_before)
        result = {"pilot_only": True, "optimizer_steps": 1, "device": device,
                  "loss": loss, "valid_pixels": valid_pixels,
                  "step_seconds": time.monotonic() - tick,
                  "encoder_and_batchnorm_unchanged": True, "train_regions": config["train"],
                  "config_sha256": sha256_file(args.config), "backbone_sha256": args.backbone_sha256,
                  "total_parameters": sum(p.numel() for p in model.parameters()),
                  "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                  "source_sha256": source_hashes(),
                  "not_a_checkpoint": "pilot decoder update is discarded; full fit starts from seed"}
        atomic_json(pilot_path, result)
        print(json.dumps(result), flush=True)
        return

    development = [open_region(args.cache, region) for region in config["development"]]
    dev_data = development_batch(development, 128)
    if len(dev_data[0]) != 16 * len(development):
        raise ValueError("requires sixteen fixed development crops per region")
    inputs = {r["region_id"]: {
        name: sha256_file(args.cache / r["region_id"] / f"{name}.npy")
        for name in ("rgb", "target", "edge", "valid")
    } | {"metadata.json": r["metadata_sha256"]} for r in training + development}
    deadline = started + float(config["resource_limits"]["spatial_wall_seconds"])
    trace, selected, best, completed, running_loss, count = [], None, -1., 0, 0., 0
    stop_reason = "requested_steps"
    provenance = {
        "config": config, "config_sha256": sha256_file(args.config),
        "train_regions": config["train"], "development_regions": config["development"],
        "seed": seed, "requested_steps": args.steps,
        "device": device, "torch": str(torch.__version__), "source_sha256": source_hashes(),
        "input_sha256": inputs, "positive_weights": [3., 6.],
        "optimizer": "AdamW decoder only; lr=.001 weight_decay=.0001 gradient_clip=5",
        "external_pretraining": True, "native_training_patch_size": 128,
        "internal_training_patch_size": 256, "development_crops_per_region": 16,
        "augmentation": "exact dihedral; per-channel RGB gain [.85,1.15], offset [-.05,.05]",
        "selection": "maximum mean footprint/boundary Dice on fixed DEV native128 crops at .5",
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }

    def evaluate(step):
        nonlocal selected, best, running_loss, count
        metrics = validate(model, dev_data, device, batch_size=4)
        row = {"step": step, "elapsed_seconds": time.monotonic() - started,
               "train_loss": running_loss / count if count else None, "development": metrics}
        trace.append(row)
        print(json.dumps(row), flush=True)
        running_loss, count = 0., 0
        if metrics["selection_score"] > best:
            best = metrics["selection_score"]
            selected = make_checkpoint(model, provenance | {
                "selected_step": step, "development_pixel_metrics": metrics,
                "trace": trace.copy(), "steps_completed": completed,
                "elapsed_seconds": time.monotonic() - started,
            })
            save_checkpoint(args.output, selected)
        atomic_json(trace_path, {"trace": trace, "steps_completed": completed,
                                 "elapsed_seconds": time.monotonic() - started})

    # The best fully saved checkpoint survives an interrupted optimizer update.
    # On macOS/Linux this alarm also covers slow validation, rather than only
    # noticing the budget at the next 100-update checkpoint. Native operations
    # finish their current call before Python can deliver the alarm exception.
    if time.monotonic() >= deadline:
        raise RuntimeError("resource deadline exhausted during initialization/input hashing")

    def deadline_alarm(_signal, _frame):
        raise TrainingDeadline("frozen wall-time limit reached")

    previous_handler = signal.signal(signal.SIGALRM, deadline_alarm)
    signal.setitimer(signal.ITIMER_REAL, max(.001, deadline - time.monotonic()))
    try:
        evaluate(0)
        for step in range(1, args.steps + 1):
            if time.monotonic() >= deadline:
                stop_reason = "wall_time_limit"
                break
            loss, _ = update()
            completed = step
            running_loss += loss
            count += 1
            if step == 1 or step % 100 == 0 or step == args.steps:
                evaluate(step)
    except TrainingDeadline:
        stop_reason = "wall_time_limit"
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
    assert_frozen(model, encoder_before)
    if selected is None:
        raise RuntimeError("wall-time limit expired before any completed development selection")
    if (source_hashes() != provenance["source_sha256"]
            or sha256_file(args.config) != provenance["config_sha256"]):
        raise RuntimeError("training source/config bytes changed during fit")
    selected["training"].update({
        "steps_completed": completed,
        "elapsed_seconds": time.monotonic() - started, "stop_reason": stop_reason,
        "trace": trace,
        "encoder_parameters_and_batchnorm_unchanged": True,
    })
    save_checkpoint(args.output, selected)
    atomic_json(trace_path, selected["training"])
    print(json.dumps({"output": str(args.output), "checkpoint_sha256": sha256_file(args.output),
                      "best_development_score": best, "stop_reason": stop_reason,
                      "steps_completed": completed}), flush=True)


def source_hashes() -> dict[str, str]:
    return {name: sha256_file(path) for name, path in (
        ("scripts/train_spatial_pretrained.py", __file__),
        ("scripts/train_spatial.py", sampler_source.__file__),
        ("src/terradelta/pretrained_spatial.py", model_source.__file__),
        ("src/terradelta/spatial.py", loss_source.__file__),
    )}


if __name__ == "__main__":
    main()

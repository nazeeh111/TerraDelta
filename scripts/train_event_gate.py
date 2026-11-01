#!/usr/bin/env python3
"""Train event correctness on TRAIN proposals, restore DEV proposal prevalence."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
import torch
from torch.nn import functional as F

from terradelta.data import sha256
from terradelta.event_gate import (
    FORMAT, GATE_FEATURE_NAMES, QUALITY_DROP_INDICES, RAW_CONFIG, EventGate,
    assert_spatial_provenance, event_features, raw_proposals,
    standardize, support_indices, validate_spatial_provenance,
)
from terradelta.evaluation import ground_truth_events, match_events
from terradelta.features import FEATURE_NAMES, validate_feature_protocol
from terradelta.protocol import semantic_metadata_hash

SPATIAL_V2_SHA256 = "7ab95679a2009918dbab62fca64244476df519080cb8c0d376c00d1494983886"
MATCHING = {"iou_threshold": 0.25, "temporal_tolerance_days": 93,
            "max_detection_delay_days": None}
MAX_LATER_USABLE_ROWS = 3
SOURCE_FILES = (
    "src/terradelta/event_gate.py", "scripts/train_event_gate.py",
    "src/terradelta/decoder.py", "src/terradelta/evaluation.py",
    "src/terradelta/features.py", "src/terradelta/schema.py",
)


def expected_spatial_provenance(config):
    """Only the historical v4 configuration may omit the explicit new binding."""
    gate_config = config.get("event_gate", {})
    if "spatial_provenance" not in gate_config:
        if (config.get("version") != "sn7-regions-v4-event-gate"
                or config.get("spatial_backbone") is not None):
            raise ValueError("new gate configuration requires explicit spatial_provenance")
        return None
    expected = validate_spatial_provenance(gate_config["spatial_provenance"])
    if (config.get("spatial_backbone") is not None
            and expected["spatial_backbone_sha256"] != config["spatial_backbone"].get("sha256")):
        raise ValueError("gate backbone binding differs from study spatial_backbone")
    return expected


def validate_region_provenance(info, region, expected):
    if (info.get("feature_names") != FEATURE_NAMES
            or info.get("feature_protocol") != "calendar_quality_v2"
            or info.get("supervision_protocol") != "physical_coverage_80_20_v2"):
        raise ValueError(f"feature/label protocol guard failed: {region}")
    if expected is not None:
        assert_spatial_provenance(info, expected)
    elif (info.get("spatial_weights_sha256") != SPATIAL_V2_SHA256
          or any(info.get(key) is not None for key in ("spatial_backbone_sha256", "spatial_inference"))):
        raise ValueError("legacy gate requires frozen spatial-v2 without new backbone/inference")
    if info.get("region_id") != region:
        raise ValueError("region identity differs from frozen split")


def preflight_spatial_provenance(feature_root, cache_root, regions, expected):
    """Check EVERY TRAIN/DEV manifest before opening any arrays or truth labels."""
    if len(regions) != len(set(regions)):
        raise ValueError("duplicate region in train/development split")
    for region in regions:
        info = validate_feature_protocol(feature_root / region, supervision=True)
        validate_region_provenance(info, region, expected)
        if info.get("metadata_semantic_sha256") != semantic_metadata_hash(
            cache_root / region / "metadata.json"
        ):
            raise ValueError("feature metadata differs from truth source metadata")


def collect_region(feature_root, cache_root, region, truth_protocol, spatial_provenance=None):
    """Retractions never define labels: even subsequently retracted IDs are matched."""
    directory = feature_root / region
    info = validate_feature_protocol(directory, supervision=True)
    validate_region_provenance(info, region, spatial_provenance)
    cache = cache_root / region
    if info.get("metadata_semantic_sha256") != semantic_metadata_hash(cache / "metadata.json"):
        raise ValueError("feature metadata differs from truth source metadata")
    with np.load(directory / "features.npz") as stored:
        x, usable, births = stored["x"], stored["usable"], stored["births"]
    raw = raw_proposals(x, usable, births, info,
                        record_origin=datetime(2026, 1, 1, tzinfo=timezone.utc))
    truth = ground_truth_events(cache, protocol=truth_protocol)
    matches = match_events(raw, truth, **MATCHING)
    positive = {raw[prediction_index].event_id for prediction_index, _, _ in matches}
    first_confirmed, retractions = {}, {}
    for event in raw:
        _, _, _, current = support_indices(event)
        if event.state == "confirmed_by_model":
            first_confirmed.setdefault(event.event_id, event)
        if event.state == "retracted":
            retractions.setdefault(event.event_id, current)
    rows, labels, masses, keys, times = [], [], [], [], []
    for event_id, event in first_confirmed.items():
        site_id, _, _, first_eligible = support_indices(event)
        usable_after = np.flatnonzero(usable[site_id - 1, first_eligible:]) + first_eligible
        # A later raw retraction only stops rows AT that observed prefix. It never
        # removes or relabels an earlier confirmation/example based on its future.
        usable_after = usable_after[usable_after < retractions.get(event_id, x.shape[1])]
        selected = usable_after[:1 + MAX_LATER_USABLE_ROWS]
        if not len(selected) or selected[0] != first_eligible:
            raise ValueError("raw first confirmation lacks usable first eligible support")
        for current in selected:
            values, _ = event_features(x, usable, event, int(current), info)
            rows.append(values)
            labels.append(int(event_id in positive))
            masses.append(1 / len(selected))
            keys.append(event_id)
            times.append(int(current))
    hashes = {
        "features_sha256": sha256(directory / "features.npz"),
        "sites_sha256": sha256(directory / "sites.json"),
        "instances_sha256": sha256(cache / "instances.npy"),
        "valid_sha256": sha256(cache / "valid.npy"),
        "metadata_semantic_sha256": semantic_metadata_hash(cache / "metadata.json"),
    }
    audit = {
        "region_id": region, "truth_events": len(truth),
        "raw_proposed_ids": len({event.event_id for event in raw if event.state == "proposed"}),
        "eligible_proposals": len(first_confirmed), "positive_proposals": len(positive),
        "negative_proposals": len(first_confirmed) - len(positive), "prefix_rows": len(rows),
        "subsequently_retracted_eligible_proposals": sum(
            event_id in retractions for event_id in first_confirmed
        ),
        "hashes": hashes,
    }
    return {"x": np.asarray(rows, np.float32).reshape(-1, len(GATE_FEATURE_NAMES)),
            "y": np.asarray(labels, np.float32), "mass": np.asarray(masses, np.float64),
            "event_ids": np.asarray(keys), "indices": np.asarray(times, np.int32),
            "regions": np.asarray([region] * len(rows)), "audit": audit}


def collect_split(feature_root, cache_root, regions, truth_protocol, spatial_provenance=None):
    parts = []
    for region in regions:
        part = collect_region(feature_root, cache_root, region, truth_protocol, spatial_provenance)
        print(json.dumps(part["audit"]), flush=True)
        parts.append(part)
    result = {key: np.concatenate([part[key] for part in parts])
              for key in ("x", "y", "mass", "event_ids", "indices", "regions")}
    result["audits"] = [part["audit"] for part in parts]
    return result


def platt_calibration(scores, labels, mass):
    """Weighted DEV MLE; every actual proposal has total weight one, never balanced."""
    weights = mass / mass.sum()
    prevalence = float(np.dot(weights, labels))
    if not np.any(labels == 0) or not np.any(labels == 1):
        scale, bias, success = 0.0, float(logit(np.clip(prevalence, 1e-6, 1 - 1e-6))), True
        calibration_kind = "constant_one_class_development"
    else:
        def objective(parameters):
            scale = np.exp(parameters[0])
            z = scale * scores + parameters[1]
            error = expit(z) - labels
            value = np.dot(weights, np.logaddexp(0, z) - labels * z)
            penalty = 1e-4 * np.dot(parameters, parameters)
            gradient = np.array([np.dot(weights, error * scale * scores),
                                 np.dot(weights, error)]) + 2e-4 * parameters
            return value + penalty, gradient

        fit = minimize(objective, np.array([0., logit(np.clip(prevalence, 1e-6, 1 - 1e-6))]),
                       jac=True, method="BFGS", options={"gtol": 1e-7})
        if not np.isfinite(fit.x).all():
            raise ValueError("development calibration failed to produce finite parameters")
        scale, bias, success = float(np.exp(fit.x[0])), float(fit.x[1]), bool(fit.success)
        calibration_kind = "positive_slope_platt"
    probability = expit(scale * scores + bias)
    bins = []
    for lower in np.linspace(0, 0.9, 10):
        included = (probability >= lower) & (probability < lower + 0.1 + 1e-9)
        if included.any():
            relative = weights[included] / weights[included].sum()
            bins.append({"lower": float(lower), "rows": int(included.sum()),
                         "proposal_weight": float(mass[included].sum()),
                         "score_mean": float(np.dot(relative, probability[included])),
                         "positive_fraction": float(np.dot(relative, labels[included]))})
    return {
        "scale": scale, "bias": bias, "threshold": 0.5, "success": success,
        "calibration_kind": calibration_kind, "prefix_rows": len(labels),
        "proposal_mass": float(mass.sum()), "event_prevalence": prevalence,
        "brier": float(np.dot(weights, (probability - labels) ** 2)),
        "reliability_bins": bins,
        "meaning": "event compatibility under frozen one-to-one physical geometry/time matching; not arbitrary-shift or timing-coverage guarantee",
        "prevalence_weights": "each real DEV proposal totals one across bounded available prefix rows",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=None)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    config_bytes = args.config.read_bytes()
    config_hash = hashlib.sha256(config_bytes).hexdigest()
    config = json.loads(config_bytes)
    gate_config = config.get("event_gate", {})
    source_hashes = {name: sha256(repo / name) for name in SOURCE_FILES}
    if (gate_config.get("truth_ready") is not True
            or gate_config.get("truth_source_sha256") != source_hashes["src/terradelta/evaluation.py"]
            or gate_config.get("supervision_source_sha256") != source_hashes["src/terradelta/features.py"]):
        raise ValueError("event gate training requires explicit ready/audited frozen truth source hash")
    spatial_provenance = expected_spatial_provenance(config)
    for key, expected in (("proposal_threshold", RAW_CONFIG.threshold),
                          ("confirmations", RAW_CONFIG.confirmations),
                          ("minimum_dwell_days", RAW_CONFIG.minimum_dwell_days)):
        if gate_config.get(key) != expected:
            raise ValueError("frozen raw proposal policy differs from the shared decoder")
    truth_protocol = config["truth_protocol"]
    if truth_protocol != "physical_coverage_v2":
        raise ValueError("event gate training requires audited physical-coverage truth protocol")
    cap = gate_config.get("training_steps", 1000)
    if args.steps is None:
        args.steps = cap
    if not 1 <= args.steps <= cap:
        raise ValueError("event gate step budget exceeded")
    if any(set(config[a]) & set(config[b]) for a, b in
           (("train", "development"), ("train", "confirmation"), ("development", "confirmation"))):
        raise ValueError("geographic split overlap")
    metadata_path = args.output.with_suffix(".json")
    examples_path = args.output.with_suffix(".examples.npz")
    if any(path.exists() for path in (args.output, metadata_path, examples_path)):
        raise ValueError("event gate outputs are immutable; choose a new versioned output path")
    preflight_spatial_provenance(args.features_root, args.cache_root,
                                 config["train"] + config["development"], spatial_provenance)
    started = time.monotonic()
    train = collect_split(args.features_root, args.cache_root, config["train"], truth_protocol,
                          spatial_provenance)
    dev = collect_split(args.features_root, args.cache_root, config["development"], truth_protocol,
                        spatial_provenance)
    if len(train["x"]) == 0 or len(dev["x"]) == 0:
        raise ValueError("actual TRAIN and DEV eligible proposals are both required")
    positive, negative = np.flatnonzero(train["y"] == 1), np.flatnonzero(train["y"] == 0)
    if not len(positive) or not len(negative):
        raise ValueError("actual TRAIN proposals must include both correct and incorrect events")
    mean = np.average(train["x"], axis=0, weights=train["mass"]).astype(np.float32)
    variance = np.average((train["x"] - mean) ** 2, axis=0, weights=train["mass"])
    std = np.sqrt(variance).astype(np.float32)
    std[std < 1e-5] = 1
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(config["seed"])
    # Same real example indices and initialization for the full/quality ablation.
    batches = []
    for _ in range(args.steps):
        batches.append(np.concatenate([
            rng.choice(indices, 128, replace=True,
                       p=train["mass"][indices] / train["mass"][indices].sum())
            for indices in (positive, negative)
        ]))
    models, calibrations, traces = {}, {}, {}
    for kind in ("observation_gate", "gate_no_quality"):
        torch.manual_seed(config["seed"])
        model = EventGate()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
        tx = torch.from_numpy(standardize(train["x"], mean, std, kind))
        ty = torch.from_numpy(train["y"])
        trace = []
        for step, indices in enumerate(batches, 1):
            model.train()
            logits = model(tx[indices])[:, 0]
            loss = F.binary_cross_entropy_with_logits(logits, ty[indices])
            if not torch.isfinite(loss):
                raise ValueError("non-finite gate training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            if step == 1 or step % 100 == 0 or step == args.steps:
                row = {"step": step, "balanced_train_bce": float(loss.detach())}
                trace.append(row)
                print(kind, json.dumps(row), flush=True)
        model.eval()
        with torch.inference_mode():
            raw_scores = model(torch.from_numpy(standardize(dev["x"], mean, std, kind)))[:, 0].numpy()
        calibrations[kind] = platt_calibration(raw_scores, dev["y"], dev["mass"])
        models[kind] = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
        traces[kind] = trace
    if (source_hashes != {name: sha256(repo / name) for name in SOURCE_FILES}
            or args.config.read_bytes() != config_bytes):
        raise ValueError("source changed during event-gate study; outputs not frozen")
    data_protocol = {
        "truth_protocol": truth_protocol, "matching": MATCHING,
        "later_usable_rows_per_proposal": MAX_LATER_USABLE_ROWS,
        "labels": "One-to-one match of every raw first confirmation to physical truth, including later retractions; later prefix rows inherit physical-event compatibility",
        "late_detection": "Compatibility labels are not timely-alert labels; main evaluation must score matched detection horizons separately",
        "class_balance": "Each training batch draws128 actual positive and128 actual negative prefix rows; conditional sampling gives each proposal total mass1",
        "calibration": "Unbalanced global DEV proposals with per-proposal unit mass; no region-specific fitting",
        "checkpoint_selection": "Fixed training budget; final weights, no occupancy/development checkpoint selection",
        "train": train["audits"], "development": dev["audits"],
    }
    checkpoint = {
        "format": FORMAT, "feature_names": GATE_FEATURE_NAMES,
        "models": models, "mean": torch.from_numpy(mean), "std": torch.from_numpy(std),
        "calibration": calibrations,
        "spatial_weights_sha256": (spatial_provenance["spatial_weights_sha256"]
                                   if spatial_provenance is not None else SPATIAL_V2_SHA256),
        "raw_decoder": asdict(RAW_CONFIG), "quality_drop_indices": list(QUALITY_DROP_INDICES),
        "training": {"steps": args.steps, "seed": config["seed"],
                     "config_sha256": config_hash, "source_sha256": source_hashes,
                     "data_protocol": data_protocol, "trace": traces},
    }
    if spatial_provenance is not None:
        checkpoint["spatial_provenance"] = spatial_provenance
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents a late collision from replacing previous bytes.
    with examples_path.open("xb") as stream:
        np.savez_compressed(stream, **{
            f"{split}_{key}": data[key]
            for split, data in (("train", train), ("development", dev))
            for key in ("x", "y", "mass", "event_ids", "indices", "regions")
        })
    checkpoint["training"]["examples_sha256"] = sha256(examples_path)
    with args.output.open("xb") as stream:
        torch.save(checkpoint, stream)
    public = {key: value for key, value in checkpoint.items()
              if key not in ("models", "mean", "std")}
    public.update({"weights_sha256": sha256(args.output),
                   "elapsed_seconds": time.monotonic() - started,
                   "standardizer_sha256": hashlib.sha256(mean.tobytes() + std.tobytes()).hexdigest()})
    with metadata_path.open("x") as stream:
        stream.write(json.dumps(public, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "weights_sha256": public["weights_sha256"],
                      "elapsed_seconds": public["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()

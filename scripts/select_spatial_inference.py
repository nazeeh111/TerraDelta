#!/usr/bin/env python3
"""Prospective DEV-only native-instance selection of global inference settings.

Prepare freezes the grid and exact DEV scenes without opening the mutable decoder.
Run requires explicit fit-completion acknowledgement and a finalized 4,000-step
checkpoint, or a strictly hash-bound independent recovery receipt. No confirmation
or audit-only cache is opened by either mode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

TILES = [128, 256, 512, 1024]
THRESHOLDS = [0.35, 0.45, 0.55, 0.65]
IOU = 0.25
SOURCE_VALID = 0.8
WALL_SECONDS = 1800
CACHE_FILES = ["metadata.json", "rgb.npy", "valid.npy", "instances.npy", "target.npy", "edge.npy"]
SOURCE_FILES = [
    "scripts/select_spatial_inference.py",
    "src/terradelta/pretrained_spatial.py",
    "src/terradelta/spatial.py",
    "src/terradelta/features.py",
    "src/terradelta/data.py",
    "src/terradelta/schema.py",
]


class SelectionDeadline(RuntimeError):
    """The prospective complete-grid budget expired."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def source_hashes() -> dict:
    repo = Path(__file__).resolve().parents[1]
    return {name: sha256(repo / name) for name in SOURCE_FILES}


def prepare(args) -> None:
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    path = output / "selection-plan.json"
    if path.exists() or (output / "progress.json").exists() or (output / "selection.json").exists():
        raise FileExistsError("selection plan/run exists; preserve it and use a new output version")
    config = json.loads(args.config.read_text())
    regions = config["development"]
    if len(regions) != 4 or len(set(regions)) != 4:
        raise ValueError("requires the fixed four-region DEV split")
    if set(regions) & (
        set(config["train"]) | set(config["confirmation"]) | set(config["audit_only"])
    ):
        raise ValueError("DEV roles overlap TRAIN, confirmation or audit-only")
    if "0.35,0.45,0.55,0.65" not in config["candidate_threshold_selection"]:
        raise ValueError("threshold grid differs from prospective experiment declaration")
    scenes, inputs = [], {}
    for region in regions:
        directory = args.cache / region
        metadata = json.loads((directory / "metadata.json").read_text())
        if metadata["region_id"] != region:
            raise ValueError("DEV cache identity mismatch")
        usable = [
            i for i, frame in enumerate(metadata["frames"]) if frame["registration"]["usable"]
        ]
        if len(usable) < 2:
            raise ValueError(
                "requires distinct first and last registration-usable scenes per DEV region"
            )
        for role, index in (("first_usable", usable[0]), ("last_usable", usable[-1])):
            scenes.append(
                {
                    "region_id": region,
                    "frame_index": index,
                    "selection": role,
                    "acquisition_start": metadata["frames"][index]["acquisition_start"],
                    "native_shape": [metadata["height"], metadata["width"]],
                }
            )
        inputs[region] = {name: sha256(directory / name) for name in CACHE_FILES}
    plan = {
        "format": "terradelta-dev-spatial-inference-selection-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "prospectively_frozen_before_metric_outcomes",
        "config_sha256": sha256(args.config),
        "config_path": str(args.config.resolve()),
        "cache_root": str(args.cache.resolve()),
        "decoder_path": str(args.weights.resolve()),
        "backbone_path": str(args.backbone.resolve()),
        "backbone_sha256": config["spatial_backbone"]["sha256"],
        "decoder_identity": "Bind final bytes only after explicit fit-completion acknowledgement, before inference",
        "development_regions": regions,
        "scenes": scenes,
        "input_sha256": inputs,
        "source_sha256": source_hashes(),
        "tile_sizes": TILES,
        "candidate_thresholds": THRESHOLDS,
        "wall_seconds": WALL_SECONDS,
        "metric": {
            "object_matching": "maximum-cardinality one-to-one native pixel IoU>=0.25",
            "iou_threshold": IOU,
            "truth_minimum_source_valid_fraction": SOURCE_VALID,
            "truth": "Positive IDs present in native raster; full rasterized object areas, no label alignment",
            "prediction": "Exact footprint_instances output using original source-valid mask; no temporal site birth",
            "excluded_truth": "Audit low-valid IDs; predictions still count, with no additional label-assisted ignore mask",
            "empty_case": "F1=recall=precision=1 only when both truth and prediction are empty; otherwise absent precision/recall=0",
            "aggregation": "Mean two per-frame scores within each region, then equal mean over all four DEV regions",
            "selection_order": [
                "maximum global mean regional object F1",
                "maximum same-aggregation recall",
                "minimum total false positives",
                "minimum tile size",
                "minimum threshold",
            ],
            "diagnostics": "Native footprint Dice at candidate threshold; boundary Dice at0.5, source-valid pixels only",
            "limit": "Dimensionless native pixel overlap; not geodesic area or physical event-date accuracy",
        },
        "completion_policy": "All8 scenes x4 tiles x4 thresholds required; any missing/error/deadline means incomplete and no selected parameters",
    }
    atomic_json(path, plan)
    print(json.dumps({"prepared": str(path), "sha256": sha256(path), "scenes": scenes}), flush=True)


def instance_metrics(predicted: np.ndarray, truth: np.ndarray, valid: np.ndarray) -> dict:
    """Sparse maximum-cardinality matching; truth area is the full native raster."""
    if predicted.shape != truth.shape or truth.shape != valid.shape:
        raise ValueError("native instance/mask shapes disagree")
    if (
        np.any(truth < 0)
        or np.any(predicted < 0)
        or valid.dtype != np.bool_
        or not np.issubdtype(truth.dtype, np.integer)
    ):
        raise ValueError("expected nonnegative integer labels and explicit boolean validity")
    truth_ids, inverse = np.unique(truth, return_inverse=True)
    inverse = inverse.reshape(truth.shape)
    areas = np.bincount(inverse.ravel(), minlength=len(truth_ids))
    valid_areas = np.bincount(inverse[valid], minlength=len(truth_ids))
    fractions = valid_areas / np.maximum(areas, 1)
    eligible = (truth_ids > 0) & (fractions >= SOURCE_VALID)
    lookup = np.zeros(len(truth_ids), dtype=np.int64)
    lookup[eligible] = np.arange(1, int(eligible.sum()) + 1)
    dense_truth = lookup[inverse]
    prediction_ids, dense_prediction = np.unique(predicted, return_inverse=True)
    dense_prediction = dense_prediction.reshape(predicted.shape)
    # Mapping cannot assume a background label is present in a fully covered raster.
    pred_lookup = np.zeros(len(prediction_ids), dtype=np.int64)
    pred_lookup[prediction_ids > 0] = np.arange(1, np.count_nonzero(prediction_ids > 0) + 1)
    dense_prediction = pred_lookup[dense_prediction]
    n_truth, n_pred = int(eligible.sum()), int(np.count_nonzero(prediction_ids > 0))
    truth_areas = areas[eligible]
    pred_areas = np.bincount(dense_prediction.ravel(), minlength=n_pred + 1)[1:]
    overlap = (dense_truth > 0) & (dense_prediction > 0)
    pairs, intersections = np.unique(
        dense_truth[overlap] * (n_pred + 1) + dense_prediction[overlap], return_counts=True
    )
    truth_index, pred_index = pairs // (n_pred + 1) - 1, pairs % (n_pred + 1) - 1
    ious = intersections / (truth_areas[truth_index] + pred_areas[pred_index] - intersections)
    accept = ious >= IOU
    graph = csr_matrix(
        (np.ones(int(accept.sum()), dtype=np.int8), (truth_index[accept], pred_index[accept])),
        shape=(n_truth, n_pred),
    )
    matched = maximum_bipartite_matching(graph, perm_type="column") if n_truth and n_pred else []
    tp = int(np.count_nonzero(np.asarray(matched) >= 0))
    fp, fn = n_pred - tp, n_truth - tp
    empty = n_truth == n_pred == 0
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "truth_objects": n_truth,
        "predicted_objects": n_pred,
        "truth_objects_present": int(np.count_nonzero(truth_ids > 0)),
        "truth_objects_excluded_low_source_valid": int(
            np.count_nonzero((truth_ids > 0) & ~eligible)
        ),
        "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0,
        "recall": tp / n_truth if n_truth else float(empty),
        "precision": tp / n_pred if n_pred else float(empty),
    }


def dice(binary, truth, valid) -> float:
    a, b = binary & valid, (truth > 0) & valid
    return float(
        (2 * np.count_nonzero(a & b) + 1) / (np.count_nonzero(a) + np.count_nonzero(b) + 1)
    )


def summarize(rows: list[dict], regions: list[str]) -> list[dict]:
    summaries = []
    for tile in TILES:
        for threshold in THRESHOLDS:
            subset = [
                row for row in rows if row["tile_size"] == tile and row["threshold"] == threshold
            ]
            if len(subset) != 8:
                raise ValueError("incomplete prospective grid")
            per_region = {}
            for region in regions:
                values = [row for row in subset if row["region_id"] == region]
                if len(values) != 2:
                    raise ValueError("incomplete equal-regional selection")
                per_region[region] = {
                    name: float(np.mean([row[name] for row in values]))
                    for name in ("f1", "recall", "precision", "footprint_dice", "boundary_dice")
                }
            summaries.append(
                {
                    "tile_size": tile,
                    "threshold": threshold,
                    "per_region": per_region,
                    "global_mean_regional_f1": float(
                        np.mean([r["f1"] for r in per_region.values()])
                    ),
                    "global_mean_regional_recall": float(
                        np.mean([r["recall"] for r in per_region.values()])
                    ),
                    "total_false_positives": sum(row["false_positives"] for row in subset),
                    "total_true_positives": sum(row["true_positives"] for row in subset),
                    "total_false_negatives": sum(row["false_negatives"] for row in subset),
                }
            )
    return summaries


def validate_fit_recovery(args, checkpoint: dict, plan: dict) -> dict:
    """Verify the narrow audited fit exception without changing checkpoint bytes."""
    path = args.fit_recovery_receipt
    receipt = json.loads(path.read_text())
    training = checkpoint["training"]
    if (
        receipt.get("format") != "terradelta-spatial-fit-recovery-review-v1"
        or receipt.get("status") != "clear"
        or receipt.get("completed_steps") != 4000
        or receipt.get("selected_step") != training["selected_step"]
        or training["requested_steps"] != 4000
        or training["steps_completed"] != receipt.get("retained_checkpoint_steps_completed_field")
        or training["config_sha256"] != plan["config_sha256"]
        or receipt.get("config_sha256") != plan["config_sha256"]
        or receipt.get("all_audited_artifact_bytes_preserved") is not True
    ):
        raise ValueError("fit recovery receipt does not clear this exact completed fit")
    paths = receipt["artifact_paths"]
    if (
        Path(paths["checkpoint"]).resolve() != args.weights.resolve()
        or Path(paths["config"]).resolve() != args.config.resolve()
        or Path(paths["final_trace"]).resolve()
        != args.weights.with_suffix(".training.json").resolve()
    ):
        raise ValueError("fit recovery artifact paths differ")
    hashes = {
        "checkpoint": receipt["checkpoint_sha256"],
        "final_trace": receipt["trace_sha256"],
        "config": receipt["config_sha256"],
        "log": receipt["log_sha256"],
    }
    if (
        hashes != receipt["artifacts_sha256_before"]
        or hashes != receipt["artifacts_sha256_after"]
        or any(sha256(Path(paths[name])) != expected for name, expected in hashes.items())
    ):
        raise ValueError("fit recovery artifact hash mismatch")
    checks = receipt["source_checks"]
    if set(checks) != set(training["source_sha256"]) or any(
        checks[name]["recorded"] != expected
        or checks[name]["commit17_sha256"] != expected
        or checks[name]["recorded_matches_commit17"] is not True
        for name, expected in training["source_sha256"].items()
    ):
        raise ValueError("fit recovery training-source provenance mismatch")
    final = json.loads(Path(paths["final_trace"]).read_text())
    trace = final["trace"]
    log_rows = [
        json.loads(line) for line in Path(paths["log"]).read_text().splitlines()
        if line.startswith("{")
    ]
    if (
        final["steps_completed"] != 4000
        or [row["step"] for row in trace] != [0, 1, *range(100, 4001, 100)]
        or log_rows != trace
        or training["trace"] != trace[:len(training["trace"])]
        or not training["trace"]
        or training["trace"][-1]["step"] != training["selected_step"]
    ):
        raise ValueError("fit recovery completed trace mismatch")
    best = max(trace, key=lambda row: row["development"]["selection_score"])
    if (
        best["step"] != training["selected_step"]
        or best["development"] != training["development_pixel_metrics"]
        or receipt["retained_selection_metric"] != best["development"]
    ):
        raise ValueError("fit recovery retained checkpoint is not the complete-trace selection")
    return {
        "mode": "independently_audited_recovery",
        "receipt_path": str(path.resolve()),
        "receipt_sha256": sha256(path),
        "training_source_commit": receipt["commit"],
        "completed_steps": 4000,
        "selected_step": training["selected_step"],
        "checkpoint_steps_completed_field": training["steps_completed"],
        "artifact_sha256": hashes,
        "checkpoint_bytes_unchanged": True,
    }


def run(args) -> None:
    if not args.fit_completed:
        raise ValueError(
            "explicit --fit-completed acknowledgement is required before decoder access"
        )
    path = args.output / "selection-plan.json"
    plan = json.loads(path.read_text())
    if (args.output / "progress.json").exists() or (args.output / "selection.json").exists():
        raise FileExistsError("run exists; never resume/rewrite partial selection outcomes")
    if (
        plan["config_sha256"] != sha256(args.config)
        or plan["source_sha256"] != source_hashes()
        or plan["cache_root"] != str(args.cache.resolve())
        or plan["decoder_path"] != str(args.weights.resolve())
        or plan["backbone_path"] != str(args.backbone.resolve())
        or plan["tile_sizes"] != TILES
        or plan["candidate_thresholds"] != THRESHOLDS
    ):
        raise ValueError("frozen plan/source/path mismatch")
    config = json.loads(args.config.read_text())
    regions = config["development"]
    if (
        plan["development_regions"] != regions
        or set(plan["input_sha256"]) != set(regions)
        or len(plan["scenes"]) != 8
        or any(scene["region_id"] not in regions for scene in plan["scenes"])
        or any(
            sum(scene["region_id"] == region for scene in plan["scenes"]) != 2 for region in regions
        )
    ):
        raise ValueError("selection plan must contain only the four frozen DEV regions")
    started = time.monotonic()
    rows = []
    report = {
        "status": "running",
        "plan_sha256": sha256(path),
        "source_sha256": source_hashes(),
        "created_at": datetime.now(UTC).isoformat(),
        "rows": rows,
        "complete_scene_tile_predictions": 0,
        "selected": None,
    }

    def save():
        report["elapsed_seconds"] = time.monotonic() - started
        atomic_json(args.output / "progress.json", report)

    def deadline(_signal, _frame):
        raise SelectionDeadline("prospective1800-second wall limit reached")

    previous = signal.signal(signal.SIGALRM, deadline)
    signal.setitimer(signal.ITIMER_REAL, WALL_SECONDS)
    try:
        # All input identity checks precede interpretation of truth arrays.
        for region, hashes in plan["input_sha256"].items():
            for name, expected in hashes.items():
                if sha256(args.cache / region / name) != expected:
                    raise ValueError("frozen DEV input hash mismatch")
        report["decoder_sha256"] = sha256(args.weights)
        report["backbone_sha256"] = sha256(args.backbone)
        if report["backbone_sha256"] != plan["backbone_sha256"]:
            raise ValueError("frozen backbone mismatch")
        import torch

        from terradelta.features import footprint_instances
        from terradelta.pretrained_spatial import load_model
        from terradelta.spatial import predict_scene

        torch.set_num_threads(2)
        checkpoint = torch.load(args.weights, map_location="cpu", weights_only=True)
        training = checkpoint["training"]
        if args.fit_recovery_receipt is not None:
            report["fit_completion"] = validate_fit_recovery(args, checkpoint, plan)
        else:
            if (
                training["steps_completed"] != 4000
                or training["requested_steps"] != 4000
                or training.get("stop_reason") != "requested_steps"
                or training["config_sha256"] != plan["config_sha256"]
            ):
                raise ValueError("decoder is not the completed frozen4000-step DEV-selected fit")
            report["fit_completion"] = {"mode": "normal_terminal_receipt"}
        report["checkpoint_training"] = {
            key: training.get(key)
            for key in (
                "selected_step",
                "steps_completed",
                "requested_steps",
                "config_sha256",
                "development_pixel_metrics",
                "source_sha256",
            )
        }
        report["torch"] = str(torch.__version__)
        if args.device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS unavailable; no silent CPU fallback")
        report["device"] = args.device
        model = load_model(args.weights, backbone_path=args.backbone, device=args.device)
        if model.native_alignment != 16:
            raise ValueError("requires native16 alignment for frozen2x ResNet pooling phase")
        report["native_alignment"] = model.native_alignment
        save()
        for tile in TILES:
            for scene in plan["scenes"]:
                directory = args.cache / scene["region_id"]
                index = scene["frame_index"]
                arrays = {
                    name: np.load(directory / f"{name}.npy", mmap_mode="r")
                    for name in ("rgb", "valid", "instances", "target", "edge")
                }
                rgb, valid, truth = (arrays[name][index] for name in ("rgb", "valid", "instances"))
                if list(truth.shape) != scene["native_shape"] or any(
                    arrays[name][index].shape != truth.shape for name in ("valid", "target", "edge")
                ):
                    raise ValueError("native scene grids differ from the frozen plan")
                tick = time.monotonic()
                probability = predict_scene(model, rgb, device=args.device, tile_size=tile)
                if args.device == "mps":
                    torch.mps.synchronize()
                prediction_seconds = time.monotonic() - tick
                if probability.shape != (2, *truth.shape) or not np.isfinite(probability).all():
                    raise ValueError("invalid native probability output")
                for threshold in THRESHOLDS:
                    tick = time.monotonic()
                    predicted = footprint_instances(probability, valid, threshold=threshold)
                    row = dict(
                        scene,
                        tile_size=tile,
                        threshold=threshold,
                        **instance_metrics(predicted, truth, valid),
                        footprint_dice=dice(
                            probability[0] >= threshold, arrays["target"][index], valid
                        ),
                        boundary_dice=dice(probability[1] >= 0.5, arrays["edge"][index], valid),
                        prediction_seconds=prediction_seconds,
                        candidate_and_scoring_seconds=time.monotonic() - tick,
                        source_valid_fraction=float(valid.mean()),
                    )
                    rows.append(row)
                report["complete_scene_tile_predictions"] += 1
                save()
                print(
                    json.dumps(
                        {
                            "tile_size": tile,
                            "region_id": scene["region_id"],
                            "frame_index": index,
                            "prediction_seconds": prediction_seconds,
                            "completed": report["complete_scene_tile_predictions"],
                        }
                    ),
                    flush=True,
                )
                if args.device == "mps":
                    torch.mps.empty_cache()
        if (
            source_hashes() != plan["source_sha256"]
            or sha256(args.weights) != report["decoder_sha256"]
        ):
            raise ValueError("source or decoder mutated during selection")
        if (
            sha256(args.config) != plan["config_sha256"]
            or sha256(args.backbone) != plan["backbone_sha256"]
        ):
            raise ValueError("configuration or backbone mutated during selection")
        if args.fit_recovery_receipt is not None and (
            sha256(args.fit_recovery_receipt) != report["fit_completion"]["receipt_sha256"]
        ):
            raise ValueError("fit recovery receipt mutated during selection")
        for region, hashes in plan["input_sha256"].items():
            for name, expected in hashes.items():
                if sha256(args.cache / region / name) != expected:
                    raise ValueError("frozen DEV input mutated during selection")
        summaries = summarize(rows, plan["development_regions"])
        best = max(
            summaries,
            key=lambda s: (
                s["global_mean_regional_f1"],
                s["global_mean_regional_recall"],
                -s["total_false_positives"],
                -s["tile_size"],
                -s["threshold"],
            ),
        )
        report.update(status="complete", summaries=summaries, selected=best)
    except SelectionDeadline as exc:
        report.update(status="incomplete_wall_limit", error=str(exc), selected=None)
    except Exception as exc:
        report.update(
            status="incomplete_error", error=f"{type(exc).__name__}: {exc}", selected=None
        )
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        save()
        atomic_json(args.output / "selection.json", report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "selected": report["selected"],
                "elapsed_seconds": report["elapsed_seconds"],
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "run"])
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["mps", "cpu"], default="mps")
    parser.add_argument("--fit-completed", action="store_true")
    parser.add_argument("--fit-recovery-receipt", type=Path,
                        help="Independent clear receipt for the exact preserved fit artifacts")
    args = parser.parse_args()
    prepare(args) if args.mode == "prepare" else run(args)


if __name__ == "__main__":
    main()

"""Compare fixed temporal methods on complete geographic sequences.

Confirmation labels/features are not opened before a matching frozen artifact
manifest is supplied. This script never fits models, thresholds or calibration.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

from terradelta.data import sha256
from terradelta.evaluation import (
    evaluate_events,
    ground_truth_events,
    ground_truth_audit,
    summarize_regions,
)
from terradelta.pipeline import method_events
from terradelta.protocol import validate_experiment

DEFAULT_METHODS = [
    "pairwise",
    "spatial_tracking",
    "smoothing",
    "hmm",
    "ted",
    "sn7_retrospective",
    "observation",
    "no_quality",
    "watch_lstm",
    "proposal_tracking",
    "observation_gate",
    "gate_no_quality",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--event-gate-weights", type=Path)
    parser.add_argument("--spatial-weights", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--split", choices=["train", "development", "confirmation"], default="development"
    )
    parser.add_argument("--freeze-manifest", type=Path)
    parser.add_argument("--embedding-root", type=Path)
    parser.add_argument("--methods", nargs="+")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("evaluation output must be new/empty; preserve earlier runs")
    config = json.loads(args.config.read_text())
    args.methods = args.methods or [
        *DEFAULT_METHODS, *(["embedding_distance"] if config.get("embedding_baseline") else []),
    ]
    from terradelta.pipeline import METHODS, OPTIONAL_METHODS

    if any(method not in (*METHODS, *OPTIONAL_METHODS) for method in args.methods):
        raise ValueError("Unknown comparison method")
    embedding_data = {}
    if "embedding_distance" in args.methods:
        from terradelta.embeddings import load_embedding_data

        if args.embedding_root is None or config.get("embedding_baseline") is None:
            raise ValueError("Embedding comparison requires its source-bound sidecars and config")
        for region in config[args.split]:
            embedding_data[region] = load_embedding_data(
                args.embedding_root / region, feature_dir=args.features_root / region,
                cache_dir=args.cache_root / region, expected_plan=config["embedding_baseline"],
                expected_config_sha256=sha256(args.config),
            )
    freeze = (
        validate_experiment(
            args.freeze_manifest,
            repo=Path(__file__).resolve().parents[1],
            features_root=args.features_root,
            cache_root=args.cache_root,
            regions=config["confirmation"],
            temporal_weights=args.weights,
            event_gate_weights=args.event_gate_weights,
            spatial_weights=args.spatial_weights,
            calibration=args.calibration,
            config=args.config,
            embedding_root=args.embedding_root,
        )
        if args.split == "confirmation"
        else None
    )
    calibration = json.loads(args.calibration.read_text())
    # All confirmation access begins below this point, after immutable hash checks.
    per_method = {method: {} for method in args.methods}
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = {
        "split": args.split,
        "regions": config[args.split],
        "methods": args.methods,
        "temporal_weights_sha256": sha256(args.weights),
        "event_gate_weights_sha256": sha256(args.event_gate_weights)
        if args.event_gate_weights
        else None,
        "calibration_sha256": sha256(args.calibration),
        "config_sha256": sha256(args.config),
        "evaluation_sha256": sha256(
            Path(__file__).resolve().parents[1] / "src/terradelta/evaluation.py"
        ),
        "freeze_manifest": freeze,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "regions_inputs": {},
    }
    for region in config[args.split]:
        features = args.features_root / region
        cache = args.cache_root / region
        info = json.loads((features / "sites.json").read_text())
        feature_data = np.load(features / "features.npz")
        metadata = json.loads((cache / "metadata.json").read_text())
        if info["region_id"] != region or metadata["region_id"] != region:
            raise ValueError("region identities disagree with split selection")
        metadata["candidate_geometries"] = info["geometries"]
        truth = ground_truth_events(cache)
        audit = ground_truth_audit(cache)
        (args.output / f"{region}-truth-audit.json").write_text(
            json.dumps(audit, indent=2, allow_nan=False) + "\n"
        )
        provenance["regions_inputs"][region] = {
            "features_sha256": sha256(features / "features.npz"),
            "sites_sha256": sha256(features / "sites.json"),
            "instances_sha256": sha256(cache / "instances.npy"),
            "valid_sha256": sha256(cache / "valid.npy"),
            "metadata_sha256": sha256(cache / "metadata.json"),
        }
        for method in args.methods:
            started = time.perf_counter()
            events = method_events(
                feature_data["x"],
                feature_data["usable"],
                feature_data["births"],
                info,
                method,
                temporal_weights=args.weights,
                calibration=calibration,
                event_gate_weights=args.event_gate_weights,
                model_revision=f"evaluation:{sha256(args.weights)[:12]}:{method}",
                embedding_data=embedding_data.get(region) if method == "embedding_distance" else None,
            )
            result = evaluate_events(events, truth, metadata)
            result["runtime_seconds_temporal_decode_evaluation"] = time.perf_counter() - started
            result["decoder"] = calibration.get("event_gates", {}).get(
                method, calibration.get("decoders", {}).get(method, {})
            )
            result["method"] = method
            per_method[method][region] = result
            folder = args.output / method
            folder.mkdir(exist_ok=True)
            (folder / f"{region}.json").write_text(
                json.dumps(result, indent=2, allow_nan=False) + "\n"
            )
            print(method, region, result["event"], result["within_62_days"], flush=True)
    summaries = {
        method: summarize_regions(regions, seed=config.get("seed", 2901))
        for method, regions in per_method.items()
    }
    report = {
        "provenance": provenance,
        "methods": summaries,
        "comparison_basis": "Shared native imagery, labels, candidate sites and spatial predictions; fixed 62-day detection horizon plus measured delay.",
        "claim_limit": "A controlled local temporal adaptation comparison, not reproduction of published full-system scores or physical event-date accuracy.",
    }
    (args.output / "comparison.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()

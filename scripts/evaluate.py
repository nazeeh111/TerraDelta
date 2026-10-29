"""Compare fixed temporal methods on complete geographic sequences.

Confirmation labels/features are not opened before a matching frozen artifact
manifest is supplied. This script never fits models, thresholds or calibration.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

from terradelta.data import sha256
from terradelta.decoder import DecoderConfig, decode_events
from terradelta.evaluation import evaluate_events, ground_truth_events, summarize_regions
from terradelta.pipeline import method_scores
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
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--spatial-weights", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--split", choices=["train", "development", "confirmation"], default="development"
    )
    parser.add_argument("--freeze-manifest", type=Path)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("evaluation output must be new/empty; preserve earlier runs")
    config = json.loads(args.config.read_text())
    freeze = (
        validate_experiment(
            args.freeze_manifest,
            repo=Path(__file__).resolve().parents[1],
            features_root=args.features_root,
            cache_root=args.cache_root,
            regions=config["confirmation"],
            temporal_weights=args.weights,
            spatial_weights=args.spatial_weights,
            calibration=args.calibration,
            config=args.config,
        )
        if args.split == "confirmation"
        else None
    )
    calibration = json.loads(args.calibration.read_text())
    occupancy_calibration = calibration.get("occupancy", calibration.get("calibration", {}))
    # All confirmation access begins below this point, after immutable hash checks.
    per_method = {method: {} for method in args.methods}
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = {
        "split": args.split,
        "regions": config[args.split],
        "methods": args.methods,
        "temporal_weights_sha256": sha256(args.weights),
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
        provenance["regions_inputs"][region] = {
            "features_sha256": sha256(features / "features.npz"),
            "sites_sha256": sha256(features / "sites.json"),
            "instances_sha256": sha256(cache / "instances.npy"),
            "valid_sha256": sha256(cache / "valid.npy"),
            "metadata_sha256": sha256(cache / "metadata.json"),
        }
        for method in args.methods:
            started = time.perf_counter()
            scores = method_scores(
                feature_data["x"],
                feature_data["usable"],
                method,
                args.weights,
                occupancy_calibration,
            )
            settings = DecoderConfig(
                **calibration.get("decoders", {}).get(method, calibration.get("decoder", {}))
            )
            events = decode_events(
                scores,
                feature_data["usable"],
                feature_data["births"],
                info,
                settings,
                model_revision=f"evaluation:{sha256(args.weights)[:12]}:{method}",
                retrospective=method == "sn7_retrospective",
            )
            result = evaluate_events(events, truth, metadata)
            result["runtime_seconds_temporal_decode_evaluation"] = time.perf_counter() - started
            result["decoder"] = asdict(settings)
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

"""Choose global development operating points; freeze before confirmation access."""

from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from scipy.special import expit, logit

from terradelta.data import sha256
from terradelta.decoder import DecoderConfig, decode_events
from terradelta.evaluation import ground_truth_events, evaluate_events, match_events
from terradelta.pipeline import method_scores
from terradelta.protocol import freeze_experiment
from train_temporal import calibrate

METHODS = [
    "pairwise",
    "spatial_tracking",
    "ted",
    "smoothing",
    "hmm",
    "watch_lstm",
    "sn7_retrospective",
    "no_quality",
    "observation",
]


def main():
    parser = argparse.ArgumentParser()
    for arg in [
        "features-root",
        "cache-root",
        "weights",
        "spatial-weights",
        "config",
        "output",
        "freeze-output",
    ]:
        parser.add_argument("--" + arg, type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    development = []
    for aoi in config["development"]:
        root = args.features_root / aoi
        f = np.load(root / "features.npz")
        s = np.load(root / "supervision.npz")
        info = json.loads((root / "sites.json").read_text())
        truth = ground_truth_events(args.cache_root / aoi)
        development.append((aoi, f, s, info, truth))
    occupancy = {}
    decoders = {}
    selection = {}
    timing = {}
    for method in METHODS:
        raw = []
        ys = []
        masks = []
        regional_scores = []
        for _, f, s, _, _ in development:
            scores = method_scores(f["x"], f["usable"], method, args.weights)
            regional_scores.append(scores)
            raw.append(scores.ravel())
            ys.append(s["y"].ravel())
            masks.append((f["usable"] & s["supervised"]).ravel())
        occupancy[method] = calibrate(
            np.concatenate(raw), np.concatenate(ys), np.concatenate(masks)
        )
        transform = occupancy[method]
        regional_scores = [
            expit(transform["scale"] * logit(np.clip(p, 0.001, 0.999)) + transform["bias"])
            for p in regional_scores
        ]
        candidates = []
        # All thresholds are shared across regions. Original unsmoothed pairwise
        # baseline has immediate detection; other methods use the same dwell policy.
        for threshold in [0.55, 0.65, 0.75, 0.85]:
            settings = DecoderConfig(
                threshold=threshold,
                confirmations=1 if method == "pairwise" else 2,
                minimum_dwell_days=0 if method == "pairwise" else 20,
            )
            reports = []
            all_events = []
            for (aoi, f, s, info, truth), score in zip(development, regional_scores):
                events = decode_events(
                    score,
                    f["usable"],
                    f["births"],
                    info,
                    settings,
                    model_revision="development-selection",
                    retrospective=method == "sn7_retrospective",
                )
                report = evaluate_events(events, truth, info["metadata"])
                reports.append(report)
                all_events.append(events)
            timely_f1 = float(np.mean([r["within_62_days"]["f1"] for r in reports]))
            ordinary_f1 = float(np.mean([r["event"]["f1"] for r in reports]))
            # Fixed62-day horizon measures precision/recall at the same delay budget.
            # Retrospective models are also reported, but cannot claim an online win.
            value = timely_f1 if method != "sn7_retrospective" else ordinary_f1
            candidates.append((value, ordinary_f1, threshold, settings, reports, all_events))
            print(
                method,
                threshold,
                "development62d_macroF1",
                round(timely_f1, 4),
                "ordinary_macroF1",
                round(ordinary_f1, 4),
                flush=True,
            )
        chosen = max(candidates, key=lambda row: (row[0], row[1], row[2]))
        _, _, _, settings, reports, event_sets = chosen
        # Distribution-free coverage is NOT claimed: label intervals are censored and
        # only two development regions exist. Fit global endpoint padding and report
        # its real confirmation containment, including clipping at detection time.
        residuals = []
        for (_, _, _, _, truth), events in zip(development, event_sets):
            for pred_index, truth_index, _ in match_events(events, truth):
                event = events[pred_index]
                target = truth[truth_index]
                lower = (
                    datetime.fromisoformat(target["lower"])
                    if isinstance(target["lower"], str)
                    else target["lower"]
                )
                upper = (
                    datetime.fromisoformat(target["upper"])
                    if isinstance(target["upper"], str)
                    else target["upper"]
                )
                residuals.append(
                    max(
                        0.0,
                        (event.observation_interval.lower - lower).total_seconds() / 86400,
                        (upper - event.observation_interval.upper).total_seconds() / 86400,
                    )
                )
        padding = int(np.ceil(np.quantile(residuals, 0.9, method="higher"))) if residuals else 0
        decoders[method] = {**asdict(settings), "timing_padding_days": padding}
        selection[method] = {
            "candidate_settings": [
                {"threshold": c[2], "objective": c[0], "macro_event_f1": c[1]} for c in candidates
            ],
            "selected_development_reports": reports,
            "objective": "mean regional F1 within62days; retrospective uses ordinary eventF1",
        }
        timing[method] = {
            "target_empirical_containment": 0.9,
            "matched_development_events": len(residuals),
            "padding_days": padding,
            "interpretation": "empirical censored-label endpoint padding, not guaranteed physical-date coverage under geographic shift",
        }
    payload = {
        "version": "terradelta-calibration-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "development_regions": config["development"],
        "occupancy": occupancy,
        "decoders": decoders,
        "decoder": decoders["observation"],
        "selection": selection,
        "timing_calibration": timing,
        "confirmation_untouched": True,
        "matched_detection_horizon_days": 62,
        "weights_sha256": sha256(args.weights),
        "spatial_weights_sha256": sha256(args.spatial_weights),
        "config_sha256": sha256(args.config),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    freeze_experiment(
        args.freeze_output,
        repo=Path(__file__).resolve().parents[1],
        features_root=args.features_root,
        cache_root=args.cache_root,
        regions=config["confirmation"],
        temporal_weights=args.weights,
        spatial_weights=args.spatial_weights,
        calibration=args.output,
        config=args.config,
    )


if __name__ == "__main__":
    main()

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
from terradelta.features import validate_feature_protocol
from terradelta.event_gate import generate_events, load_gate
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
    parser.add_argument("--event-gate-weights", type=Path)
    parser.add_argument(
        "--skip-freeze",
        action="store_true",
        help="Fit development only; freeze separately after source review",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Calibration output exists; use a new versioned path")
    config = json.loads(args.config.read_text())
    from terradelta.contracts import (
        preflight_heads, preflight_manifests, preflight_spatial_checkpoint, study_binding,
    )

    binding = study_binding(config)
    if config.get("event_gate") and args.event_gate_weights is None:
        raise ValueError("Study requires --event-gate-weights")
    if binding is not None:
        preflight_manifests(
            args.features_root, args.cache_root, config["development"], binding, supervision=True,
        )
        preflight_spatial_checkpoint(args.spatial_weights, binding)
        preflight_heads(
            binding, temporal_weights=args.weights, event_gate_weights=args.event_gate_weights,
            temporal_kinds=("observation", "no_quality", "watch_lstm"), require_bound=True,
            config_sha256=sha256(args.config),
        )
    development = []
    for aoi in config["development"]:
        root = args.features_root / aoi
        if config.get("truth_protocol") == "physical_coverage_v2":
            validate_feature_protocol(root, supervision=True)
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
        for _, f, s, info, _ in development:
            scores = method_scores(f["x"], f["usable"], method, args.weights, spatial_info=info)
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
    gates = {}
    if args.event_gate_weights:
        bundle = load_gate(args.event_gate_weights)
        for kind in ["proposal_tracking", "observation_gate", "gate_no_quality"]:
            params = {} if kind == "proposal_tracking" else dict(bundle.calibration[kind])
            candidates = []
            for threshold in (
                [0.55] if kind == "proposal_tracking" else [0.02, 0.05, 0.1, 0.2, 0.35, 0.5]
            ):
                operating = {**params, "threshold": threshold, "timing_padding_days": 0}
                reports, event_sets = [], []
                for _, f, _, info, truth in development:
                    events = generate_events(
                        f["x"],
                        f["usable"],
                        f["births"],
                        info,
                        weights=bundle,
                        kind=kind,
                        calibration={kind: operating},
                    )
                    reports.append(evaluate_events(events, truth, info["metadata"]))
                    event_sets.append(events)
                timely = float(np.mean([r["within_62_days"]["f1"] for r in reports]))
                ordinary = float(np.mean([r["event"]["f1"] for r in reports]))
                candidates.append((timely, ordinary, threshold, reports, event_sets))
                print(
                    kind,
                    threshold,
                    "development62d_macroF1",
                    round(timely, 4),
                    "ordinary_macroF1",
                    round(ordinary, 4),
                    flush=True,
                )
            chosen = max(candidates, key=lambda row: (row[0], row[1], row[2]))
            residuals = []
            for (_, _, _, _, truth), events in zip(development, chosen[4]):
                for pi, ti, _ in match_events(events, truth):
                    event, target = events[pi], truth[ti]
                    lower, upper = [
                        datetime.fromisoformat(target[k])
                        if isinstance(target[k], str)
                        else target[k]
                        for k in ["lower", "upper"]
                    ]
                    residuals.append(
                        max(
                            0.0,
                            (event.observation_interval.lower - lower).total_seconds() / 86400,
                            (upper - event.observation_interval.upper).total_seconds() / 86400,
                        )
                    )
            padding = int(np.ceil(np.quantile(residuals, 0.9, method="higher"))) if residuals else 0
            gates[kind] = {**params, "threshold": chosen[2], "timing_padding_days": padding}
            selection[kind] = {
                "candidate_settings": [
                    {"threshold": c[2], "objective": c[0], "macro_event_f1": c[1]}
                    for c in candidates
                ],
                "selected_development_reports": chosen[3],
                "objective": "mean regional F1 within62days",
            }
            timing[kind] = {
                "target_empirical_containment": 0.9,
                "matched_development_events": len(residuals),
                "padding_days": padding,
                "interpretation": "empirical censored-label endpoint padding; no geographic-shift guarantee",
            }
    payload = {
        "version": "terradelta-calibration-v2",
        "event_gates": gates,
        "event_gate_weights_sha256": sha256(args.event_gate_weights)
        if args.event_gate_weights
        else None,
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
        "spatial_provenance": config.get("event_gate", {}).get("spatial_provenance"),
        "config_sha256": sha256(args.config),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    if args.skip_freeze:
        return
    freeze_experiment(
        args.freeze_output,
        repo=Path(__file__).resolve().parents[1],
        features_root=args.features_root,
        cache_root=args.cache_root,
        regions=config["confirmation"],
        temporal_weights=args.weights,
        event_gate_weights=args.event_gate_weights,
        spatial_weights=args.spatial_weights,
        calibration=args.output,
        config=args.config,
    )


if __name__ == "__main__":
    main()

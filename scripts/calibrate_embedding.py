"""Add one globally selected frozen-neural-descriptor comparator using DEV only."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from scipy.special import expit, logit

from terradelta.contracts import (
    assert_calibration, head_training_config_sha256, preflight_manifests, study_binding,
)
from terradelta.data import sha256
from terradelta.decoder import DecoderConfig, decode_events
from terradelta.embeddings import descriptor_plan, distance_gated_scores, load_embedding_data
from terradelta.evaluation import evaluate_events, ground_truth_events, match_events
from train_temporal import calibrate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "features-root", "cache-root", "embedding-root",
                 "base-calibration", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Calibration is immutable; use a new output path")
    config = json.loads(args.config.read_text())
    from terradelta.protocol import PROTOCOL_FILES

    repo = Path(__file__).resolve().parents[1]
    source_hashes = {name: sha256(repo / name) for name in PROTOCOL_FILES}
    plan = config.get("embedding_baseline")
    if plan != descriptor_plan():
        raise ValueError("Study must declare the fixed neural descriptor and DEV grid")
    if any(set(config[a]) & set(config[b]) for a, b in (
        ("train", "development"), ("train", "confirmation"), ("development", "confirmation"),
    )):
        raise ValueError("Geographic split overlap")
    payload = json.loads(args.base_calibration.read_text())
    if "embedding_distance" in payload:
        raise ValueError("Preserve the previously selected neural comparator")
    binding = study_binding(config)
    assert_calibration(binding, payload, require_bound=True, config_sha256=sha256(args.config))
    if payload.get("head_training_config_sha256") != head_training_config_sha256(args.config, config):
        raise ValueError("Base calibration head-training identity differs from the actual fit")
    infos = preflight_manifests(
        args.features_root, args.cache_root, config["development"], binding, supervision=True,
    )
    sidecars = {}
    # Reject mixed source/model/site identities for ALL DEV regions before labels.
    for region in config["development"]:
        sidecars[region] = load_embedding_data(
            args.embedding_root / region, feature_dir=args.features_root / region,
            cache_dir=args.cache_root / region, expected_plan=plan,
            expected_config_sha256=sha256(args.config),
        )
    development = []
    for region in config["development"]:
        with np.load(args.features_root / region / "features.npz") as f:
            x, usable, births = f["x"], f["usable"], f["births"]
        with np.load(args.features_root / region / "supervision.npz") as s:
            labels, supervised = s["y"], s["supervised"]
        truth = ground_truth_events(args.cache_root / region)
        development.append((region, x, usable, births, labels, supervised,
                            infos[region], truth, sidecars[region]))
    candidates, chosen, best_key = [], None, None
    for distance in plan["distance_threshold_grid"]:
        raw, effective = [], []
        for _, x, usable, births, _, _, _, _, sidecar in development:
            scores, available, _ = distance_gated_scores(
                x[:, :, 0], sidecar["vectors"], usable & sidecar["usable"], births,
                distance, lookback=plan["reference_lookback"],
            )
            raw.append(scores)
            effective.append(available)
        fit = calibrate(
            np.concatenate([p.ravel() for p in raw]),
            np.concatenate([d[4].ravel() for d in development]),
            np.concatenate([(u & d[5]).ravel() for u, d in zip(effective, development)]),
        )
        calibrated = [expit(fit["scale"] * logit(np.clip(p, .001, .999)) + fit["bias"])
                      for p in raw]
        for threshold in plan["decoder_threshold_grid"]:
            settings = DecoderConfig(threshold=threshold, confirmations=2, minimum_dwell_days=20)
            events, reports = [], []
            for data, score, available in zip(development, calibrated, effective):
                region, _, _, births, _, _, info, truth, _ = data
                event = decode_events(score, available, births, info, settings,
                                      model_revision="development-neural-embedding-selection")
                result = evaluate_events(event, truth, info["metadata"])
                events.append(event)
                reports.append({"region_id": region, "event": result["event"],
                                "within_62_days": result["within_62_days"]})
            timely = float(np.mean([r["within_62_days"]["f1"] or 0 for r in reports]))
            ordinary = float(np.mean([r["event"]["f1"] or 0 for r in reports]))
            row = {"distance_threshold": distance, "decoder_threshold": threshold,
                   "macro_f1_62d": timely, "macro_event_f1": ordinary,
                   "region_reports": reports}
            candidates.append(row)
            key = (timely, ordinary, -distance, -threshold)
            if best_key is None or key > best_key:
                best_key = key
                chosen = (distance, fit, settings, events, reports)
            print(json.dumps(row), flush=True)
    distance, fit, settings, event_sets, reports = chosen
    residuals = []
    for data, events in zip(development, event_sets):
        truth = data[7]
        for prediction, target, _ in match_events(events, truth):
            interval, reference = events[prediction].observation_interval, truth[target]
            lower, upper = reference["lower"], reference["upper"]
            lower = datetime.fromisoformat(lower) if isinstance(lower, str) else lower
            upper = datetime.fromisoformat(upper) if isinstance(upper, str) else upper
            residuals.append(max(0., (interval.lower-lower).total_seconds()/86400,
                                 (upper-interval.upper).total_seconds()/86400))
    padding = int(np.ceil(np.quantile(residuals, .9, method="higher"))) if residuals else 0
    payload["version"] = "terradelta-calibration-v3-neural-comparator"
    payload["created_at"] = datetime.now(timezone.utc).isoformat()
    payload["base_calibration_sha256"] = sha256(args.base_calibration)
    payload["embedding_distance"] = {
        "descriptor": plan, "distance_threshold": distance,
        "reference_lookback": plan["reference_lookback"],
        "development_inputs": {
            region: {"manifest_sha256": sha256(args.embedding_root / region / "embedding.json"),
                     "embeddings_sha256": sha256(args.embedding_root / region / "embeddings.npz")}
            for region in config["development"]
        },
        "selection_source_sha256": source_hashes["scripts/calibrate_embedding.py"],
        "selection_sources_sha256": source_hashes,
    }
    payload["occupancy"]["embedding_distance"] = fit
    payload["decoders"]["embedding_distance"] = {
        **asdict(settings), "timing_padding_days": padding,
    }
    payload["selection"]["embedding_distance"] = {
        "candidate_settings": candidates, "selected_development_reports": reports,
        "objective": plan["selection"],
    }
    payload["timing_calibration"]["embedding_distance"] = {
        "matched_development_events": len(residuals), "padding_days": padding,
        "interpretation": "empirical censored-label endpoint padding; no geographic-shift guarantee",
    }
    assert_calibration(binding, payload, require_bound=True, config_sha256=sha256(args.config))
    if {name: sha256(repo / name) for name in PROTOCOL_FILES} != source_hashes:
        raise ValueError("Calibration source changed during neural comparator selection")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()

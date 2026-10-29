"""Evaluate frozen event-compatibility scores on every held-out raw proposal.

No fitting occurs. This uses the same up-to-four available-prefix evaluation
sampling as development calibration, with unit total mass per raw proposal.
"""

from datetime import datetime, timezone
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import expit
import torch

from terradelta.event_gate import (
    event_features,
    load_gate,
    raw_proposals,
    standardize,
    support_indices,
)
from terradelta.evaluation import ground_truth_events, match_events
from terradelta.features import validate_feature_protocol
from terradelta.protocol import validate_experiment


def reliability(probability, labels, mass):
    weights = mass / mass.sum() if mass.sum() else mass
    bins = []
    ece = 0.0
    for index in range(10):
        selected = (probability >= index / 10) & (probability < (index + 1) / 10)
        if index == 9:
            selected |= probability == 1
        if selected.any():
            total = weights[selected].sum()
            predicted = float(np.dot(weights[selected], probability[selected]) / total)
            actual = float(np.dot(weights[selected], labels[selected]) / total)
            ece += total * abs(predicted - actual)
            bins.append(
                dict(
                    lower=index / 10,
                    rows=int(selected.sum()),
                    proposal_mass=float(mass[selected].sum()),
                    mean_score=predicted,
                    positive_fraction=actual,
                )
            )
    prevalence = float(np.dot(weights, labels)) if len(labels) else None
    return dict(
        prefix_rows=len(labels),
        proposal_mass=float(mass.sum()),
        positive_fraction=prevalence,
        brier=float(np.dot(weights, (probability - labels) ** 2)) if len(labels) else None,
        constant_prevalence_oracle_brier=prevalence * (1 - prevalence)
        if prevalence is not None
        else None,
        expected_calibration_error=float(ece),
        bins=bins,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in [
        "features-root",
        "cache-root",
        "weights",
        "spatial-weights",
        "event-gate-weights",
        "calibration",
        "config",
        "freeze-manifest",
        "output",
    ]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config.get("truth_protocol") != "physical_coverage_v2":
        raise ValueError("Requires physical_coverage_v2 truth")
    freeze = validate_experiment(
        args.freeze_manifest,
        repo=Path(__file__).resolve().parents[1],
        features_root=args.features_root,
        cache_root=args.cache_root,
        regions=config["confirmation"],
        temporal_weights=args.weights,
        spatial_weights=args.spatial_weights,
        event_gate_weights=args.event_gate_weights,
        calibration=args.calibration,
        config=args.config,
    )
    if args.output.exists():
        raise ValueError("Output exists; preserve prior results")
    torch.set_num_threads(2)
    bundle = load_gate(args.event_gate_weights)
    calibration = json.loads(args.calibration.read_text())["event_gates"]
    results = {}
    for region in config["confirmation"]:
        root = args.features_root / region
        info = validate_feature_protocol(root, supervision=False)
        if info.get("spatial_weights_sha256") != bundle.spatial_weights_sha256:
            raise ValueError("Spatial checkpoint differs from event-gate training provenance")
        with np.load(root / "features.npz") as f:
            x, usable, births = f["x"], f["usable"], f["births"]
        raw = raw_proposals(
            x, usable, births, info, record_origin=datetime(2026, 1, 1, tzinfo=timezone.utc)
        )
        truth = ground_truth_events(args.cache_root / region, protocol="physical_coverage_v2")
        positives = {raw[p].event_id for p, _, _ in match_events(raw, truth)}
        first, retractions = {}, {}
        for event in raw:
            if event.state == "confirmed_by_model":
                first.setdefault(event.event_id, event)
            if event.state == "retracted":
                retractions.setdefault(event.event_id, support_indices(event)[3])
        rows, labels, mass = [], [], []
        for event_id, event in first.items():
            site, _, _, current = support_indices(event)
            indices = np.flatnonzero(usable[site - 1, current:]) + current
            indices = indices[indices < retractions.get(event_id, x.shape[1])][:4]
            for index in indices:
                row, _ = event_features(x, usable, event, int(index), info)
                rows.append(row)
                labels.append(int(event_id in positives))
                mass.append(1 / len(indices))
        labels, mass = np.asarray(labels), np.asarray(mass)
        regional = {}
        for kind in ["observation_gate", "gate_no_quality"]:
            if rows:
                inputs = standardize(np.asarray(rows), bundle.mean, bundle.std, kind)
                with torch.inference_mode():
                    logits = bundle.models[kind](torch.from_numpy(inputs))[:, 0].numpy()
                params = calibration[kind]
                probability = expit(params["scale"] * logits + params["bias"])
            else:
                probability = np.array([])
            regional[kind] = reliability(probability, labels, mass)
        results[region] = regional
        print(region, {k: v["brier"] for k, v in regional.items()}, flush=True)
    args.output.write_text(
        json.dumps(
            dict(
                freeze_manifest=freeze,
                regions=results,
                target="One-to-one physical geometry/time compatibility of raw first-confirmed proposals; not construction-date coverage",
                sampling="At most four available prefixes per proposal, stopping at observed retraction; each proposal totals one",
                limits="This evaluates the candidate distribution, not unknown objects omitted by spatial proposals. Three regions do not guarantee arbitrary-shift calibration.",
            ),
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

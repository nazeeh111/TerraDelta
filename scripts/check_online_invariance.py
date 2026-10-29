"""Real-image candidate-prefix and trained temporal/event-gate future-edit checks."""

from copy import deepcopy
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path

import numpy as np

from terradelta.data import load_region, sha256
from terradelta.features import candidate_sites, feature_cube, validate_feature_protocol
from terradelta.pipeline import method_events, method_scores

OCCUPANCY_METHODS = (
    "pairwise",
    "spatial_tracking",
    "ted",
    "smoothing",
    "hmm",
    "watch_lstm",
    "no_quality",
    "observation",
)
GATE_METHODS = ("proposal_tracking", "observation_gate", "gate_no_quality")


def historical_revisions(events, cutoff):
    """Compare complete historical revisions using a shared transaction origin."""
    result = []
    for event in events:
        if event.detected_at <= cutoff:
            revision = event.model_dump(mode="json")
            result.append(revision)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in [
        "cache-root",
        "features-root",
        "predictions-root",
        "weights",
        "calibration",
        "config",
        "output",
    ]:
        parser.add_argument("--" + arg, type=Path, required=True)
    parser.add_argument("--event-gate-weights", type=Path)
    parser.add_argument("--methods", nargs="+", choices=OCCUPANCY_METHODS + GATE_METHODS)
    args = parser.parse_args()
    methods_to_check = args.methods or list(OCCUPANCY_METHODS) + ["proposal_tracking"] + (
        list(GATE_METHODS[1:]) if args.event_gate_weights else []
    )
    if (
        any(method in GATE_METHODS[1:] for method in methods_to_check)
        and not args.event_gate_weights
    ):
        parser.error("learned event-gate checks require --event-gate-weights")
    if args.output.exists():
        raise ValueError("Invariance output must be new; preserve earlier checks")
    config = json.loads(args.config.read_text())
    if config.get("event_gate") and not args.event_gate_weights:
        raise ValueError("Event-gate experiment requires --event-gate-weights")
    cal = json.loads(args.calibration.read_text())
    report = {}
    for region_id in config["confirmation"]:
        region = load_region(args.cache_root / region_id)  # no labels or performance
        feature_dir = args.features_root / region_id
        info = validate_feature_protocol(feature_dir, supervision=False)
        full = np.load(feature_dir / "features.npz")
        probabilities = np.load(
            args.predictions_root / region_id / "probabilities.npy", mmap_mode="r"
        )
        cutoff = min(12, len(probabilities) - 1)
        if cutoff < 1:
            raise ValueError("Invariance checks require a prefix and at least one future frame")
        goodframes = np.array(
            [f["registration"]["usable"] for f in info["metadata"]["frames"][:cutoff]]
        )
        sites, births = candidate_sites(
            probabilities[:cutoff],
            region["valid"][:cutoff],
            goodframes,
            threshold=info["candidate_threshold"],
        )
        fullsites = np.load(feature_dir / "sites.npy")
        np.testing.assert_array_equal(np.where(fullsites <= len(births), fullsites, 0), sites)
        np.testing.assert_array_equal(full["births"][: len(births)], births)
        prefix_info = deepcopy(info)
        prefix_info["metadata"]["frames"] = prefix_info["metadata"]["frames"][:cutoff]
        prefix_info["geometries"] = {
            k: v for k, v in info["geometries"].items() if int(k) <= len(births)
        }
        x, usable = feature_cube(
            probabilities[:cutoff],
            region["rgb"][:cutoff],
            region["valid"][:cutoff],
            prefix_info["metadata"],
            sites,
        )
        np.testing.assert_allclose(x, full["x"][: len(births), :cutoff], atol=1e-6, rtol=1e-6)
        np.testing.assert_array_equal(usable, full["usable"][: len(births), :cutoff])
        altered = full["x"].copy()
        altered[:, cutoff:] = np.flip(altered[:, cutoff:], axis=1)
        altered[:, cutoff:, 0] = 1 - altered[:, cutoff:, 0]
        altered_usable = full["usable"].copy()
        altered_usable[:, cutoff::2] = False
        last = datetime.fromisoformat(prefix_info["metadata"]["frames"][-1]["acquisition_end"])
        methods = {}
        for method in methods_to_check:
            if method in OCCUPANCY_METHODS:
                scores = method_scores(
                    full["x"], full["usable"], method, args.weights, cal["occupancy"]
                )
                short_scores = method_scores(x, usable, method, args.weights, cal["occupancy"])
                np.testing.assert_allclose(
                    scores[: len(births), :cutoff], short_scores, atol=2e-6, rtol=1e-5
                )
                edited_scores = method_scores(
                    altered, altered_usable, method, args.weights, cal["occupancy"]
                )
                np.testing.assert_allclose(
                    scores[:, :cutoff], edited_scores[:, :cutoff], atol=1e-7, rtol=1e-6
                )
            kwargs = dict(
                method=method,
                temporal_weights=args.weights,
                calibration=cal,
                event_gate_weights=args.event_gate_weights,
                model_revision="prefix-check",
                record_origin=datetime(2026, 1, 1, tzinfo=timezone.utc),
            )
            long = method_events(full["x"], full["usable"], full["births"], info, **kwargs)
            short = method_events(x, usable, births, prefix_info, **kwargs)
            edited = method_events(altered, altered_usable, full["births"], info, **kwargs)
            before = historical_revisions(long, last)
            assert before == historical_revisions(short, last), (region_id, method, "prefix")
            assert before == historical_revisions(edited, last), (region_id, method, "future edit")
            methods[method] = {
                "passed": True,
                "prefix_revisions": len(before),
                "occupancy_score_prefix_checked": method in OCCUPANCY_METHODS,
                "event_prefix_checked": True,
                "future_feature_and_mask_edits_checked": True,
            }
        report[region_id] = {
            "prefix_observations": cutoff,
            "prefix_sites": len(births),
            "full_sites": len(full["births"]),
            "methods": methods,
            "features_sha256": sha256(feature_dir / "features.npz"),
        }
        print(region_id, "PASS", flush=True)
    args.output.write_text(
        json.dumps(
            {
                "checks": report,
                "labels_read": False,
                "event_gate_weights_sha256": sha256(args.event_gate_weights)
                if args.event_gate_weights
                else None,
                "scope": "Real native-image candidate/feature prefix; trained temporal occupancy and event-gate revision prefixes; synthetic future feature and availability edits. Spatial CNN is frame independent.",
                "transaction_clock": "Fixed shared origin: complete revisions including record_time must agree.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

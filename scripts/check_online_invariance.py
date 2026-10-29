"""Real-image candidate-prefix and trained temporal future-edit checks."""

from copy import deepcopy
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path

import numpy as np

from terradelta.data import load_region, sha256
from terradelta.decoder import DecoderConfig, decode_events
from terradelta.features import candidate_sites, feature_cube
from terradelta.pipeline import method_scores

parser = argparse.ArgumentParser()
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
args = parser.parse_args()
config = json.loads(args.config.read_text())
cal = json.loads(args.calibration.read_text())
report = {}
for region_id in config["confirmation"]:
    region = load_region(args.cache_root / region_id)  # no labels
    feature_dir = args.features_root / region_id
    info = json.loads((feature_dir / "sites.json").read_text())
    full = np.load(feature_dir / "features.npz")
    probabilities = np.load(args.predictions_root / region_id / "probabilities.npy", mmap_mode="r")
    cutoff = min(12, len(probabilities) - 1)
    goodframes = np.array(
        [f["registration"]["usable"] for f in info["metadata"]["frames"][:cutoff]]
    )
    sites, births = candidate_sites(probabilities[:cutoff], region["valid"][:cutoff], goodframes)
    fullsites = np.load(feature_dir / "sites.npy")
    np.testing.assert_array_equal(np.where(fullsites <= len(births), fullsites, 0), sites)
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
    methods = {}
    for method in [
        "pairwise",
        "spatial_tracking",
        "ted",
        "smoothing",
        "hmm",
        "watch_lstm",
        "no_quality",
        "observation",
    ]:
        scores = method_scores(full["x"], full["usable"], method, args.weights, cal["occupancy"])
        short_scores = method_scores(x, usable, method, args.weights, cal["occupancy"])
        np.testing.assert_allclose(
            scores[: len(births), :cutoff], short_scores, atol=2e-6, rtol=1e-5
        )
        altered = full["x"].copy()
        altered[:, cutoff:] = np.flip(altered[:, cutoff:], axis=1)
        edited_scores = method_scores(
            altered, full["usable"], method, args.weights, cal["occupancy"]
        )
        np.testing.assert_allclose(
            scores[:, :cutoff], edited_scores[:, :cutoff], atol=1e-7, rtol=1e-6
        )
        decoder = DecoderConfig(**cal["decoders"][method])
        origin = datetime(2026, 1, 1, tzinfo=timezone.utc)
        long = decode_events(
            scores,
            full["usable"],
            full["births"],
            info,
            decoder,
            model_revision="prefix-check",
            record_origin=origin,
        )
        short = decode_events(
            short_scores,
            usable,
            births,
            prefix_info,
            decoder,
            model_revision="prefix-check",
            record_origin=origin,
        )
        last = datetime.fromisoformat(prefix_info["metadata"]["frames"][-1]["acquisition_end"])
        before = [e.model_dump(mode="json") for e in long if e.detected_at <= last]
        assert before == [e.model_dump(mode="json") for e in short], (region_id, method)
        methods[method] = {"passed": True, "prefix_revisions": len(before)}
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
            "scope": "Real native-image candidate/feature prefix; trained temporal probabilities and event revisions; admissible future feature edits. Spatial CNN is frame independent.",
        },
        indent=2,
    )
    + "\n"
)

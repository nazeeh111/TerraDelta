"""Frozen confirmation diagnostics; perturbations are labeled controls, not observations."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import shift
import torch

from terradelta.data import load_region, registration_diagnostic
from terradelta.evaluation import evaluate_events, ground_truth_events
from terradelta.features import feature_cube, labels_for_sites, validate_feature_protocol
from terradelta.pipeline import method_events, method_scores
from terradelta.protocol import validate_experiment
from terradelta.spatial import load_model, predict_scene


def reliability(p, y, valid):
    prediction = p[valid]
    truth = y[valid]
    bins = []
    ece = 0.0
    for lo in np.arange(0, 1, 0.1):
        selected = (prediction >= lo) & (prediction < lo + 0.1 + 1e-8)
        if selected.any():
            mean = float(prediction[selected].mean())
            actual = float(truth[selected].mean())
            ece += int(selected.sum()) * abs(mean - actual) / max(len(truth), 1)
            bins.append(
                {
                    "lower": float(lo),
                    "n": int(selected.sum()),
                    "mean_score": mean,
                    "presence_rate": actual,
                }
            )
    return {
        "n": len(truth),
        "brier": float(np.mean((prediction - truth) ** 2)) if len(truth) else None,
        "expected_calibration_error": ece,
        "bins": bins,
        "target": "candidate occupancy coverage>=80percent; <=20percent absent; middle coverage unknown; not event correctness",
    }


def dice(probability, truth, valid):
    predicted = (probability >= 0.45) & valid
    target = truth & valid
    return float(2 * (predicted & target).sum() / max(1, predicted.sum() + target.sum()))


def main():
    parser = argparse.ArgumentParser()
    for arg in [
        "cache-root",
        "features-root",
        "predictions-root",
        "weights",
        "spatial-weights",
        "calibration",
        "config",
        "freeze-manifest",
        "output",
    ]:
        parser.add_argument("--" + arg, type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--event-gate-weights", type=Path)
    parser.add_argument("--backbone-weights", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    cal = json.loads(args.calibration.read_text())
    if config.get("truth_protocol") != "physical_coverage_v2":
        raise ValueError("Stress scoring requires audited physical_coverage_v2 truth")
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
        event_gate_weights=args.event_gate_weights,
    )
    if args.output.exists():
        raise ValueError("Stress output must be new")
    torch.set_num_threads(2)
    spatial = load_model(
        args.spatial_weights, device=args.device, backbone_weights=args.backbone_weights
    )
    report = {}
    for aoi in config["confirmation"]:
        root = args.features_root / aoi
        f = np.load(root / "features.npz")
        info = validate_feature_protocol(root, supervision=False)
        region = load_region(
            args.cache_root / aoi, labels=True
        )  # allowed only after full validation
        sites = np.load(root / "sites.npy")
        p = np.load(args.predictions_root / aoi / "probabilities.npy", mmap_mode="r")
        truth = ground_truth_events(args.cache_root / aoi, protocol="physical_coverage_v2")
        y, supervised = labels_for_sites(region["instances"], region["valid"], sites)
        occupancy = {}
        for method in ["pairwise", "smoothing", "hmm", "watch_lstm", "no_quality", "observation"]:
            scores = method_scores(
                f["x"], f["usable"], method, args.weights, cal["occupancy"], spatial_info=info
            )
            occupancy[method] = reliability(scores, y, supervised & f["usable"])
        # Conditional-on-common-sites controls isolate the temporal method. Proposal
        # recall loss from absent images is NOT measured by this mask perturbation.
        altered_valid = np.array(region["valid"], copy=True)
        gap_start = min(8, len(p) // 3)
        gap_end = min(gap_start + 4, len(p))
        altered_valid[gap_start:gap_end] = False
        cloud_x, cloud_usable = feature_cube(
            p, region["rgb"], altered_valid, region["metadata"], sites
        )
        static_x = f["x"].copy()
        for i in range(len(static_x)):
            good = np.flatnonzero(f["usable"][i])
            if len(good):
                # A static real candidate evidence control with the actual calendar
                # and quality schedule, not a claim of newly acquired imagery.
                static_x[i, :, 0:3] = f["x"][i, good[0], 0:3]
                static_x[i, :, 8:11] = f["x"][i, good[0], 8:11]
        static_x[:, :, [11, 12, 13]] = 0
        event_methods = [
            "smoothing",
            "hmm",
            "watch_lstm",
            "no_quality",
            "observation",
            "proposal_tracking",
        ]
        if args.event_gate_weights:
            event_methods.extend(["observation_gate", "gate_no_quality"])
        controls = {}
        for name, x, usable, target in [
            ("four_missing_observations", cloud_x, cloud_usable, truth),
            ("static_evidence_calendar_control", static_x, f["usable"], []),
            ("all_observations_missing", np.zeros_like(f["x"]), np.zeros_like(f["usable"]), []),
        ]:
            controls[name] = {}
            for method in event_methods:
                events = method_events(
                    x,
                    usable,
                    f["births"],
                    info,
                    method,
                    temporal_weights=args.weights,
                    calibration=cal,
                    event_gate_weights=args.event_gate_weights,
                    model_revision="frozen-stress:" + method,
                )
                if name == "all_observations_missing" and events:
                    raise AssertionError(f"{method} emitted events with no usable observations")
                if method in {"proposal_tracking", "observation_gate", "gate_no_quality"}:
                    if name == "static_evidence_calendar_control" and events:
                        raise AssertionError(f"{method} changed state on static spatial evidence")
                    for event in events:
                        site = event.quality["site_id"] - 1
                        current = event.quality["support_current_index"]
                        if not usable[site, current]:
                            raise AssertionError(f"{method} emitted on missing evidence")
                controls[name][method] = evaluate_events(events, target, region["metadata"])
        eligible = [
            i
            for i, frame in enumerate(region["metadata"]["frames"])
            if frame["registration"]["usable"] and frame["valid_fraction"] >= 0.5
        ]
        if not eligible:
            raise ValueError("No usable scene for radiometry/shift control")
        index = eligible[0]
        rgb = np.asarray(region["rgb"][index])
        valid = np.asarray(region["valid"][index])
        target = region["target"][index].astype(bool)
        image_controls = {
            "reference_index": index,
            "reference_source": info["metadata"]["frames"][index]["source_id"],
            "base_footprint_dice": dice(p[index, 0], target, valid),
            "brightness": {},
        }
        for factor in [0.7, 1.3]:
            changed = np.clip(rgb.astype("float32") * factor, 0, 255).astype("uint8")
            predicted = predict_scene(
                spatial, changed, device=args.device,
                tile_size=info.get("spatial_inference", {}).get("tile_size", 1024)
                if info.get("spatial_inference") else 1024,
            )
            image_controls["brightness"][str(factor)] = {
                "footprint_dice": dice(predicted[0], target, valid),
                "changed_binary_pixels_fraction": float(
                    np.mean(((p[index, 0] >= 0.45) != (predicted[0] >= 0.45))[valid])
                ),
            }
        shifted = np.stack([shift(band, (0, 3), order=0, mode="nearest") for band in rgb])
        image_controls["injected_3px_shift"] = registration_diagnostic(rgb, shifted, valid)
        report[aoi] = {
            "occupancy_reliability": occupancy,
            "event_methods": event_methods,
            "event_gate_score_scope": "Event compatibility is evaluated through event metrics; it is not candidate occupancy reliability.",
            "controls": controls,
            "image_controls": image_controls,
            "cloud_gap_frame_indices": [gap_start, gap_end],
            "limits": [
                "Controls are synthetic perturbations of real evidence, never new supporting observations.",
                "Event-gate methods share the proposal_tracking raw policy and use frozen event-gate calibration, not occupancy calibration.",
                "Temporal controls condition on the common candidate set and do not measure proposal recall loss.",
                "Brightness is a radiometry sensitivity test, not a complete simulated season.",
                "Real complete regional sequences also contain natural seasonal/cloud variation.",
            ],
        }
        print(aoi, "stress complete", flush=True)
    args.output.write_text(
        json.dumps(
            {
                "frozen": True,
                "truth_protocol": "physical_coverage_v2",
                "event_gate_weights_provided": args.event_gate_weights is not None,
                "regions": report,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

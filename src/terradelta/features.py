"""Causal candidate sites and observation-aware features from predictions only."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
from rasterio.features import shapes
from rasterio.transform import Affine
from rasterio.warp import transform_geom
from scipy import ndimage
from skimage.segmentation import watershed

from .data import load_region, sha256

FEATURE_NAMES = [
    "mean_footprint",
    "mean_interior",
    "max_footprint",
    "valid_fraction",
    "registration_pixels_over_two",
    "elapsed_months",
    "season_sin",
    "season_cos",
    "red",
    "green",
    "blue",
    "spatial_delta",
    "seasonal_reference_delta",
    "rgb_delta",
    "log_area",
    "time_since_usable_months",
    "registration_usable",
]
QUALITY_COLUMNS = [3, 4, 5, 12, 13, 15, 16]


def validate_feature_protocol(directory: Path, *, supervision: bool | None = None) -> dict:
    info = json.loads((Path(directory) / "sites.json").read_text())
    if info.get("feature_protocol") != "calendar_quality_v2":
        raise ValueError("Feature protocol mismatch; regenerate the versioned feature cache")
    expected = "physical_coverage_80_20_v2" if supervision else None
    if supervision is not None and info.get("supervision_protocol") != expected:
        raise ValueError(
            "Supervision protocol mismatch; cached label semantics do not match this experiment"
        )
    return info


def footprint_instances(
    probability: np.ndarray, valid: np.ndarray, threshold: float = 0.45, min_pixels: int = 3
) -> np.ndarray:
    """Common native-resolution watershed for every method, before temporal decisions."""
    foreground = (probability[0] >= threshold) & valid
    interior = probability[0] * (1 - probability[1])
    seeds = (interior >= max(0.18, threshold * 0.65)) & foreground
    markers, _ = ndimage.label(seeds)
    # A footprint without an interior seed still gets a reproducible component seed.
    whole, n = ndimage.label(foreground)
    maxima = ndimage.maximum(markers, whole, np.arange(1, n + 1)) if n else []
    count = int(markers.max())
    unseeded = np.flatnonzero(np.asarray(maxima) == 0) + 1
    if len(unseeded):
        boxes = ndimage.find_objects(whole)
        for index in unseeded:
            count += 1
            # Preserve label order and exact membership without rescanning the
            # full image for every component. Boxes can overlap (e.g. a ring
            # around an island), so retain the component-ID mask inside each.
            bounds = boxes[index - 1]
            local = markers[bounds]
            local[whole[bounds] == index] = count
    objects = watershed(-interior, markers, mask=foreground).astype("int32")
    areas = np.bincount(objects.ravel())
    too_small = areas < min_pixels
    too_large = areas > 5000
    reject = too_small | too_large
    reject[0] = False
    objects[reject[objects]] = 0
    # Consecutive IDs make later vectorized pooling bounded by actual candidate count.
    return np.unique(objects, return_inverse=True)[1].reshape(objects.shape).astype("int32")


def candidate_sites(
    probabilities: np.ndarray, valid: np.ndarray, usable: np.ndarray, threshold: float = 0.45
) -> tuple[np.ndarray, np.ndarray]:
    """Freeze each site's first geometry. Later frames can only add new site IDs.

    Only already observed evidence exists at birth. Earlier features may be inspected
    when the site is first discovered, but events cannot be emitted before its birth.
    This lets a future image add a site without changing old candidates/answers.
    """
    sites = np.zeros(valid.shape[1:], dtype="int32")
    births = []
    next_id = 1
    for t in range(len(valid)):
        if not usable[t]:
            continue
        objects = footprint_instances(probabilities[t], valid[t], threshold)
        boxes = ndimage.find_objects(objects)
        occupied = ndimage.maximum_filter(sites, size=3)
        for object_id, box in enumerate(boxes, 1):
            if box is None:
                continue
            obj = objects[box] == object_id
            area = obj.sum()
            overlap = occupied[box][obj]
            if np.count_nonzero(overlap) / area >= 0.25:
                continue
            new = obj & (occupied[box] == 0)
            if new.sum() < 3:
                continue
            # New candidates cannot overwrite or grow a prior site's geometry.
            view = sites[box]
            view[new] = next_id
            births.append(t)
            next_id += 1
    return sites, np.asarray(births, dtype="int32")


def feature_cube(
    probabilities: np.ndarray, rgb: np.ndarray, valid: np.ndarray, metadata: dict, sites: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Pool the same spatial evidence for all temporal baselines; never read labels."""
    n, times = int(sites.max()), len(rgb)
    indices = np.arange(1, n + 1)
    areas = np.bincount(sites.ravel(), minlength=n + 1)[1:].astype("float32")
    features = np.zeros((n, times, len(FEATURE_NAMES)), dtype="float32")
    usable = np.zeros((n, times), bool)
    first_time = datetime.fromisoformat(metadata["frames"][0]["acquisition_end"]).timestamp()
    last_good_time = np.full(n, first_time)
    previous_rgb = np.zeros((n, 3), "float32")
    previous_p = np.zeros(n, "float32")
    calendar_history = {}
    for t, frame in enumerate(metadata["frames"]):
        p = np.nan_to_num(probabilities[t, 0], nan=0.0)
        inside_valid = valid[t]
        vf = ndimage.mean(inside_valid, sites, indices)
        mean = ndimage.sum(p * inside_valid, sites, indices) / np.maximum(areas * vf, 1)
        interior = ndimage.sum(
            p * (1 - probabilities[t, 1]) * inside_valid, sites, indices
        ) / np.maximum(areas * vf, 1)
        maximum = ndimage.maximum(p * inside_valid, sites, indices)
        colors = np.stack(
            [
                ndimage.sum(c * inside_valid, sites, indices) / np.maximum(areas * vf, 1) / 255
                for c in rgb[t]
            ],
            axis=1,
        )
        date = datetime.fromisoformat(frame["acquisition_end"])
        prev_date = datetime.fromisoformat(metadata["frames"][max(t - 1, 0)]["acquisition_end"])
        elapsed = (date - prev_date).total_seconds() / (86400 * 31)
        reg = frame["registration"]
        shift = reg.get("shift_magnitude_pixels", 9.0)
        isgood = (vf >= 0.8) & reg["usable"]
        prior_year = calendar_history.get((date.year - 1, date.month))
        seasonal = (
            np.where(prior_year[1], prior_year[0], previous_p)
            if prior_year is not None
            else previous_p
        )
        out = np.column_stack(
            [
                mean,
                interior,
                maximum,
                vf,
                np.full(n, min(shift / 2, 5)),
                np.full(n, elapsed),
                np.full(n, np.sin(2 * np.pi * date.month / 12)),
                np.full(n, np.cos(2 * np.pi * date.month / 12)),
                colors,
                mean - previous_p,
                mean - seasonal,
                np.linalg.norm(colors - previous_rgb, axis=1),
                np.log1p(areas) / 8,
                (date.timestamp() - last_good_time) / (86400 * 31),
                np.full(n, float(reg["usable"])),
            ]
        )
        features[:, t] = np.nan_to_num(out)
        usable[:, t] = isgood
        last_good_time[isgood] = date.timestamp()
        previous_rgb[isgood] = colors[isgood]
        previous_p[isgood] = mean[isgood]
        calendar_history[(date.year, date.month)] = (mean.copy(), isgood.copy())
    return features, usable


def labels_for_sites(
    instances: np.ndarray, valid: np.ndarray, sites: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Training-only site supervision. Any GT overlap is not passed to inference."""
    ids = np.arange(1, int(sites.max()) + 1)
    y, supervised = [], []
    for labels, good in zip(instances, valid):
        overlap = ndimage.sum((labels > 0) & good, sites, ids)
        area = ndimage.sum(good, sites, ids)
        fraction = overlap / np.maximum(area, 1)
        y.append(fraction >= 0.8)
        supervised.append(
            (area / ndimage.sum(np.ones_like(good), sites, ids) >= 0.8)
            & ((fraction <= 0.2) | (fraction >= 0.8))
        )
    return np.array(y, dtype="float32").T, np.array(supervised, bool).T


def prepare_features(
    cache_dir: Path,
    probability_file: Path,
    out_dir: Path,
    *,
    labels: bool = False,
    threshold: float = 0.45,
) -> dict:
    region = load_region(cache_dir, labels=labels)
    meta = region["metadata"]
    probs = np.load(probability_file, mmap_mode="r")
    if probs.shape != (len(region["rgb"]), 2, *region["valid"].shape[-2:]):
        raise ValueError("Probability grid must equal the original native sequence grid")
    goodframes = np.array([f["registration"]["usable"] for f in meta["frames"]])
    sites, births = candidate_sites(probs, region["valid"], goodframes, threshold)
    x, good = feature_cube(probs, region["rgb"], region["valid"], meta, sites)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "sites.npy", sites)
    np.savez_compressed(out_dir / "features.npz", x=x, usable=good, births=births)
    if labels:
        y, supervised = labels_for_sites(region["instances"], region["valid"], sites)
        np.savez_compressed(out_dir / "supervision.npz", y=y, supervised=supervised)
    geometries = {}
    for geom, value in shapes(sites, mask=sites > 0, transform=Affine(*meta["transform"])):
        geometries.setdefault(str(int(value)), []).append(
            transform_geom(meta["crs"], "EPSG:4326", geom)
        )
    # A first candidate clipped by old occupancy may be disconnected; retain all parts.
    converted = {}
    for key, parts in geometries.items():
        converted[key] = (
            parts[0]
            if len(parts) == 1
            else {"type": "MultiPolygon", "coordinates": [p["coordinates"] for p in parts]}
        )
    public_meta = json.loads(json.dumps(meta))
    for frame in public_meta["frames"]:
        frame.pop("local_image", None)
    info = {
        "region_id": meta["region_id"],
        "feature_names": FEATURE_NAMES,
        "feature_protocol": "calendar_quality_v2",
        "supervision_protocol": "physical_coverage_80_20_v2" if labels else None,
        "probability_sha256": sha256(Path(probability_file)),
        "candidate_threshold": threshold,
        "geometry_crs": "EPSG:4326",
        "geometries": converted,
        "metadata": public_meta,
    }
    from .protocol import semantic_metadata_hash

    prediction_meta = Path(probability_file).parent / "prediction.json"
    if not prediction_meta.exists():
        raise ValueError("Probability products require their actual checkpoint provenance")
    prediction = json.loads(prediction_meta.read_text())
    if (
        prediction["probability_sha256"] != info["probability_sha256"]
        or prediction["region_id"] != meta["region_id"]
    ):
        raise ValueError("Probability bytes or region identity differ from their prediction record")
    info["spatial_weights_sha256"] = prediction["spatial_weights_sha256"]
    info["spatial_backbone_sha256"] = prediction.get("spatial_backbone_sha256")
    info["spatial_architecture"] = prediction.get("spatial_architecture")
    info["spatial_inference"] = prediction.get("spatial_inference")
    info["feature_implementation_sha256"] = sha256(Path(__file__))
    info["metadata_semantic_sha256"] = semantic_metadata_hash(Path(cache_dir) / "metadata.json")
    (out_dir / "sites.json").write_text(json.dumps(info) + "\n")
    return {"sites": len(births), "usable_fraction": float(good.mean()) if len(births) else 0}

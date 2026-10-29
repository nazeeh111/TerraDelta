"""Native-grid SpaceNet7 ingestion. Labels never enter image-only inference."""

from __future__ import annotations

import calendar
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import re

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import transform_geom, transform_bounds
from scipy import ndimage
from skimage.registration import phase_cross_correlation


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def month_bounds(filename: str) -> tuple[str, str]:
    match = re.search(r"global_monthly_(\d{4})_(\d{2})_", filename)
    if not match:
        raise ValueError("SpaceNet7 filenames must encode the observation month")
    y, m = map(int, match.groups())
    return (
        datetime(y, m, 1, tzinfo=timezone.utc).isoformat(),
        datetime(y, m, calendar.monthrange(y, m)[1], 23, 59, 59, tzinfo=timezone.utc).isoformat(),
    )


def registration_diagnostic(reference: np.ndarray, current: np.ndarray, valid: np.ndarray) -> dict:
    """Image-only phase diagnostic, not a ground-truth alignment guarantee.

    No displacement is applied. A large estimated shift causes abstention downstream.
    Invalid borders/clouds are eroded before estimating on gradient texture.
    """
    support = ndimage.binary_erosion(valid, iterations=3)
    if support.mean() < 0.25:
        return {
            "shift_pixels": None,
            "support_fraction": float(support.mean()),
            "usable": False,
            "reason": "insufficient_registration_support",
        }

    def textured(im):
        gray = im.astype(np.float32).mean(axis=0) / 255
        grad = ndimage.gaussian_gradient_magnitude(gray, 1)
        grad = (grad - grad[support].mean()) * support
        return grad * np.outer(np.hanning(grad.shape[0]), np.hanning(grad.shape[1]))

    a, b = textured(reference), textured(current)
    if min(float(np.linalg.norm(a)), float(np.linalg.norm(b))) < 1e-5:
        return {
            "shift_pixels": None,
            "support_fraction": float(support.mean()),
            "usable": False,
            "reason": "insufficient_texture_for_registration",
        }
    shift, error, _ = phase_cross_correlation(a, b, upsample_factor=10)
    magnitude = float(np.linalg.norm(shift))
    return {
        "shift_pixels": [float(v) for v in shift],
        "shift_magnitude_pixels": magnitude,
        "phase_error": float(error),
        "support_fraction": float(support.mean()),
        "usable": magnitude <= 2.0,
        "reason": "image_only_proxy_with_no_alignment_applied",
    }


def sequence_registration(rgb, valid, maximum_relative_shift: float = 2.0) -> list[dict]:
    """Causal QA against an immutable image anchor and every accepted site reference.

    Adjacent agreement alone cannot bound drift relative to a frozen candidate.
    Anchor offsets must remain within the threshold of every earlier ACCEPTED offset.
    Rejected frames never become candidate references. No pixels are warped and no
    labels are used. This diagnostic remains a proxy, not certified alignment.
    """
    result = []
    anchor = None
    accepted_offsets = []
    for t in range(len(rgb)):
        if anchor is None:
            diagnostic = registration_diagnostic(rgb[t], rgb[t], valid[t])
            if diagnostic["usable"]:
                anchor = t
                accepted_offsets.append(np.zeros(2))
            diagnostic.update(
                anchor_frame=anchor,
                reference_spread_pixels=0.0,
                protocol="anchor_and_all_accepted_offsets_v2",
            )
            result.append(diagnostic)
            continue
        diagnostic = registration_diagnostic(rgb[anchor], rgb[t], valid[anchor] & valid[t])
        adjacent = registration_diagnostic(rgb[t - 1], rgb[t], valid[t - 1] & valid[t])
        shift = diagnostic.get("shift_pixels")
        spread = (
            max(
                float(np.linalg.norm(np.asarray(shift) - previous)) for previous in accepted_offsets
            )
            if shift is not None
            else None
        )
        usable = bool(
            diagnostic["usable"] and spread is not None and spread <= maximum_relative_shift
        )
        # Adjacent diagnostics without support cannot certify a large displacement;
        # the long-lived anchor still supplies a direct comparison when clear.
        if adjacent.get("shift_pixels") is not None and not adjacent["usable"]:
            usable = False
        diagnostic.update(
            anchor_frame=anchor,
            adjacent=adjacent,
            reference_spread_pixels=spread,
            usable=usable,
            protocol="anchor_and_all_accepted_offsets_v2",
        )
        if shift is not None:
            diagnostic["shift_magnitude_pixels"] = max(float(np.linalg.norm(shift)), spread or 0.0)
        if usable:
            accepted_offsets.append(np.asarray(shift))
        elif spread is not None and spread > maximum_relative_shift:
            diagnostic["reason"] = "drift_from_frozen_candidate_references"
        result.append(diagnostic)
    return result


def ingest_region(region_dir: Path, cache_dir: Path, *, labels: bool = False) -> dict:
    """Validate all imagery before writing native-resolution memory-mapped arrays.

    Requires RGBA validity, common CRS/grid/units and monotonic unique months.
    SN7 labels_match are postprocessed ID-matched labels, used only when labels=True.
    """
    region_dir, cache_dir = Path(region_dir), Path(cache_dir)
    images = sorted((region_dir / "images_masked").glob("*.tif"))
    if not images:
        raise ValueError(f"No permitted images_masked GeoTIFFs in {region_dir}")
    frames, rgbs, valids, targets, edges, instances = [], [], [], [], [], []
    grid = None
    for image in images:
        with rasterio.open(image) as ds:
            if ds.crs is None or ds.count != 4 or ds.dtypes != ("uint8",) * 4:
                raise ValueError("Expected explicit CRS and four uint8 RGB/alpha bands")
            this_grid = (ds.crs, ds.transform, ds.width, ds.height)
            if grid is not None and (
                ds.crs != grid[0]
                or ds.width != grid[2]
                or ds.height != grid[3]
                or not np.allclose(tuple(ds.transform), tuple(grid[1]), rtol=0, atol=1e-6)
            ):
                raise ValueError("Incompatible grids: reproject explicitly before ingestion")
            if (
                ds.transform.b != 0
                or ds.transform.d != 0
                or ds.transform.a <= 0
                or ds.transform.e >= 0
            ):
                raise ValueError("Expected a north-up native pixel grid")
            if grid is None:
                grid = this_grid
            x = ds.read()
            valid = x[3] > 0
            if ds.nodata is not None:
                valid &= ~np.all(x[:3] == ds.nodata, axis=0)
            # Include source UDMs, whose CRS differs from images/labels_match.
            udm_path = region_dir / "labels" / f"{image.stem}_UDM.geojson"
            udm_count = 0
            if udm_path.exists():
                udm = json.loads(udm_path.read_text())
                crs = udm.get("crs", {}).get("properties", {}).get("name", "EPSG:4326")
                geoms = [
                    transform_geom(crs, ds.crs, f["geometry"])
                    for f in udm["features"]
                    if f.get("geometry")
                ]
                udm_count = len(geoms)
                if geoms:
                    valid &= (
                        rasterize(
                            [(g, 1) for g in geoms],
                            out_shape=ds.shape,
                            transform=ds.transform,
                            fill=0,
                        ).astype(bool)
                        == 0
                    )
            start, end = month_bounds(image.name)
            registration = {"usable": False, "reason": "pending_sequence_registration"}
            digest = sha256(image)
            frames.append(
                {
                    "source_id": f"sn7:{image.stem}",
                    "filename": image.name,
                    "sha256": digest,
                    "uri": f"https://spacenet-dataset.s3.amazonaws.com/spacenet/SN7_buildings/train/{region_dir.name}/images_masked/{image.name}",
                    "local_image": str(image.resolve()),
                    "source_transform": list(ds.transform)[:6],
                    "source_nodata": ds.nodata,
                    "acquisition_start": start,
                    "acquisition_end": end,
                    "published_at": None,
                    "available_at": end,
                    "availability_assumption": "retrospective_month_end",
                    "valid_fraction": float(valid.mean()),
                    "udm_polygons": udm_count,
                    "registration": registration,
                }
            )
            rgbs.append(x[:3])
            valids.append(valid)
            if labels:
                label_file = region_dir / "labels_match" / f"{image.stem}_Buildings.geojson"
                geojson = json.loads(label_file.read_text())
                label_crs = geojson.get("crs", {}).get("properties", {}).get("name")
                if not label_crs or (
                    geojson["features"] and rasterio.crs.CRS.from_string(label_crs) != ds.crs
                ):
                    raise ValueError("Matched labels must carry the actual image CRS")
                shapes = [
                    (f["geometry"], int(f["properties"]["Id"]) + 1)
                    for f in geojson["features"]
                    if f.get("geometry") and f["geometry"].get("coordinates")
                ]
                inst = (
                    rasterize(
                        shapes, out_shape=ds.shape, transform=ds.transform, fill=0, dtype="int32"
                    )
                    if shapes
                    else np.zeros(ds.shape, "int32")
                )
                target = inst > 0
                edge = target & (
                    (ndimage.minimum_filter(inst, size=3) != inst)
                    | (ndimage.maximum_filter(inst, size=3) != inst)
                )
                targets.append(target.astype("uint8"))
                edges.append(edge.astype("uint8"))
                instances.append(inst)
                frames[-1]["label_sha256"] = sha256(label_file)
                frames[-1]["label_crs"] = label_crs
                frames[-1]["empty_label_collection"] = not bool(geojson["features"])
    assert grid is not None
    for frame, diagnostic in zip(frames, sequence_registration(rgbs, valids)):
        frame["registration"] = diagnostic
    dates = [f["acquisition_start"] for f in frames]
    if dates != sorted(set(dates)):
        raise ValueError("Observation months must be unique and ordered")
    crs, transform, width, height = grid
    bbox = rasterio.transform.array_bounds(height, width, transform)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for name, values in [
        ("rgb", rgbs),
        ("valid", valids),
        ("target", targets),
        ("edge", edges),
        ("instances", instances),
    ]:
        if values:
            np.save(cache_dir / f"{name}.npy", np.stack(values))
    metadata = {
        "region_id": region_dir.name,
        "crs": crs.to_string(),
        "transform": list(transform)[:6],
        "width": width,
        "height": height,
        "bounds_native": list(bbox),
        "bbox": list(transform_bounds(crs, "EPSG:4326", *bbox)),
        "bands": ["red", "green", "blue", "alpha_validity"],
        "resolution_native": list(map(abs, (transform.a, transform.e))),
        "resolution_units": crs.linear_units,
        "nodata": "alpha==0 or source UDM",
        "license": "CC-BY-SA-4.0",
        "labels_postprocessed": labels,
        "registration_protocol": "anchor_and_all_accepted_offsets_v2",
        "alignment": "source grid unchanged; affine roundoff tolerance 1e-6 native units; image-only diagnostics, no label-assisted alignment",
        "frames": frames,
    }
    (cache_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def load_region(cache_dir: Path, *, labels: bool = False) -> dict:
    """Inference default intentionally cannot load label arrays."""
    cache_dir = Path(cache_dir)
    from .schema import validate_observation_order

    metadata = json.loads((cache_dir / "metadata.json").read_text())
    validate_observation_order(metadata["frames"])
    names = ["rgb", "valid"] + (["target", "edge", "instances"] if labels else [])
    return {
        "metadata": metadata,
        **{name: np.load(cache_dir / f"{name}.npy", mmap_mode="r") for name in names},
    }

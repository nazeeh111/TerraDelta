"""Versioned image-only execution from a validated native observation cache."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import rasterio
from rasterio.transform import Affine
from scipy.special import expit, logit

from .data import load_region, sha256
from .decoder import DecoderConfig, decode_events
from .schema import SourceIdentity, validate_observation_order
from .store import EventStore


def prefix_frames(frames: list[dict], end: str | None) -> list[dict]:
    """Availability, not acquisition start, determines the online input prefix."""
    validate_observation_order(frames)
    if end is None:
        return frames
    from datetime import time as day_time
    from .schema import utc_datetime

    cutoff = (
        datetime.combine(datetime.fromisoformat(end).date(), day_time.max, tzinfo=timezone.utc)
        if len(end) == 10
        else utc_datetime(end)
    )
    return [
        f for f in frames if utc_datetime(f.get("published_at") or f["acquisition_end"]) <= cutoff
    ]


def predict_sequence(
    cache: Path,
    weights: Path,
    output: Path,
    *,
    device: str = "cpu",
    tile_size: int = 1024,
    end: str | None = None,
) -> dict:
    import torch
    from .spatial import load_model, predict_scene

    torch.set_num_threads(2)
    model = load_model(weights, device=device)
    region = load_region(cache)  # labels=False by construction
    rgb = region["rgb"]
    meta = region["metadata"]
    frames = meta["frames"]
    frames = prefix_frames(frames, end)
    if not frames:
        raise ValueError("No observation exists before the requested end date")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Prediction output must be new/empty; preserve the previous artifact")
    output.mkdir(parents=True, exist_ok=True)
    mmap = np.lib.format.open_memmap(
        output / "probabilities.npy",
        mode="w+",
        dtype="float32",
        shape=(len(frames), 2, *region["valid"].shape[-2:]),
    )
    start = time.perf_counter()
    for t, frame in enumerate(frames):
        prediction = predict_scene(model, rgb[t], device=device, tile_size=tile_size)
        mmap[t] = prediction
        # Source-linked native GeoTIFF for independent GIS consumers.
        raster = np.concatenate(
            [
                np.where(region["valid"][t][None], prediction, np.nan),
                region["valid"][t][None].astype("float32"),
            ]
        )
        path = output / f"{frame['filename'][:-4]}_probability.tif"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=meta["width"],
            height=meta["height"],
            count=3,
            dtype="float32",
            crs=meta["crs"],
            transform=Affine(*meta["transform"]),
            nodata=np.nan,
            compress="deflate",
        ) as ds:
            ds.write(raster)
            ds.set_band_description(1, "building_footprint_probability")
            ds.set_band_description(2, "building_boundary_probability")
            ds.set_band_description(3, "observation_validity")
            ds.update_tags(
                source_id=frame["source_id"],
                source_sha256=frame["sha256"],
                source_uri=frame["uri"],
                acquisition_start=frame["acquisition_start"],
                acquisition_end=frame["acquisition_end"],
                model_sha256=sha256(weights),
                license="CC-BY-SA-4.0",
                publication_assumption="retrospective_month_end",
            )
        print(
            meta["region_id"], t + 1, len(frames), round(time.perf_counter() - start, 2), flush=True
        )
    mmap.flush()
    result = {
        "region_id": meta["region_id"],
        "frames": len(frames),
        "seconds": time.perf_counter() - start,
        "device": device,
        "tile_size": tile_size,
        "spatial_weights_sha256": sha256(weights),
        "probability_sha256": sha256(output / "probabilities.npy"),
    }
    (output / "prediction.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def method_scores(
    features: np.ndarray,
    usable: np.ndarray,
    method: str,
    temporal_weights: Path | None = None,
    calibration: dict | None = None,
) -> np.ndarray:
    from .temporal import (
        causal_smooth,
        hmm_filter,
        load_temporal,
        causal_ted,
        predict_temporal,
        sn7_retrospective,
    )

    base = features[:, :, 0]
    if method in ["pairwise", "spatial_tracking"]:
        score = base
    elif method == "smoothing":
        score = causal_smooth(base, usable)
    elif method == "hmm":
        score = hmm_filter(base, usable)
    elif method == "sn7_retrospective":
        score = sn7_retrospective(base, usable)
    elif method == "ted":
        distance = causal_ted(features, usable)
        score = np.empty_like(base)
        previous = np.full(len(base), 0.5)
        for t in range(base.shape[1]):
            previous = np.where(
                usable[:, t] & ((distance[:, t] >= 0.1) | (t == 0)), base[:, t], previous
            )
            score[:, t] = previous
    elif method in ["observation", "no_quality", "watch_lstm"]:
        if temporal_weights is None:
            raise ValueError("Learned temporal methods require versioned trained weights")
        score = predict_temporal(load_temporal(temporal_weights, method), features, usable)
    else:
        raise ValueError("Unknown temporal method")
    if calibration and method in calibration:
        params = calibration[method]
        score = expit(params["scale"] * logit(np.clip(score, 0.001, 0.999)) + params["bias"])
    return np.asarray(score, dtype="float32")


def reconstruct(
    feature_dir: Path,
    temporal_weights: Path,
    output: Path,
    *,
    method: str = "observation",
    calibration_path: Path | None = None,
    decoder: DecoderConfig | None = None,
    model_revision: str | None = None,
) -> dict:
    """Reconstruct into a new directory. Never overwrite a prior answer/store."""
    feature_dir, output = Path(feature_dir), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be new/empty so previous histories are preserved")
    output.mkdir(parents=True, exist_ok=True)
    info = json.loads((feature_dir / "sites.json").read_text())
    f = np.load(feature_dir / "features.npz")
    calibration = json.loads(calibration_path.read_text()) if calibration_path else {}
    score = method_scores(
        f["x"], f["usable"], method, temporal_weights, calibration.get("occupancy")
    )
    settings = decoder or DecoderConfig(
        **calibration.get("decoders", {}).get(method, calibration.get("decoder", {}))
    )
    revision = (
        model_revision
        or f"spatial:{info['probability_sha256'][:12]};temporal:{sha256(temporal_weights)[:12]};{method}"
    )
    events = decode_events(
        score,
        f["usable"],
        f["births"],
        info,
        settings,
        model_revision=revision,
        retrospective=method == "sn7_retrospective",
    )
    store = EventStore(output / "events.duckdb")
    for event in events:
        store.append(event)
    geojson = store.export_geojson(output / "events.geojson", include_retracted=True)
    parquet = store.export_parquet(output / "events.parquet", include_retracted=True)
    with (output / "revisions.jsonl").open("w") as stream:
        for event in events:
            stream.write(event.model_dump_json() + "\n")
    report = {
        "region_id": info["region_id"],
        "method": method,
        "model_revision": revision,
        "decoder": asdict(settings),
        "revisions": len(events),
        "snapshot_time": store.snapshot_time().isoformat() if store.snapshot_time() else None,
        "events": len(store.query()),
        "geojson_sha256": sha256(geojson),
        "geoparquet_sha256": sha256(parquet),
        "sources": [
            SourceIdentity(
                **{
                    k: frame[k]
                    for k in [
                        "source_id",
                        "uri",
                        "sha256",
                        "acquisition_start",
                        "acquisition_end",
                        "published_at",
                        "availability_assumption",
                    ]
                }
            ).model_dump(mode="json")
            for frame in info["metadata"]["frames"]
        ],
    }
    (output / "run.json").write_text(json.dumps(report, indent=2) + "\n")
    return report

"""Versioned image-only execution from a validated native observation cache."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import rasterio
from rasterio.transform import Affine
from scipy.special import expit, logit

from .data import load_region, sha256
from .decoder import DecoderConfig, decode_events
from .schema import EventRevision, SourceIdentity, validate_observation_order
from .store import EventStore

LEARNED_TEMPORAL_METHODS = ("observation", "no_quality", "watch_lstm")
GATE_METHODS = ("observation_gate", "gate_no_quality", "proposal_tracking")
METHODS = ("pairwise", "spatial_tracking", "smoothing", "hmm", "sn7_retrospective", "ted",
           *LEARNED_TEMPORAL_METHODS, *GATE_METHODS)


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


def export_probability_raster(
    path: Path,
    probability: np.ndarray,
    valid: np.ndarray,
    metadata: dict,
    frame: dict,
    model_sha256: str,
) -> Path:
    """Write both per-band nodata and a shared internal mask for GIS readers."""
    path = Path(path)
    if path.exists():
        raise ValueError("Raster output exists; preserve the previous product")
    if probability.shape != (2, *valid.shape) or valid.dtype != np.bool_:
        raise ValueError("Probability/validity grids disagree")
    if not np.isfinite(probability[:, valid]).all():
        raise ValueError("Usable probability pixels must be finite")
    raster = np.concatenate(
        [np.where(valid[None], probability, np.nan), valid[None].astype("float32")]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=metadata["width"],
            height=metadata["height"],
            count=3,
            dtype="float32",
            crs=metadata["crs"],
            transform=Affine(*metadata["transform"]),
            nodata=np.nan,
            compress="deflate",
        ) as ds:
            ds.write(raster)
            ds.write_mask(valid.astype("uint8") * 255)
            for index, name in enumerate(
                [
                    "building_footprint_probability",
                    "building_boundary_probability",
                    "observation_validity",
                ],
                1,
            ):
                ds.set_band_description(index, name)
            ds.update_tags(
                source_id=frame["source_id"],
                source_sha256=frame["sha256"],
                source_uri=frame["uri"],
                acquisition_start=frame["acquisition_start"],
                acquisition_end=frame["acquisition_end"],
                date_basis="mosaic_filename_month",
                model_sha256=model_sha256,
                license=metadata.get("license") or "unspecified",
                publication_assumption="retrospective_month_end",
                processing_revision="native_probability_internal_mask_v2",
            )
    return path


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
        path = output / f"{frame['filename'][:-4]}_probability.tif"
        export_probability_raster(
            path, prediction, np.asarray(region["valid"][t]), meta, frame, sha256(weights)
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


def _method_decoder(method, calibration, decoder=None):
    if method in GATE_METHODS:
        from .event_gate import RAW_CONFIG

        from dataclasses import replace

        params = calibration.get("event_gates", {}).get(method, {})
        if decoder is not None:
            settings = decoder
        elif method in calibration.get("decoders", {}):
            settings = DecoderConfig(**calibration["decoders"][method])
        else:
            settings = replace(RAW_CONFIG,
                               timing_padding_days=int(params.get("timing_padding_days", 0)))
        if replace(settings, timing_padding_days=0) != RAW_CONFIG:
            raise ValueError("Gate methods must retain the shared raw proposal decoder policy")
        return settings
    return decoder or DecoderConfig(
        **calibration.get("decoders", {}).get(method, calibration.get("decoder", {}))
    )


def _method_revision(info, method, temporal_weights=None, event_gate_weights=None):
    if method in GATE_METHODS:
        if method == "proposal_tracking":
            return f"spatial:{info.get('spatial_weights_sha256', 'unspecified')[:12]};proposal-v1"
        if event_gate_weights is None:
            raise ValueError("Learned event methods require --event-gate-weights")
        return f"event-gate:{sha256(event_gate_weights)};{method}"
    temporal = sha256(temporal_weights)[:12] if temporal_weights else "unused"
    spatial = info.get("spatial_weights_sha256", "unspecified")[:12]
    return f"spatial:{spatial};temporal:{temporal};{method}"


def _processing_revision(base, info, method, calibration, decoder,
                         temporal_weights=None, event_gate_weights=None):
    """Bind effective processing parameters even when callers supply a display label."""
    payload = {
        "method": method, "calibration": calibration, "decoder": asdict(decoder),
        "spatial_weights_sha256": info.get("spatial_weights_sha256"),
        "temporal_weights_sha256": sha256(temporal_weights)
        if temporal_weights and method in LEARNED_TEMPORAL_METHODS else None,
        "event_gate_weights_sha256": sha256(event_gate_weights)
        if event_gate_weights and method in GATE_METHODS[:2] else None,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return f"{base};processing:{hashlib.sha256(encoded).hexdigest()}"


def method_events(
    features: np.ndarray,
    usable: np.ndarray,
    births: np.ndarray,
    info: dict,
    method: str | None = None,
    *,
    temporal_weights: Path | None = None,
    event_gate_weights: Path | None = None,
    calibration: dict | None = None,
    decoder: DecoderConfig | None = None,
    model_revision: str | None = None,
    record_origin: datetime | None = None,
) -> list[EventRevision]:
    """Dispatch all twelve methods to causal histories, without opening truth data.

    Gate calibration lives under event_gates; occupancy and decoder calibration
    retain their existing keys. Gate models filter raw events rather than expose
    an invented occupancy score array. Only their timing padding can override the
    shared raw candidate decoder policy.
    """
    method = method or ("observation_gate" if event_gate_weights else "observation")
    if method not in METHODS:
        raise ValueError(f"Unknown event method: {method}")
    calibration = calibration or {}
    settings = _method_decoder(method, calibration, decoder)
    base_revision = model_revision or _method_revision(info, method, temporal_weights, event_gate_weights)
    revision = _processing_revision(base_revision, info, method, calibration, settings,
                                    temporal_weights, event_gate_weights)
    if method in GATE_METHODS:
        from .event_gate import generate_events

        params = dict(calibration.get("event_gates", {}).get(method, {}))
        params["timing_padding_days"] = settings.timing_padding_days
        events = generate_events(features, usable, births, info, event_gate_weights,
                                 kind=method, calibration={method: params}, record_origin=record_origin)
        return [event if event.model_revision == revision
                else event.model_copy(update={"model_revision": revision}) for event in events]
    scores = method_scores(features, usable, method, temporal_weights, calibration.get("occupancy"))
    return decode_events(scores, usable, births, info, settings, model_revision=revision,
                         record_origin=record_origin, retrospective=method == "sn7_retrospective")


def reconstruct(
    feature_dir: Path,
    temporal_weights: Path | None = None,
    output: Path | None = None,
    *,
    method: str | None = None,
    event_gate_weights: Path | None = None,
    calibration_path: Path | None = None,
    decoder: DecoderConfig | None = None,
    model_revision: str | None = None,
) -> dict:
    """Reconstruct into a new directory; preserve old positional/API method support."""
    if output is None:
        raise ValueError("A new output directory is required")
    feature_dir, output = Path(feature_dir), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be new/empty so previous histories are preserved")
    method = method or ("observation_gate" if event_gate_weights else "observation")
    info = json.loads((feature_dir / "sites.json").read_text())
    calibration = json.loads(calibration_path.read_text()) if calibration_path else {}
    settings = _method_decoder(method, calibration, decoder)
    base_revision = model_revision or _method_revision(info, method, temporal_weights, event_gate_weights)
    with np.load(feature_dir / "features.npz") as features:
        events = method_events(
            features["x"], features["usable"], features["births"], info, method,
            temporal_weights=temporal_weights, event_gate_weights=event_gate_weights,
            calibration=calibration, decoder=decoder, model_revision=base_revision,
        )
    revision = events[0].model_revision if events else _processing_revision(
        base_revision, info, method, calibration, settings, temporal_weights, event_gate_weights
    )
    output.mkdir(parents=True, exist_ok=True)
    with EventStore(output / "events.duckdb") as store:
        for event in events:
            store.append(event)
        geojson = store.export_geojson(output / "events.geojson", include_retracted=True)
        parquet = store.export_parquet(output / "events.parquet", include_retracted=True)
        snapshot = store.snapshot_time()
        event_count = len(store.query())
    with (output / "revisions.jsonl").open("w") as stream:
        for event in events:
            stream.write(event.model_dump_json() + "\n")
    report = {
        "region_id": info["region_id"],
        "method": method,
        "model_revision": revision,
        "decoder": asdict(settings),
        "revisions": len(events),
        "snapshot_time": snapshot.isoformat() if snapshot else None,
        "events": event_count,
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
    if method in GATE_METHODS:
        report["event_gate_weights_sha256"] = sha256(event_gate_weights) if event_gate_weights else None
        report["event_gate_parameters"] = calibration.get("event_gates", {}).get(method, {})
    (output / "run.json").write_text(json.dumps(report, indent=2) + "\n")
    return report

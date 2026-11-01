"""Causal event-correctness gate over a common raw spatial proposal history.

The gate filters event hypotheses, not occupancy pixels. Geometry, event IDs,
birth gating and raw persistence policy are shared with proposal_tracking.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import expit
import torch
from torch import nn

from .data import sha256
from .decoder import DecoderConfig, _source, decode_events
from .features import FEATURE_NAMES
from .schema import EventRevision, ObservationInterval, geodesic_area_m2, utc_datetime

RAW_CONFIG = DecoderConfig(threshold=0.55, confirmations=2, minimum_dwell_days=20)
KINDS = ("observation_gate", "gate_no_quality", "proposal_tracking")
_BLOCKS = ("current", "pre_mean", "post_mean", "post_std")
GATE_FEATURE_NAMES = [f"{block}:{name}" for block in _BLOCKS for name in FEATURE_NAMES] + [
    "target_direction", "months_since_first_support", "months_since_before_support",
    "log_usable_support_count", "pre_window_count_over_three", "post_window_count_over_three",
    "directional_current_margin", "directional_post_pre_contrast", "post_target_fraction",
    "log_geometry_area_m2_over_ten",
]
EXTRA_QUALITY_COLUMNS = (3, 4, 5, 6, 7, 12, 13, 15, 16)
QUALITY_DROP_INDICES = tuple(
    block * len(FEATURE_NAMES) + column for block in range(4) for column in EXTRA_QUALITY_COLUMNS
)
FORMAT = "terradelta-event-gate-v1"
SPATIAL_PROVENANCE_KEYS = (
    "spatial_weights_sha256", "spatial_backbone_sha256", "spatial_inference",
    "candidate_threshold",
)


def _full_sha256(value):
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def validate_spatial_provenance(value):
    """Validate an explicit model/base/inference binding, with no file/label reads."""
    if not isinstance(value, dict) or set(value) != set(SPATIAL_PROVENANCE_KEYS):
        raise ValueError("explicit spatial provenance requires model, backbone and inference keys")
    if not _full_sha256(value["spatial_weights_sha256"]):
        raise ValueError("spatial provenance requires a full lowercase checkpoint SHA256")
    threshold = value["candidate_threshold"]
    if type(threshold) not in (int, float) or not 0 < threshold < 1:
        raise ValueError("spatial provenance requires a finite candidate threshold between zero and one")
    backbone = value["spatial_backbone_sha256"]
    if backbone is not None and not _full_sha256(backbone):
        raise ValueError("spatial provenance backbone must be full SHA256 or explicit null")
    inference = value["spatial_inference"]
    integer_fields = {"tile_size": 16, "context_halo_native_pixels": 0,
                      "native_alignment": 1, "internal_input_scale": 1}
    if (not isinstance(inference, dict)
            or set(inference) != set(integer_fields) | {"implementation_sha256"}):
        raise ValueError("spatial inference provenance is missing or has unsupported fields")
    if any(type(inference[key]) is not int or inference[key] < minimum
           for key, minimum in integer_fields.items()):
        raise ValueError("spatial inference grid parameters must be explicit valid integers")
    if inference["internal_input_scale"] not in (1, 2):
        raise ValueError("unsupported spatial inference internal scale")
    implementations = inference["implementation_sha256"]
    required_implementations = {"spatial.py", "pretrained_spatial.py"} if backbone else {"spatial.py"}
    if (not isinstance(implementations, dict) or not implementations
            or set(implementations) != required_implementations
            or any(key not in {"spatial.py", "pretrained_spatial.py"} or not _full_sha256(sha)
                   for key, sha in implementations.items())):
        raise ValueError("spatial inference requires full implementation source hashes")
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def assert_legacy_spatial_compatibility(info):
    """Allow historical metadata or the independently checked v2 replay grid.

    This is an explicit implementation allowlist, never a hash of whichever
    implementation happens to be installed. Original freeze replay uses bd33082.
    """
    if (info.get("spatial_backbone_sha256") is not None
            or ("candidate_threshold" in info and info["candidate_threshold"] != .45)):
        raise ValueError("legacy gate has no binding for new backbone/inference provenance")
    inference = info.get("spatial_inference")
    if inference is None:
        return
    expected = {
        "tile_size": 1024, "context_halo_native_pixels": 64,
        "native_alignment": 8, "internal_input_scale": 2,
        "implementation_sha256": {
            "spatial.py": "a801eef9f770057cb7e5e26f74b7edc027ea73cd3136da8d1e7541d20b03ebff",
        },
    }
    if (info.get("spatial_weights_sha256") !=
            "7ab95679a2009918dbab62fca64244476df519080cb8c0d376c00d1494983886"
            or info.get("candidate_threshold") != .45 or inference != expected):
        raise ValueError("legacy gate has no binding for new backbone/inference provenance")


def assert_spatial_provenance(info, expected):
    expected = validate_spatial_provenance(expected)
    if any(key not in info for key in SPATIAL_PROVENANCE_KEYS):
        raise ValueError("features are missing explicit spatial provenance fields")
    observed = validate_spatial_provenance({key: info[key] for key in SPATIAL_PROVENANCE_KEYS})
    if observed != expected:
        raise ValueError("feature spatial model/backbone/inference differs from frozen provenance")


class EventGate(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Linear(len(GATE_FEATURE_NAMES), 64), nn.ReLU(),
                         nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))


def raw_proposals(x, usable, births, info, *, record_origin=None):
    """One common causal spatial stream, with no learned filtering or label reads."""
    x, usable, births = np.asarray(x), np.asarray(usable, bool), np.asarray(births)
    if x.ndim != 3 or x.shape[-1] != len(FEATURE_NAMES) or x.shape[:2] != usable.shape:
        raise ValueError("event gate requires [site,time,17] features and matching usability")
    if not np.isfinite(x[usable]).all():
        raise ValueError("usable feature rows must be finite")
    if info.get("feature_names") != FEATURE_NAMES:
        raise ValueError("feature names/order differ from the gate contract")
    probability = np.where(usable, x[:, :, 0], 0.5)
    return decode_events(
        probability, usable, births, info, RAW_CONFIG,
        model_revision=f"spatial:{info.get('spatial_weights_sha256', 'unspecified')[:12]};proposal-v1",
        record_origin=record_origin,
    )


def support_indices(event: EventRevision) -> tuple[int, int, int, int]:
    """Explicit raw-decoder indices are required; do not infer them from dates."""
    keys = ("site_id", "support_before_index", "support_first_index", "support_current_index")
    try:
        values = tuple(event.quality[key] for key in keys)
    except KeyError as error:
        raise ValueError("raw decoder is missing explicit gate support indices") from error
    if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in values):
        raise ValueError("gate support indices must be integers")
    site, before, first, current = map(int, values)
    if site < 1 or not 0 <= before < first <= current:
        raise ValueError("gate support indices are out of order")
    return site, before, first, current


def event_features(x, usable, event, current, info, *, area_m2=None):
    """A 78-vector using only this usable observation and its available prefix.

    Pre-change = last <=3 usable observations ending at the last stable support.
    Post-change = latest <=3 usable observations since first opposite support.
    Source dates, direction and frozen image-derived area are inference inputs.
    """
    site_id, before, first, _ = support_indices(event)
    site = site_id - 1
    if not first <= current < x.shape[1] or not usable[site, current]:
        raise ValueError("gate rows require a current usable observation after first support")
    pre = np.flatnonzero(usable[site, :before + 1])[-3:]
    post_all = np.flatnonzero(usable[site, first:current + 1]) + first
    post = post_all[-3:]
    if not len(pre) or not len(post):
        raise ValueError("both pre-change and post-change usable support are required")
    direction = 1.0 if event.change_class == "building_addition" else -1.0
    post_values = x[site, post]
    pre_mean = np.asarray(x[site, pre].mean(axis=0), dtype=np.float32)
    post_mean = np.asarray(post_values.mean(axis=0), dtype=np.float32)
    frame = info["metadata"]["frames"]
    now = utc_datetime(frame[current]["acquisition_end"])
    first_date = utc_datetime(frame[first]["acquisition_end"])
    before_date = utc_datetime(frame[before]["acquisition_end"])
    target_support = (
        x[site, post_all, 0] >= RAW_CONFIG.threshold if direction > 0
        else x[site, post_all, 0] <= 1 - RAW_CONFIG.threshold
    )
    if area_m2 is None:
        area_m2 = geodesic_area_m2(event.geometry)
    extra = [
        direction, (now - first_date).total_seconds() / (86400 * 31),
        (now - before_date).total_seconds() / (86400 * 31),
        np.log1p(target_support.sum()), len(pre) / 3, len(post) / 3,
        direction * (x[site, current, 0] - 0.5),
        direction * (post_mean[0] - pre_mean[0]),
        float(np.mean(target_support[-3:])), np.log1p(area_m2) / 10,
    ]
    result = np.concatenate((x[site, current], pre_mean, post_mean,
                             post_values.std(axis=0), extra)).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("non-finite gate evidence")
    return result, sorted(set(pre.tolist() + post.tolist() + [int(current)]))


def standardize(values, mean, std, kind):
    normalized = np.clip((np.asarray(values, np.float32) - mean) / std, -12, 12)
    if kind == "gate_no_quality":
        normalized = normalized.copy()
        normalized[..., list(QUALITY_DROP_INDICES)] = 0
    return np.asarray(normalized, np.float32)


@dataclass
class GateBundle:
    models: dict
    mean: np.ndarray
    std: np.ndarray
    calibration: dict
    model_revision: str
    spatial_weights_sha256: str
    spatial_provenance: dict | None = None

    @torch.inference_mode()
    def logit(self, row, kind):
        values = standardize(row, self.mean, self.std, kind)
        return float(self.models[kind](torch.from_numpy(values)[None])[0, 0])


def load_gate(weights: str | Path | GateBundle) -> GateBundle:
    if isinstance(weights, GateBundle):
        return weights
    path = Path(weights)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != FORMAT or checkpoint.get("feature_names") != GATE_FEATURE_NAMES:
        raise ValueError("unsupported event gate format or feature order")
    if (checkpoint.get("raw_decoder") != asdict(RAW_CONFIG)
            or checkpoint.get("quality_drop_indices") != list(QUALITY_DROP_INDICES)):
        raise ValueError("stored event-gate decoder or quality-ablation policy differs from runtime")
    provenance = None
    if "spatial_provenance" in checkpoint:
        provenance = validate_spatial_provenance(checkpoint["spatial_provenance"])
        if provenance["spatial_weights_sha256"] != checkpoint["spatial_weights_sha256"]:
            raise ValueError("event-gate spatial identity fields disagree")
    models = {}
    for kind in KINDS[:2]:
        model = EventGate()
        model.load_state_dict(checkpoint["models"][kind], strict=True)
        models[kind] = model.eval()
    mean = checkpoint["mean"].numpy()
    std = checkpoint["std"].numpy()
    if (mean.shape != (len(GATE_FEATURE_NAMES),) or std.shape != mean.shape
            or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0)):
        raise ValueError("invalid train-only gate standardizer")
    for kind in KINDS[:2]:
        fit = checkpoint.get("calibration", {}).get(kind, {})
        if (any(type(fit.get(key)) not in (int, float) or not np.isfinite(fit[key])
                for key in ("scale", "bias")) or fit["scale"] < 0):
            raise ValueError("invalid or missing development gate calibration")
    return GateBundle(models, mean, std, checkpoint["calibration"],
                      f"event-gate:{sha256(path)}", checkpoint["spatial_weights_sha256"], provenance)


def generate_events(
    x, usable, births, info, weights=None, kind="observation_gate", calibration=None,
    record_origin=None,
) -> list[EventRevision]:
    """Gate eligible raw events without changing candidate IDs, geometry or raw state.

    A low score retains a proposed revision; later usable evidence may confirm.
    A raw reversal retracts support, even for an event never gate-confirmed.
    Confirmation is sticky until that raw correction, avoiding score-jitter alerts.
    """
    if kind not in KINDS:
        raise ValueError("unknown event gate method")
    bundle = None
    if kind != "proposal_tracking":
        if weights is None:
            raise ValueError("learned event methods require trained gate weights")
        bundle = load_gate(weights)
        if info.get("spatial_weights_sha256") != bundle.spatial_weights_sha256:
            raise ValueError("spatial checkpoint differs from event-gate training provenance")
        if bundle.spatial_provenance is not None:
            assert_spatial_provenance(info, bundle.spatial_provenance)
        else:
            assert_legacy_spatial_compatibility(info)
    origin = record_origin or datetime.now(timezone.utc)
    raw = raw_proposals(x, usable, births, info, record_origin=origin)
    params = (calibration or {}).get(kind, calibration or {})
    padding = int(params.get("timing_padding_days", 0))
    if padding < 0:
        raise ValueError("timing padding must be nonnegative")
    if kind == "proposal_tracking":
        if padding:
            probability = np.where(usable, x[:, :, 0], 0.5)
            return decode_events(probability, usable, births, info,
                                 replace(RAW_CONFIG, timing_padding_days=padding),
                                 model_revision=raw[0].model_revision if raw else "proposal-v1",
                                 record_origin=origin)
        return raw
    assert bundle is not None
    params = {**bundle.calibration[kind], **params}
    padding = int(params.get("timing_padding_days", 0))
    if padding < 0:
        raise ValueError("timing padding must be nonnegative")
    threshold = float(params.get("threshold", 0.5))
    if (not 0 <= threshold <= 1 or float(params["scale"]) < 0
            or not np.isfinite([float(params["scale"]), float(params["bias"])]).all()):
        raise ValueError("invalid event gate threshold/calibration")
    effective_parameters = {
        "scale": float(params["scale"]), "bias": float(params["bias"]),
        "threshold": threshold, "timing_padding_days": padding,
    }
    raw_policy = {**asdict(RAW_CONFIG), "timing_padding_days": padding}
    processing_hash = hashlib.sha256(json.dumps(
        {"parameters": effective_parameters, "raw_decoder": raw_policy},
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()
    frames = info["metadata"]["frames"]
    sources = [_source(frame) for frame in frames]
    by_time = {}
    for event in raw:
        site_id, _, _, current = support_indices(event)
        by_time.setdefault(current, []).append((site_id, event))
    active, revisions, output, areas = {}, {}, [], {}

    def emit(template, current, state, score, reason, feature_indices=None):
        site_id, before, first, _ = support_indices(template)
        event_id = template.event_id
        revisions[event_id] = revisions.get(event_id, 0) + 1
        quality = dict(template.quality)
        quality.update({
            "support_current_index": int(current), "registration": frames[current]["registration"],
            "event_gate_kind": kind, "event_gate_threshold": threshold,
            "event_gate_parameters": effective_parameters,
            "raw_proposal_decoder": raw_policy,
            "score_meaning": "development_calibrated_event_compatibility_not_universal_guarantee",
            "gate_feature_indices": feature_indices or [],
            "gate_calibration_scope": "unbalanced proposal prevalence in development regions",
        })
        raw_lower = sources[before].acquisition_start
        raw_upper = sources[first].acquisition_end
        lower = max(sources[0].acquisition_start, raw_lower - timedelta(days=padding))
        upper = min(sources[current].available_at, raw_upper + timedelta(days=padding))
        quality["development_timing_padding_days"] = padding
        # Current feature rows include causal seasonal references. Listing the full
        # usable prefix preserves their source dependency, not just three snapshots.
        indices = np.flatnonzero(usable[site_id - 1, :current + 1]).tolist()
        payload = template.model_dump(mode="python")
        payload.update(
            revision=revisions[event_id], state=state, confidence=score, reason=reason,
            detected_at=sources[current].available_at,
            record_time=origin + timedelta(microseconds=len(output)),
            sources=[sources[index] for index in indices], quality=quality,
            observation_interval=ObservationInterval(lower=lower, upper=upper),
            model_revision=f"{bundle.model_revision};{kind};calibration:{processing_hash}",
        )
        output.append(EventRevision(**payload))

    for current in range(x.shape[1]):
        attempted = set()
        for site_id, event in by_time.get(current, []):
            if event.state == "proposed":
                active[event.event_id] = {"event": event, "eligible": False, "confirmed": False}
                emit(event, current, "proposed", None,
                     "Raw opposite-state evidence; ordinary persistence pending")
            elif event.state == "retracted":
                active.pop(event.event_id, None)
                emit(event, current, "retracted", None, event.reason)
            elif event.state == "confirmed_by_model":
                entry = active[event.event_id]
                entry.update(event=event, eligible=True)
        for event_id, entry in active.items():
            template = entry["event"]
            site_id, _, _, _ = support_indices(template)
            if not entry["eligible"] or entry["confirmed"] or not usable[site_id - 1, current]:
                continue
            if event_id in attempted:
                continue
            attempted.add(event_id)
            if site_id not in areas:
                areas[site_id] = geodesic_area_m2(template.geometry)
            row, indices = event_features(x, usable, template, current, info,
                                         area_m2=areas[site_id])
            score = float(expit(float(params["scale"]) * bundle.logit(row, kind) + params["bias"]))
            entry["score"] = score
            state = "confirmed_by_model" if score >= threshold else "proposed"
            entry["confirmed"] = state == "confirmed_by_model"
            emit(template, current, state, score,
                 "Event gate accepted ordinary persistent evidence" if entry["confirmed"]
                 else "Persistent raw evidence remains ambiguous under the event gate", indices)
    return output

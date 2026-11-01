"""Hash the complete confirmation boundary before any held-out scoring."""

from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from .data import sha256
from .features import validate_feature_protocol

PROTOCOL_FILES = [
    "pyproject.toml",
    "uv.lock",
    "src/terradelta/acquire.py",
    "src/terradelta/data.py",
    "src/terradelta/spatial.py",
    "src/terradelta/pretrained_spatial.py",
    "src/terradelta/features.py",
    "src/terradelta/temporal.py",
    "src/terradelta/event_gate.py",
    "src/terradelta/contracts.py",
    "src/terradelta/decoder.py",
    "src/terradelta/pipeline.py",
    "src/terradelta/evaluation.py",
    "src/terradelta/schema.py",
    "src/terradelta/store.py",
    "src/terradelta/service.py",
    "src/terradelta/cli.py",
    "src/terradelta/protocol.py",
    "scripts/train_spatial.py",
    "scripts/train_spatial_pretrained.py",
    "scripts/select_spatial_inference.py",
    "scripts/train_temporal.py",
    "scripts/train_event_gate.py",
    "scripts/calibrate.py",
    "scripts/evaluate.py",
    "scripts/evaluate_gate_calibration.py",
    "scripts/predict_all.py",
    "scripts/rebuild_features.py",
    "scripts/check_online_invariance.py",
    "scripts/stress_and_calibration.py",
    "scripts/export_rasters.py",
]


def semantic_metadata_hash(path: Path) -> str:
    metadata = json.loads(path.read_text())
    for frame in metadata["frames"]:
        frame.pop("local_image", None)
    encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def experiment_boundary(
    repo: Path,
    features_root: Path,
    cache_root: Path,
    regions: list[str],
    temporal_weights: Path,
    spatial_weights: Path,
    calibration: Path,
    config: Path,
    event_gate_weights: Path | None = None,
) -> dict:
    """Hash bytes without interpreting confirmation labels/outcomes.

    Frozen truth-cache bytes ensure scoring uses the same prepared data. Image and
    probability provenance are bound in sites.json plus explicit checkpoint hashes.
    Host-local image paths are excluded from semantic metadata hashing.
    """
    settings = json.loads(config.read_text())
    from .contracts import (
        assert_calibration, preflight_heads, preflight_implementation,
        preflight_spatial_checkpoint, study_binding,
    )

    spatial_binding = study_binding(settings)
    if settings.get("event_gate") and event_gate_weights is None:
        raise ValueError("Event-gate experiment requires frozen gate weights")
    identity = preflight_spatial_checkpoint(spatial_weights, spatial_binding)
    model_info = spatial_binding if spatial_binding is not None else identity
    if spatial_binding is not None:
        preflight_implementation(spatial_binding)
    preflight_heads(
        model_info, temporal_weights=temporal_weights, event_gate_weights=event_gate_weights,
        temporal_kinds=("observation", "no_quality", "watch_lstm"),
        require_bound=spatial_binding is not None,
        config_sha256=sha256(config) if spatial_binding is not None else None,
    )
    assert_calibration(
        model_info, json.loads(calibration.read_text()), temporal_weights=temporal_weights,
        event_gate_weights=event_gate_weights, require_bound=spatial_binding is not None,
        config_sha256=sha256(config) if spatial_binding is not None else None,
    )
    inputs = {}
    for region in regions:
        feature = features_root / region
        cache = cache_root / region
        info = validate_feature_protocol(feature, supervision=False)
        if spatial_binding is not None:
            from .event_gate import assert_spatial_provenance

            assert_spatial_provenance(info, spatial_binding)
        metadata = json.loads((cache / "metadata.json").read_text())
        if info.get("region_id") != region or metadata.get("region_id") != region:
            raise ValueError(f"Frozen region identities disagree: {region}")
        if info.get("spatial_weights_sha256") != sha256(spatial_weights):
            raise ValueError(f"Spatial checkpoint provenance mismatch: {region}")
        if info.get("metadata_semantic_sha256") != semantic_metadata_hash(cache / "metadata.json"):
            raise ValueError(f"Feature observation metadata changed after inference: {region}")
        inputs[region] = {
            "features_sha256": sha256(feature / "features.npz"),
            "sites_sha256": sha256(feature / "sites.json"),
            "sites_raster_sha256": sha256(feature / "sites.npy"),
            "metadata_semantic_sha256": semantic_metadata_hash(cache / "metadata.json"),
            "rgb_sha256": sha256(cache / "rgb.npy"),
            "valid_sha256": sha256(cache / "valid.npy"),
            "truth_instances_sha256": sha256(cache / "instances.npy"),
            "probability_sha256": info["probability_sha256"],
        }
    return {
        "format": "terradelta-full-freeze-v4",
        "spatial_provenance": spatial_binding,
        "event_gate_weights_sha256": sha256(event_gate_weights) if event_gate_weights else None,
        "temporal_weights_sha256": sha256(temporal_weights),
        "spatial_weights_sha256": sha256(spatial_weights),
        "calibration_sha256": sha256(calibration),
        "config_sha256": sha256(config),
        "confirmation_regions": regions,
        "region_inputs": inputs,
        "protocol_sources": {relative: sha256(repo / relative) for relative in PROTOCOL_FILES},
    }


def freeze_experiment(path: Path, **kwargs) -> dict:
    if path.exists():
        raise ValueError("Freeze manifest is immutable; use a new versioned path")
    payload = experiment_boundary(**kwargs)
    payload["frozen_at"] = datetime.now(timezone.utc).isoformat()
    payload["policy"] = (
        "No model, threshold, data, decoder or metric tuning after confirmation scoring."
    )
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def validate_experiment(path: Path | None, **kwargs) -> dict:
    if path is None:
        raise ValueError("Confirmation requires an immutable full freeze manifest")
    frozen = json.loads(path.read_text())
    if not frozen.get("frozen_at"):
        raise ValueError("Missing freeze timestamp")
    for argument, key in (
        ("spatial_weights", "spatial_weights_sha256"),
        ("temporal_weights", "temporal_weights_sha256"),
        ("event_gate_weights", "event_gate_weights_sha256"),
        ("calibration", "calibration_sha256"), ("config", "config_sha256"),
    ):
        artifact = kwargs.get(argument)
        if frozen.get(key) != (sha256(artifact) if artifact is not None else None):
            raise ValueError(f"Frozen experiment mismatch: {key}")
    current = experiment_boundary(**kwargs)
    for key, value in current.items():
        if frozen.get(key) != value:
            raise ValueError(f"Frozen experiment mismatch: {key}")
    return frozen

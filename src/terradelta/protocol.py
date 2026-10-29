"""Hash the complete confirmation boundary before any held-out scoring."""

from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from .data import sha256

PROTOCOL_FILES = [
    "pyproject.toml",
    "uv.lock",
    "src/terradelta/acquire.py",
    "src/terradelta/data.py",
    "src/terradelta/spatial.py",
    "src/terradelta/features.py",
    "src/terradelta/temporal.py",
    "src/terradelta/decoder.py",
    "src/terradelta/pipeline.py",
    "src/terradelta/evaluation.py",
    "src/terradelta/schema.py",
    "src/terradelta/store.py",
    "src/terradelta/service.py",
    "src/terradelta/cli.py",
    "src/terradelta/protocol.py",
    "scripts/train_spatial.py",
    "scripts/train_temporal.py",
    "scripts/calibrate.py",
    "scripts/evaluate.py",
    "scripts/predict_all.py",
    "scripts/rebuild_features.py",
    "scripts/check_online_invariance.py",
    "scripts/stress_and_calibration.py",
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
) -> dict:
    """Hash bytes without interpreting confirmation labels/outcomes.

    Frozen truth-cache bytes ensure scoring uses the same prepared data. Image and
    probability provenance are bound in sites.json plus explicit checkpoint hashes.
    Host-local image paths are excluded from semantic metadata hashing.
    """
    inputs = {}
    for region in regions:
        feature = features_root / region
        cache = cache_root / region
        info = json.loads((feature / "sites.json").read_text())
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
        "format": "terradelta-full-freeze-v2",
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
    current = experiment_boundary(**kwargs)
    for key, value in current.items():
        if frozen.get(key) != value:
            raise ValueError(f"Frozen experiment mismatch: {key}")
    return frozen

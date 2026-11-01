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
    "src/terradelta/__init__.py",
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
    "src/terradelta/embeddings.py",
    "scripts/extract_embeddings.py",
    "scripts/calibrate_embedding.py",
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
    embedding_root: Path | None = None,
) -> dict:
    """Hash bytes without interpreting confirmation labels/outcomes.

    Frozen truth-cache bytes ensure scoring uses the same prepared data. Image and
    probability provenance are bound in sites.json plus explicit checkpoint hashes.
    Host-local image paths are excluded from semantic metadata hashing.
    """
    settings = json.loads(config.read_text())
    from .contracts import (
        assert_calibration, head_training_config_sha256, preflight_heads, preflight_implementation,
        preflight_manifests, preflight_spatial_checkpoint, study_binding,
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
        config_sha256=head_training_config_sha256(config, settings)
        if spatial_binding is not None else None,
    )
    assert_calibration(
        model_info, json.loads(calibration.read_text()), temporal_weights=temporal_weights,
        event_gate_weights=event_gate_weights, require_bound=spatial_binding is not None,
        config_sha256=sha256(config) if spatial_binding is not None else None,
    )
    expected_training = head_training_config_sha256(config, settings)
    if (settings.get("head_training_config") is not None
            and json.loads(calibration.read_text()).get("head_training_config_sha256")
            != expected_training):
        raise ValueError("Calibration head-training identity differs from the actual fit")
    preflight_manifests(
        features_root, cache_root, regions, spatial_binding, supervision=False,
    )
    embedding_inputs = {}
    if settings.get("embedding_baseline") is not None:
        from .embeddings import descriptor_plan, load_embedding_data

        embedding_calibration = json.loads(calibration.read_text()).get("embedding_distance", {})
        if (embedding_root is None or settings["embedding_baseline"] != descriptor_plan()
                or embedding_calibration.get("descriptor") != settings["embedding_baseline"]):
            raise ValueError("Study requires its frozen neural embedding comparison")
        if embedding_calibration.get("selection_source_sha256") != sha256(
            repo / "scripts/calibrate_embedding.py"
        ):
            raise ValueError("Embedding calibration selection source changed")
        if embedding_calibration.get("selection_sources_sha256") != {
            name: sha256(repo / name) for name in PROTOCOL_FILES
        }:
            raise ValueError("Embedding calibration dependencies changed")
        # Check every image-only sidecar before hashing any confirmation truth.
        for region in regions:
            directory = Path(embedding_root) / region
            load_embedding_data(
                directory, feature_dir=features_root / region, cache_dir=cache_root / region,
                expected_plan=settings["embedding_baseline"],
                expected_config_sha256=sha256(config),
            )
            embedding_inputs[region] = {
                "embeddings_sha256": sha256(directory / "embeddings.npz"),
                "manifest_sha256": sha256(directory / "embedding.json"),
            }
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
    result = {
        "format": "terradelta-full-freeze-v5" if settings.get("embedding_baseline")
        else "terradelta-full-freeze-v4",
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
    if embedding_inputs:
        result["embedding_inputs"] = embedding_inputs
    if settings.get("head_training_config") is not None:
        result["head_training_config_sha256"] = expected_training
    return result


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
    settings = json.loads(kwargs["config"].read_text())
    expected_format = "terradelta-full-freeze-v5" if settings.get("embedding_baseline") \
        else "terradelta-full-freeze-v4"
    if frozen.get("format") != expected_format:
        raise ValueError("Frozen experiment mismatch: format")
    if settings.get("head_training_config") is not None:
        from .contracts import head_training_config_sha256

        if frozen.get("head_training_config_sha256") != head_training_config_sha256(
            kwargs["config"], settings
        ):
            raise ValueError("Frozen experiment mismatch: head_training_config_sha256")
    if settings.get("embedding_baseline") is not None and not isinstance(
        frozen.get("embedding_inputs"), dict
    ):
        raise ValueError("Frozen experiment mismatch: missing embedding_inputs")
    # Recheck every frozen source and image-only input before any truth bytes.
    # A self-consistent replacement sidecar is still a different experiment.
    repo = kwargs["repo"]
    sources = {name: sha256(repo / name) for name in PROTOCOL_FILES}
    if frozen.get("protocol_sources") != sources:
        raise ValueError("Frozen experiment mismatch: protocol_sources")
    if frozen.get("confirmation_regions") != kwargs["regions"]:
        raise ValueError("Frozen experiment mismatch: confirmation_regions")
    region_inputs = frozen.get("region_inputs", {})
    if set(region_inputs) != set(kwargs["regions"]):
        raise ValueError("Frozen experiment mismatch: region_inputs")
    for region in kwargs["regions"]:
        feature, cache = kwargs["features_root"] / region, kwargs["cache_root"] / region
        actual = {
            "features_sha256": sha256(feature / "features.npz"),
            "sites_sha256": sha256(feature / "sites.json"),
            "sites_raster_sha256": sha256(feature / "sites.npy"),
            "metadata_semantic_sha256": semantic_metadata_hash(cache / "metadata.json"),
            "rgb_sha256": sha256(cache / "rgb.npy"),
            "valid_sha256": sha256(cache / "valid.npy"),
        }
        if any(region_inputs[region].get(key) != value for key, value in actual.items()):
            raise ValueError(f"Frozen experiment mismatch: image inputs {region}")
    if "embedding_inputs" in frozen:
        root, expected = kwargs.get("embedding_root"), frozen["embedding_inputs"]
        if root is None or set(expected) != set(kwargs["regions"]):
            raise ValueError("Frozen experiment mismatch: embedding_inputs")
        for region in kwargs["regions"]:
            directory = Path(root) / region
            actual = {
                "embeddings_sha256": sha256(directory / "embeddings.npz"),
                "manifest_sha256": sha256(directory / "embedding.json"),
            }
            if actual != expected[region]:
                raise ValueError(f"Frozen experiment mismatch: embedding_inputs {region}")
    current = experiment_boundary(**kwargs)
    for key, value in current.items():
        if frozen.get(key) != value:
            raise ValueError(f"Frozen experiment mismatch: {key}")
    return frozen

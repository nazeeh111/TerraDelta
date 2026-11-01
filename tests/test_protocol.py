import json
from pathlib import Path
import shutil
from dataclasses import asdict

import numpy as np
import pytest
import torch

from terradelta.data import sha256
from terradelta.protocol import (
    PROTOCOL_FILES,
    freeze_experiment,
    validate_experiment,
    semantic_metadata_hash,
)


def operating_points():
    from terradelta.pipeline import METHODS, GATE_METHODS
    from terradelta.decoder import DecoderConfig

    return {
        "occupancy": {kind: {"scale": 1., "bias": 0.} for kind in METHODS if kind not in GATE_METHODS},
        "decoders": {kind: asdict(DecoderConfig()) for kind in METHODS if kind not in GATE_METHODS},
        "event_gates": {kind: {"scale": 1., "bias": 0., "threshold": .5, "timing_padding_days": 0}
                        for kind in GATE_METHODS},
    }


def heads(root, binding=None, config_sha256=None):
    from terradelta.temporal import make_temporal
    from terradelta.features import FEATURE_NAMES
    from terradelta.event_gate import (
        EventGate, FORMAT, GATE_FEATURE_NAMES, QUALITY_DROP_INDICES, RAW_CONFIG,
    )

    temporal = {
        "format": "terradelta-temporal-v1", "feature_names": FEATURE_NAMES,
        "models": {kind: make_temporal(kind).state_dict()
                   for kind in ("observation", "no_quality", "watch_lstm")},
    }
    gate = {
        "format": FORMAT, "feature_names": GATE_FEATURE_NAMES,
        "models": {kind: EventGate().state_dict() for kind in ("observation_gate", "gate_no_quality")},
        "mean": torch.zeros(len(GATE_FEATURE_NAMES)), "std": torch.ones(len(GATE_FEATURE_NAMES)),
        "calibration": {kind: {"scale": 1., "bias": 0.}
                        for kind in ("observation_gate", "gate_no_quality")},
        "raw_decoder": asdict(RAW_CONFIG),
        "quality_drop_indices": list(QUALITY_DROP_INDICES),
        "spatial_weights_sha256": sha256(root / "spatial.pt"),
    }
    if binding is not None:
        temporal.update(spatial_provenance=binding, config_sha256=config_sha256)
        gate.update(spatial_provenance=binding, training={"config_sha256": config_sha256})
    torch.save(temporal, root / "temporal.pt")
    torch.save(gate, root / "gate.pt")


def boundary(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    clone = tmp_path / "source"
    for relative in PROTOCOL_FILES:
        destination = clone / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / relative, destination)
    features = tmp_path / "features" / "area"
    features.mkdir(parents=True)
    cache = tmp_path / "cache" / "area"
    cache.mkdir(parents=True)
    meta = {
        "region_id": "area",
        "frames": [{"local_image": "/first/host/image.tif", "sha256": "a" * 64}],
    }
    (cache / "metadata.json").write_text(json.dumps(meta))
    for name in ["rgb", "valid", "instances"]:
        np.save(cache / f"{name}.npy", np.zeros((1, 4, 4), dtype="uint8"))
    np.savez(
        features / "features.npz",
        x=np.zeros((1, 1, 17)),
        usable=np.ones((1, 1), bool),
        births=np.array([0]),
    )
    np.save(features / "sites.npy", np.ones((4, 4), dtype="int32"))
    for name in ["spatial.pt", "temporal.pt", "calibration.json", "config.json"]:
        (tmp_path / name).write_text("{}")
    from terradelta.spatial import BuildingUNet

    torch.save({"format": "terradelta-spatial-v2", "architecture": {"base_channels": 4},
                "state_dict": BuildingUNet(base_channels=4).state_dict()}, tmp_path / "spatial.pt")
    binding = {
        "spatial_weights_sha256": sha256(tmp_path / "spatial.pt"),
        "spatial_backbone_sha256": None, "candidate_threshold": .55,
        "spatial_inference": {
            "tile_size": 128, "context_halo_native_pixels": 64,
            "native_alignment": 8, "internal_input_scale": 1,
            "implementation_sha256": {"spatial.py": sha256(repo / "src/terradelta/spatial.py")},
        },
    }
    (tmp_path / "config.json").write_text(json.dumps({"event_gate": {"spatial_provenance": binding}}))
    heads(tmp_path, binding, sha256(tmp_path / "config.json"))
    (tmp_path / "calibration.json").write_text(json.dumps({
        **operating_points(),
        "spatial_provenance": binding, "spatial_weights_sha256": binding["spatial_weights_sha256"],
        "weights_sha256": sha256(tmp_path / "temporal.pt"),
        "event_gate_weights_sha256": sha256(tmp_path / "gate.pt"),
        "config_sha256": sha256(tmp_path / "config.json"),
    }))
    info = {
        "feature_protocol": "calendar_quality_v2",
        "supervision_protocol": None,
        "region_id": "area",
        "spatial_weights_sha256": sha256(tmp_path / "spatial.pt"),
        "metadata_semantic_sha256": semantic_metadata_hash(cache / "metadata.json"),
        "probability_sha256": "b" * 64,
        **binding,
    }
    (features / "sites.json").write_text(json.dumps(info))
    return dict(
        repo=clone,
        features_root=tmp_path / "features",
        cache_root=tmp_path / "cache",
        regions=["area"],
        temporal_weights=tmp_path / "temporal.pt",
        spatial_weights=tmp_path / "spatial.pt",
        calibration=tmp_path / "calibration.json",
        config=tmp_path / "config.json",
        event_gate_weights=tmp_path / "gate.pt",
    )


def test_full_freeze_binds_spatial_features_truth_and_semantics(tmp_path):
    kwargs = boundary(tmp_path)
    path = tmp_path / "freeze.json"
    freeze_experiment(path, **kwargs)
    validate_experiment(path, **kwargs)
    with pytest.raises(ValueError, match="immutable"):
        freeze_experiment(path, **kwargs)
    for candidate in [
        tmp_path / "spatial.pt",
        tmp_path / "features/area/features.npz",
        tmp_path / "cache/area/instances.npy",
        tmp_path / "source/src/terradelta/decoder.py",
        tmp_path / "source/src/terradelta/pretrained_spatial.py",
    ]:
        original = candidate.read_bytes()
        candidate.write_bytes(original + b"changed")
        with pytest.raises(ValueError):
            validate_experiment(path, **kwargs)
        candidate.write_bytes(original)
    validate_experiment(path, **kwargs)


def test_freeze_rejects_inference_mixture_before_truth_hashing(tmp_path):
    kwargs = boundary(tmp_path)
    info_path = tmp_path / "features/area/sites.json"
    info = json.loads(info_path.read_text())
    binding = {
        "spatial_weights_sha256": info["spatial_weights_sha256"],
        "candidate_threshold": 0.55,
        "spatial_backbone_sha256": None,
        "spatial_inference": {
            "tile_size": 128,
            "context_halo_native_pixels": 64,
            "native_alignment": 8,
            "internal_input_scale": 1,
            "implementation_sha256": {"spatial.py": sha256(kwargs["repo"] / "src/terradelta/spatial.py")},
        },
    }
    info.update(binding)
    info_path.write_text(json.dumps(info))
    kwargs["config"].write_text(json.dumps({"event_gate": {"spatial_provenance": binding}}))
    gate = tmp_path / "gate.pt"
    heads(tmp_path, binding, sha256(kwargs["config"]))
    kwargs["calibration"].write_text(json.dumps({
        **operating_points(),
        "spatial_provenance": binding, "spatial_weights_sha256": binding["spatial_weights_sha256"],
        "weights_sha256": sha256(kwargs["temporal_weights"]), "event_gate_weights_sha256": sha256(gate),
        "config_sha256": sha256(kwargs["config"]),
    }))
    kwargs["event_gate_weights"] = gate
    freeze_experiment(tmp_path / "valid.json", **kwargs)
    info["spatial_inference"]["tile_size"] = 256
    info_path.write_text(json.dumps(info))
    # A missing truth cache must not mask the earlier inference mismatch.
    (tmp_path / "cache/area/instances.npy").unlink()
    with pytest.raises(ValueError, match="spatial model/backbone/inference"):
        freeze_experiment(tmp_path / "mixed.json", **kwargs)


def test_local_host_paths_are_not_experiment_identity(tmp_path):
    kwargs = boundary(tmp_path)
    path = tmp_path / "freeze.json"
    freeze_experiment(path, **kwargs)
    metadata = tmp_path / "cache/area/metadata.json"
    doc = json.loads(metadata.read_text())
    doc["frames"][0]["local_image"] = "/another/host/image.tif"
    metadata.write_text(json.dumps(doc))
    validate_experiment(path, **kwargs)


def test_old_supervision_semantics_are_rejected(tmp_path):
    from terradelta.features import validate_feature_protocol

    root = tmp_path / "features"
    root.mkdir()
    (root / "sites.json").write_text(
        json.dumps(
            {
                "feature_protocol": "calendar_quality_v2",
                "supervision_protocol": "legacy_presence_25percent",
            }
        )
    )
    with pytest.raises(ValueError, match="Supervision protocol"):
        validate_feature_protocol(root, supervision=True)


def test_gate_experiment_requires_and_binds_gate_weights(tmp_path):
    kwargs = boundary(tmp_path)
    kwargs.pop("event_gate_weights")
    with pytest.raises(ValueError, match="gate weights"):
        freeze_experiment(tmp_path / "missing.json", **kwargs)
    gate = tmp_path / "gate.pt"
    kwargs["event_gate_weights"] = gate
    frozen = tmp_path / "freeze.json"
    freeze_experiment(frozen, **kwargs)
    validate_experiment(frozen, **kwargs)
    gate.write_bytes(b"different weights")
    with pytest.raises(ValueError, match="event_gate_weights"):
        validate_experiment(frozen, **kwargs)


def test_gate_reliability_is_proposal_weighted():
    import importlib.util
    script = Path(__file__).resolve().parents[1] / 'scripts/evaluate_gate_calibration.py'
    spec = importlib.util.spec_from_file_location('gate_reliability_test', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # One repeated proposal and one single-row proposal carry equal mass.
    result = module.reliability(np.array([.1, .1, .9]), np.array([0, 0, 1]), np.array([.5, .5, 1.]))
    assert result['proposal_mass'] == 2
    assert result['positive_fraction'] == .5
    assert result['brier'] == pytest.approx(.01)
    assert sum(item['proposal_mass'] for item in result['bins']) == pytest.approx(2)

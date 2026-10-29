import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from terradelta.data import sha256
from terradelta.protocol import (
    PROTOCOL_FILES,
    freeze_experiment,
    validate_experiment,
    semantic_metadata_hash,
)


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
    info = {
        "feature_protocol": "calendar_quality_v2",
        "supervision_protocol": None,
        "region_id": "area",
        "spatial_weights_sha256": sha256(tmp_path / "spatial.pt"),
        "metadata_semantic_sha256": semantic_metadata_hash(cache / "metadata.json"),
        "probability_sha256": "b" * 64,
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
    ]:
        original = candidate.read_bytes()
        candidate.write_bytes(original + b"changed")
        with pytest.raises(ValueError):
            validate_experiment(path, **kwargs)
        candidate.write_bytes(original)
    validate_experiment(path, **kwargs)


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
    kwargs["config"].write_text(json.dumps({"event_gate": {"truth_ready": True}}))
    with pytest.raises(ValueError, match="gate weights"):
        freeze_experiment(tmp_path / "missing.json", **kwargs)
    gate = tmp_path / "gate.pt"
    gate.write_bytes(b"original trained gate")
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

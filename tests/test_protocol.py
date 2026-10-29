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

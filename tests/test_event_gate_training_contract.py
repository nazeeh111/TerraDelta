"""Provenance rejection uses synthetic manifests, never study labels or weights."""
import copy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

import terradelta.event_gate as gate
from terradelta.features import FEATURE_NAMES


def trainer():
    path = Path(__file__).resolve().parents[1] / "scripts/train_event_gate.py"
    spec = importlib.util.spec_from_file_location("gate_provenance_trainer_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def binding():
    return {
        "spatial_weights_sha256": "a" * 64,
        "candidate_threshold": 0.55,
        "spatial_backbone_sha256": "b" * 64,
        "spatial_inference": {
            "tile_size": 1024, "context_halo_native_pixels": 64,
            "native_alignment": 16, "internal_input_scale": 2,
            "implementation_sha256": {"spatial.py": "c" * 64,
                                      "pretrained_spatial.py": "d" * 64},
        },
    }


def info(region, provenance):
    return {"region_id": region, "feature_names": FEATURE_NAMES,
            "feature_protocol": "calendar_quality_v2",
            "supervision_protocol": "physical_coverage_80_20_v2",
            "metadata_semantic_sha256": "metadata-unit-hash", **copy.deepcopy(provenance)}


def write_info(root, region, value):
    directory = root / region
    directory.mkdir(parents=True)
    (directory / "sites.json").write_text(json.dumps(value))


def test_new_configuration_requires_complete_explicit_binding():
    module = trainer()
    with pytest.raises(ValueError, match="requires explicit"):
        module.expected_spatial_provenance({"version": "new-study", "event_gate": {}})
    for missing in gate.SPATIAL_PROVENANCE_KEYS:
        incomplete = binding()
        del incomplete[missing]
        with pytest.raises(ValueError, match="requires model"):
            module.expected_spatial_provenance({"event_gate": {"spatial_provenance": incomplete}})
    with pytest.raises(ValueError, match="differs from study"):
        module.expected_spatial_provenance({"event_gate": {"spatial_provenance": binding()},
                                            "spatial_backbone": {"sha256": "f" * 64}})


@pytest.mark.parametrize("change", ["model", "backbone", "missing_backbone", "missing_inference",
                                     "null_inference", "tiling", "implementation", "candidate"])
def test_mixed_development_provenance_rejected_before_any_arrays_or_truth(
    tmp_path, monkeypatch, change,
):
    module = trainer()
    expected = binding()
    altered = info("development-region", expected)
    if change == "model":
        altered["spatial_weights_sha256"] = "e" * 64
    elif change == "backbone":
        altered["spatial_backbone_sha256"] = "e" * 64
    elif change == "missing_backbone":
        del altered["spatial_backbone_sha256"]
    elif change == "missing_inference":
        del altered["spatial_inference"]
    elif change == "null_inference":
        altered["spatial_inference"] = None
    elif change == "tiling":
        altered["spatial_inference"]["tile_size"] = 256
    elif change == "candidate":
        altered["candidate_threshold"] = 0.45
    else:
        altered["spatial_inference"]["implementation_sha256"]["spatial.py"] = "e" * 64
    features = tmp_path / "features"
    write_info(features, "train-region", info("train-region", expected))
    write_info(features, "development-region", altered)
    config = {
        "version": "new-study", "train": ["train-region"], "development": ["development-region"],
        "confirmation": ["NEVER-OPEN"], "seed": 2901, "truth_protocol": "physical_coverage_v2",
        "event_gate": {"truth_ready": True, "proposal_threshold": .55, "confirmations": 2,
                       "minimum_dwell_days": 20, "training_steps": 1,
                       "spatial_provenance": expected,
                       "truth_source_sha256": module.sha256(
                           Path(module.__file__).parents[1] / "src/terradelta/evaluation.py"),
                       "supervision_source_sha256": module.sha256(
                           Path(module.__file__).parents[1] / "src/terradelta/features.py")},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))

    def forbidden(*args, **kwargs):
        pytest.fail("array/truth/proposal read occurred before all-region preflight")

    monkeypatch.setattr(module, "semantic_metadata_hash", lambda path: "metadata-unit-hash")
    monkeypatch.setattr(module, "ground_truth_events", forbidden)
    monkeypatch.setattr(module, "raw_proposals", forbidden)
    monkeypatch.setattr(module.np, "load", forbidden)
    monkeypatch.setattr(sys, "argv", ["train_event_gate", "--features-root", str(features),
                                     "--cache-root", str(tmp_path / "no-arrays"),
                                     "--config", str(path), "--output", str(tmp_path / "new.pt")])
    with pytest.raises(ValueError, match="spatial|backbone|inference"):
        module.main()
    assert not (tmp_path / "new.pt").exists()


def test_historical_v4_accepts_missing_or_null_optional_fields_but_not_new_encoder():
    module = trainer()
    assert module.expected_spatial_provenance({"version": "sn7-regions-v4-event-gate"}) is None
    legacy = info("old-region", {"spatial_weights_sha256": module.SPATIAL_V2_SHA256})
    module.validate_region_provenance(legacy, "old-region", None)
    legacy.update(spatial_backbone_sha256=None, spatial_inference=None)
    module.validate_region_provenance(legacy, "old-region", None)
    legacy["spatial_backbone_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="legacy"):
        module.validate_region_provenance(legacy, "old-region", None)
    with pytest.raises(ValueError, match="requires explicit"):
        module.expected_spatial_provenance({"version": "sn7-regions-v4-event-gate",
                                           "spatial_backbone": {"sha256": "b" * 64}})


def test_exact_binding_is_detached_and_checks_grid_types_and_source_hashes():
    expected = binding()
    detached = gate.validate_spatial_provenance(expected)
    detached["spatial_inference"]["tile_size"] = 256
    assert expected["spatial_inference"]["tile_size"] == 1024
    gate.assert_spatial_provenance(info("unit", expected), expected)
    wrong = info("unit", expected)
    wrong["spatial_inference"]["tile_size"] = 1024.0
    with pytest.raises(ValueError, match="integers"):
        gate.assert_spatial_provenance(wrong, expected)
    wrong["spatial_inference"]["tile_size"] = 1024
    wrong["spatial_inference"]["implementation_sha256"]["spatial.py"] = "short-sha"
    with pytest.raises(ValueError, match="full implementation"):
        gate.assert_spatial_provenance(wrong, expected)


def checkpoint(provenance):
    torch.manual_seed(78)
    states = {kind: gate.EventGate().state_dict() for kind in gate.KINDS[:2]}
    return {"format": gate.FORMAT, "feature_names": gate.GATE_FEATURE_NAMES,
            "models": states, "mean": torch.zeros(78), "std": torch.ones(78),
            "calibration": {kind: {"scale": 1., "bias": 0.} for kind in states},
            "raw_decoder": asdict(gate.RAW_CONFIG),
            "quality_drop_indices": list(gate.QUALITY_DROP_INDICES),
            "spatial_weights_sha256": provenance["spatial_weights_sha256"],
            "spatial_provenance": provenance}


def test_new_gate_roundtrip_keeps_binding_and_rejects_disagreeing_identity(tmp_path):
    state = checkpoint(binding())
    path = tmp_path / "synthetic-gate.pt"
    torch.save(state, path)
    loaded = gate.load_gate(path)
    assert loaded.spatial_provenance == binding()
    assert np.isfinite(loaded.logit(np.zeros(78, np.float32), "observation_gate"))
    state["spatial_weights_sha256"] = "e" * 64
    torch.save(state, tmp_path / "disagree.pt")
    with pytest.raises(ValueError, match="identity fields disagree"):
        gate.load_gate(tmp_path / "disagree.pt")


def test_inference_rejects_binding_before_raw_decode(tmp_path, monkeypatch):
    path = tmp_path / "synthetic-gate.pt"
    torch.save(checkpoint(binding()), path)
    loaded = gate.load_gate(path)
    wrong = info("unit", binding())
    wrong["spatial_inference"]["tile_size"] = 256

    def forbidden(*args, **kwargs):
        pytest.fail("raw proposals decoded before rejecting incompatible inference provenance")

    monkeypatch.setattr(gate, "raw_proposals", forbidden)
    with pytest.raises(ValueError, match="differs from frozen provenance"):
        gate.generate_events(np.empty((0, 0, 17)), np.empty((0, 0), bool), np.array([]), wrong, loaded)


def test_legacy_gate_cannot_accept_new_backbone_even_with_matching_model_hash(tmp_path, monkeypatch):
    state = checkpoint(binding())
    del state["spatial_provenance"]
    path = tmp_path / "synthetic-legacy-gate.pt"
    torch.save(state, path)
    loaded = gate.load_gate(path)
    assert loaded.spatial_provenance is None

    def forbidden(*args, **kwargs):
        pytest.fail("legacy gate decoded new-encoder features")

    monkeypatch.setattr(gate, "raw_proposals", forbidden)
    with pytest.raises(ValueError, match="legacy gate"):
        gate.generate_events(np.empty((0, 0, 17)), np.empty((0, 0), bool), np.array([]),
                             info("unit", binding()), loaded)

"""Regression probes at the label boundary, using valid synthetic model states."""

import json
import pickle

import pytest
import torch

from test_protocol import boundary
from terradelta import protocol
from terradelta.contracts import assert_calibration, preflight_implementation
from terradelta.event_gate import assert_legacy_spatial_compatibility, load_gate


@pytest.mark.parametrize("damage", [
    "invalid_gate", "unbound_temporal", "wrong_gate_binding", "missing_gate_fit",
    "wrong_calibration_head", "missing_calibration_head", "wrong_calibration_candidate",
    "missing_operating_point",
])
def test_invalid_models_or_calibration_fail_before_confirmation_truth(tmp_path, monkeypatch, damage):
    kwargs = boundary(tmp_path)
    if damage == "invalid_gate":
        kwargs["event_gate_weights"].write_bytes(b"not a neural checkpoint")
    elif damage == "unbound_temporal":
        state = torch.load(kwargs["temporal_weights"], weights_only=True)
        state.pop("spatial_provenance")
        torch.save(state, kwargs["temporal_weights"])
    elif damage in {"wrong_gate_binding", "missing_gate_fit"}:
        state = torch.load(kwargs["event_gate_weights"], weights_only=True)
        if damage == "wrong_gate_binding":
            state["spatial_provenance"]["candidate_threshold"] = .45
        else:
            state["calibration"] = {}
        torch.save(state, kwargs["event_gate_weights"])
    else:
        calibration = json.loads(kwargs["calibration"].read_text())
        if damage == "wrong_calibration_head":
            calibration["event_gate_weights_sha256"] = "d" * 64
        elif damage == "missing_calibration_head":
            calibration.pop("event_gate_weights_sha256")
        elif damage == "wrong_calibration_candidate":
            calibration["spatial_provenance"]["candidate_threshold"] = .45
        else:
            calibration["event_gates"].pop("observation_gate")
        kwargs["calibration"].write_text(json.dumps(calibration))
    real_sha = protocol.sha256

    def no_truth_before_valid_models(path):
        if path.name == "instances.npy":
            pytest.fail("Confirmation truth hashed before model/calibration validation")
        return real_sha(path)

    monkeypatch.setattr(protocol, "sha256", no_truth_before_valid_models)
    with pytest.raises((ValueError, RuntimeError, KeyError, EOFError, pickle.UnpicklingError)):
        protocol.freeze_experiment(tmp_path / "invalid-freeze.json", **kwargs)
    assert not (tmp_path / "invalid-freeze.json").exists()


def test_versioned_calibration_requires_selected_head_hash(tmp_path):
    kwargs = boundary(tmp_path)
    info = json.loads((kwargs["features_root"] / "area/sites.json").read_text())
    calibration = json.loads(kwargs["calibration"].read_text())
    calibration.pop("event_gate_weights_sha256")
    with pytest.raises(ValueError, match="event_gate_weights_sha256"):
        assert_calibration(info, calibration, method="observation_gate",
                           event_gate_weights=kwargs["event_gate_weights"])


def test_gate_loader_preserves_valid_one_class_constant_calibration(tmp_path):
    kwargs = boundary(tmp_path)
    state = torch.load(kwargs["event_gate_weights"], weights_only=True)
    for fit in state["calibration"].values():
        fit.update(scale=0., bias=-4., calibration_kind="constant_one_class_development")
    torch.save(state, kwargs["event_gate_weights"])
    assert all(fit["scale"] == 0 for fit in load_gate(kwargs["event_gate_weights"]).calibration.values())


def test_legacy_candidate_and_unknown_implementation_are_rejected():
    with pytest.raises(ValueError, match="legacy"):
        assert_legacy_spatial_compatibility({"candidate_threshold": .65})
    binding = {
        "spatial_weights_sha256": "a" * 64, "spatial_backbone_sha256": "b" * 64,
        "candidate_threshold": .55,
        "spatial_inference": {"tile_size": 128, "context_halo_native_pixels": 64,
                              "native_alignment": 16, "internal_input_scale": 2,
                              "implementation_sha256": {"spatial.py": "c" * 64}},
    }
    with pytest.raises(ValueError, match="implementation"):
        preflight_implementation(binding)

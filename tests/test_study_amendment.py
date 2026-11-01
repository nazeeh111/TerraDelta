"""Only the declared comparator can change when retaining actual fitted heads."""

from copy import deepcopy
import hashlib
import json

import pytest

from terradelta.contracts import head_training_config_sha256, preflight_probability_files


def fixture(tmp_path):
    training = {"version": "training", "seed": 2901, "train": ["a", "b"],
                "development": ["c"], "confirmation": ["d"],
                "resource_limits": {"steps": 2000}, "temporal_selection": "event_62d",
                "truth_protocol": "physical_coverage_v2", "candidate_threshold": .55,
                "event_gate": {"spatial_provenance": {"checkpoint": "a" * 64}},
                "embedding_baseline": {"version": 1}, "parent_experiment": "v6"}
    raw = json.dumps(training).encode()
    (tmp_path / "training.json").write_bytes(raw)
    expected = hashlib.sha256(raw).hexdigest()
    study = deepcopy(training)
    study.update(version="comparison", embedding_baseline={"version": 2},
                 embedding_baseline_declared_at="before any descriptor outcomes",
                 embedding_amendment={"basis": "synthetic newborn coverage probe"},
                 head_training_config={"path": "training.json", "sha256": expected})
    path = tmp_path / "study.json"
    path.write_text(json.dumps(study))
    return path, study, expected


def test_comparison_amendment_keeps_actual_training_config_identity(tmp_path):
    path, _, expected = fixture(tmp_path)
    assert head_training_config_sha256(path) == expected
    assert head_training_config_sha256(tmp_path / "training.json") == expected


@pytest.mark.parametrize("damage", [
    "train", "train_order", "development", "confirmation", "seed", "resource_limits",
    "temporal_selection", "truth_protocol", "candidate_threshold", "event_gate",
    "unknown_new_field", "parent_experiment", "wrong_sha", "traversal", "self", "chain",
    "seed_type", "resource_type",
])
def test_amendment_rejects_every_other_field_or_reference_change(tmp_path, damage):
    path, study, _ = fixture(tmp_path)
    if damage == "train_order":
        study["train"].reverse()
    elif damage == "seed_type":
        study["seed"] = float(study["seed"])
    elif damage == "resource_type":
        study["resource_limits"]["steps"] = 2000.0
    elif damage == "wrong_sha":
        study["head_training_config"]["sha256"] = "0" * 64
    elif damage == "traversal":
        study["head_training_config"]["path"] = "../training.json"
    elif damage == "self":
        study["head_training_config"]["path"] = "study.json"
    elif damage == "chain":
        training = json.loads((tmp_path / "training.json").read_text())
        training["head_training_config"] = {"path": "another.json", "sha256": "0" * 64}
        raw = json.dumps(training).encode()
        (tmp_path / "training.json").write_bytes(raw)
        study["head_training_config"]["sha256"] = hashlib.sha256(raw).hexdigest()
    else:
        study[damage] = "changed"
    path.write_text(json.dumps(study))
    with pytest.raises(ValueError):
        head_training_config_sha256(path)


def test_diagnostic_probability_bytes_are_checked_for_every_region(tmp_path):
    features, predictions = tmp_path / "features", tmp_path / "predictions"
    for region in ("a", "b"):
        (features / region).mkdir(parents=True)
        (predictions / region).mkdir(parents=True)
        raw = region.encode()
        (predictions / region / "probabilities.npy").write_bytes(raw)
        (features / region / "sites.json").write_text(json.dumps({
            "feature_protocol": "calendar_quality_v2", "supervision_protocol": None,
            "probability_sha256": hashlib.sha256(raw).hexdigest(),
        }))
    preflight_probability_files(features, predictions, ["a", "b"])
    (predictions / "b/probabilities.npy").write_bytes(b"stale other model")
    with pytest.raises(ValueError, match="b"):
        preflight_probability_files(features, predictions, ["a", "b"])

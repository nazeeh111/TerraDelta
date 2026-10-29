import copy
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch import nn

from terradelta.event_gate import (
    FORMAT, GATE_FEATURE_NAMES, QUALITY_DROP_INDICES, RAW_CONFIG, EventGate, GateBundle,
    event_features, generate_events, load_gate, raw_proposals, standardize,
)
from terradelta.features import FEATURE_NAMES
from terradelta.store import EventStore

REC = datetime(2026, 1, 1, tzinfo=timezone.utc)


def context(months):
    frames = [{"source_id": f"month-{month}", "uri": f"https://example.org/{month}.tif",
               "sha256": f"{month:064x}",
               "acquisition_start": f"2020-{month:02d}-01T00:00:00+00:00",
               "acquisition_end": f"2020-{month:02d}-28T23:59:59+00:00",
               "published_at": None, "availability_assumption": "retrospective_month_end",
               "registration": {"usable": True, "shift_magnitude_pixels": 0}}
              for month in range(1, months + 1)]
    return {"region_id": "unit-gate", "metadata": {"frames": frames},
            "feature_names": FEATURE_NAMES, "spatial_weights_sha256": "a" * 64,
            "geometries": {"1": {"type": "Polygon", "coordinates": [
                [[0, 0], [0.001, 0], [0.001, 0.001], [0, 0.001], [0, 0]]
            ]}}}


def cube(probability):
    x = np.zeros((1, len(probability), 17), np.float32)
    x[0, :, 0] = probability
    x[0, :, 3] = 1
    x[0, :, 16] = 1
    return x


class MarginGate(nn.Module):
    def forward(self, values):
        return 20 * (values[:, :1] - 0.8)


def bundle():
    return GateBundle(
        {kind: MarginGate() for kind in ("observation_gate", "gate_no_quality")},
        np.zeros(len(GATE_FEATURE_NAMES), np.float32),
        np.ones(len(GATE_FEATURE_NAMES), np.float32),
        {kind: {"scale": 1., "bias": 0.} for kind in ("observation_gate", "gate_no_quality")},
        "unit-gate-weights", "a" * 64,
    )


def test_same_raw_policy_then_ambiguous_proposal_confirms_later():
    x = cube([0.1, 0.1, 0.6, 0.6, 0.95, 0.95])
    usable = np.ones(x.shape[:2], bool)
    info = context(6)
    raw = raw_proposals(x, usable, np.array([0]), info, record_origin=REC)
    baseline = generate_events(x, usable, np.array([0]), info, kind="proposal_tracking",
                               record_origin=REC)
    assert [e.model_dump(mode="json") for e in raw] == [e.model_dump(mode="json") for e in baseline]
    gated = generate_events(x, usable, np.array([0]), info, bundle(), record_origin=REC)
    assert [e.state for e in gated] == ["proposed", "proposed", "confirmed_by_model"]
    assert [e.revision for e in gated] == [1, 2, 3]
    assert len({e.event_id for e in gated}) == 1
    assert gated[-1].event_id == raw[0].event_id
    assert raw[-1].detected_at.month == 4 and gated[-1].detected_at.month == 5
    assert all(source.acquisition_end.month <= 5 for source in gated[-1].sources)


def test_raw_reversal_retracts_same_id_and_preserves_ledger_history(tmp_path):
    x = cube([0.1, 0.1, 0.6, 0.6, 0.95, 0.05, 0.05])
    events = generate_events(x, np.ones(x.shape[:2], bool), np.array([0]), context(7),
                             bundle(), record_origin=REC)
    first_id = events[0].event_id
    first = [event for event in events if event.event_id == first_id]
    assert [event.state for event in first] == ["proposed", "proposed", "confirmed_by_model", "retracted"]
    assert [event.revision for event in first] == [1, 2, 3, 4]
    assert first[-1].confidence is None
    with EventStore(tmp_path / "events.duckdb") as store:
        for event in events:
            store.append(event)
        assert len(store.history(first_id)) == 4
        assert store.query(as_of=first[2].record_time)[0].state == "confirmed_by_model"


def test_future_edits_and_appending_reversal_cannot_change_old_answers():
    x = cube([0.1, 0.1, 0.6, 0.6, 0.95, 0.95, 0.95])
    other = x.copy()
    other[0, 5:, 0] = 0.05
    other[0, 5:, 4] = 4
    good = np.ones(x.shape[:2], bool)
    info = context(7)
    a = generate_events(x, good, np.array([0]), info, bundle(), record_origin=REC)
    b = generate_events(other, good, np.array([0]), info, bundle(), record_origin=REC)
    short_info = copy.deepcopy(info)
    short_info["metadata"]["frames"] = short_info["metadata"]["frames"][:5]
    prefix = generate_events(x[:, :5], good[:, :5], np.array([0]), short_info,
                             bundle(), record_origin=REC)
    def serialize(events):
        return [event.model_dump(mode="json") for event in events if event.detected_at.month <= 5]
    assert serialize(a) == serialize(b) == serialize(prefix)


def test_invalid_observations_never_enter_scores_or_become_absence():
    x = cube([0.1, 0.1, 0.95, 0.95, 0.6, 0.6, 0.95])
    usable = np.array([[1, 1, 0, 0, 1, 1, 1]], bool)
    changed = x.copy()
    changed[~usable] = np.nan
    a = generate_events(x, usable, np.array([0]), context(7), bundle(), record_origin=REC)
    b = generate_events(changed, usable, np.array([0]), context(7), bundle(), record_origin=REC)
    assert [event.model_dump(mode="json") for event in a] == [event.model_dump(mode="json") for event in b]
    assert a[0].detected_at.month == 5 and a[-1].detected_at.month == 7
    assert all(source.acquisition_end.month not in (3, 4) for event in a for source in event.sources)


def test_feature_windows_are_prefix_only_and_quality_ablation_is_explicit():
    x = cube([0.1, 0.2, 0.6, 0.7, 0.95, 0.95])
    x[0, :, 4] = np.arange(6)
    usable = np.ones(x.shape[:2], bool)
    raw = raw_proposals(x, usable, np.array([0]), context(6), record_origin=REC)
    eligible = next(event for event in raw if event.state == "confirmed_by_model")
    values, indices = event_features(x, usable, eligible, 3, context(6))
    changed = x.copy()
    changed[:, 4:] = 999
    after, after_indices = event_features(changed, usable, eligible, 3, context(6))
    np.testing.assert_array_equal(values, after)
    assert indices == after_indices and max(indices) == 3
    full = standardize(values, np.zeros(78), np.ones(78), "observation_gate")
    ablated = standardize(values, np.zeros(78), np.ones(78), "gate_no_quality")
    assert np.all(ablated[list(QUALITY_DROP_INDICES)] == 0)
    retained = np.setdiff1d(np.arange(78), QUALITY_DROP_INDICES)
    np.testing.assert_array_equal(full[retained], ablated[retained])


def test_external_calibration_changes_identity_and_records_effective_policy():
    x = cube([0.1, 0.1, 0.6, 0.6, 0.95, 0.95])
    usable = np.ones(x.shape[:2], bool)
    common = dict(weights=bundle(), record_origin=REC)
    a = generate_events(x, usable, np.array([0]), context(6),
                        calibration={"observation_gate": {"scale": 1., "bias": 0.}}, **common)
    b = generate_events(x, usable, np.array([0]), context(6),
                        calibration={"observation_gate": {"scale": 2., "bias": 0.}}, **common)
    assert a[0].model_revision != b[0].model_revision
    assert a[1].confidence != b[1].confidence
    assert a[1].quality["event_gate_parameters"] == {
        "scale": 1., "bias": 0., "threshold": 0.5, "timing_padding_days": 0,
    }
    assert a[1].quality["raw_proposal_decoder"] == asdict(RAW_CONFIG)


def test_checkpoint_roundtrip_and_wrong_spatial_provenance_rejected(tmp_path):
    torch.manual_seed(5)
    states = {kind: EventGate().state_dict() for kind in ("observation_gate", "gate_no_quality")}
    path = tmp_path / "gate.pt"
    torch.save({"format": FORMAT, "feature_names": GATE_FEATURE_NAMES, "models": states,
                "mean": torch.zeros(78), "std": torch.ones(78),
                "calibration": {kind: {"scale": 1., "bias": 0.} for kind in states},
                "raw_decoder": asdict(RAW_CONFIG),
                "quality_drop_indices": list(QUALITY_DROP_INDICES),
                "spatial_weights_sha256": "a" * 64}, path)
    loaded = load_gate(path)
    assert np.isfinite(loaded.logit(np.zeros(78, np.float32), "observation_gate"))
    x = cube([0.1, 0.1, 0.9, 0.9])
    wrong = context(4)
    wrong["spatial_weights_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="spatial checkpoint"):
        generate_events(x, np.ones(x.shape[:2], bool), np.array([0]), wrong, loaded)


@pytest.mark.parametrize("changed_field", ["raw_decoder", "quality_drop_indices"])
def test_loader_rejects_checkpoint_runtime_policy_mismatch(tmp_path, changed_field):
    # Refuse the artifact before deserializing model states; omitted states are
    # deliberate because policy mismatch alone must be decisive.
    checkpoint = {"format": FORMAT, "feature_names": GATE_FEATURE_NAMES,
                  "raw_decoder": asdict(RAW_CONFIG),
                  "quality_drop_indices": list(QUALITY_DROP_INDICES)}
    if changed_field == "raw_decoder":
        checkpoint[changed_field]["threshold"] = 0.95
    else:
        checkpoint[changed_field] = []
    path = tmp_path / "different-policy.pt"
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="policy differs"):
        load_gate(path)


def training_module():
    path = Path(__file__).resolve().parents[1] / "scripts/train_event_gate.py"
    spec = importlib.util.spec_from_file_location("event_gate_training_for_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_platt_restores_proposal_prevalence_despite_repeated_prefix_rows():
    module = training_module()
    result = module.platt_calibration(np.zeros(4), np.array([1., 1., 1., 0.]),
                                      np.array([1 / 3, 1 / 3, 1 / 3, 1.]))
    assert result["event_prevalence"] == 0.5
    assert result["proposal_mass"] == 2
    assert abs(result["bias"]) < 1e-6


def test_training_refuses_unreviewed_truth_before_opening_any_dataset(tmp_path, monkeypatch):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"event_gate": {"truth_ready": False}}))
    monkeypatch.setattr(sys, "argv", ["train_event_gate", "--features-root", str(tmp_path / "absent"),
                                     "--cache-root", str(tmp_path / "absent"), "--config", str(config),
                                     "--output", str(tmp_path / "new.pt")])
    with pytest.raises(ValueError, match="ready/audited"):
        training_module().main()
    assert not (tmp_path / "new.pt").exists()

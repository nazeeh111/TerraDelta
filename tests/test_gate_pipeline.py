from dataclasses import asdict
from datetime import datetime, timezone
import json

import geopandas as gpd
import duckdb
import numpy as np
import pytest
import torch
from typer.testing import CliRunner

from terradelta import pipeline
from terradelta.cli import app
from terradelta.decoder import DecoderConfig
from terradelta.event_gate import (
    FORMAT, GATE_FEATURE_NAMES, QUALITY_DROP_INDICES, RAW_CONFIG, EventGate,
)
from terradelta.features import FEATURE_NAMES

ORIGIN = datetime(2026, 1, 1, tzinfo=timezone.utc)


def fixture_inputs():
    probability = [0.1, 0.1, 0.9, 0.9, 0.1, 0.1, 0.9, 0.9]
    x = np.zeros((1, len(probability), 17), np.float32)
    x[0, :, 0] = probability
    x[0, :, 3] = x[0, :, 16] = 1
    frames = [{"source_id": f"month-{month}", "uri": f"https://example.org/{month}.tif",
               "sha256": f"{month:064x}",
               "acquisition_start": f"2020-{month:02d}-01T00:00:00+00:00",
               "acquisition_end": f"2020-{month:02d}-28T23:59:59+00:00",
               "published_at": None, "availability_assumption": "retrospective_month_end",
               "registration": {"usable": True, "shift_magnitude_pixels": 0}}
              for month in range(1, 9)]
    info = {"region_id": "unit-pipeline", "metadata": {"frames": frames},
            "feature_names": FEATURE_NAMES, "feature_protocol": "calendar_quality_v2",
            "supervision_protocol": None, "spatial_weights_sha256": "a" * 64,
            "probability_sha256": "b" * 64,
            "geometries": {"1": {"type": "Polygon", "coordinates": [
                [[0, 0], [0.001, 0], [0.001, 0.001], [0, 0.001], [0, 0]]
            ]}}}
    return x, np.ones(x.shape[:2], bool), np.array([0]), info


def gate_weights(path):
    models = {}
    for kind in pipeline.GATE_METHODS[:2]:
        model = EventGate()
        for parameter in model.parameters():
            torch.nn.init.zeros_(parameter)
        model[-1].bias.data.fill_(2)
        models[kind] = model.state_dict()
    torch.save({"format": FORMAT, "feature_names": GATE_FEATURE_NAMES,
                "models": models, "mean": torch.zeros(78), "std": torch.ones(78),
                "raw_decoder": asdict(RAW_CONFIG), "quality_drop_indices": list(QUALITY_DROP_INDICES),
                "calibration": {kind: {"scale": 1., "bias": 0.} for kind in models},
                "spatial_weights_sha256": "a" * 64}, path)
    return path


def test_dispatch_all_twelve_methods_keeps_gates_out_of_occupancy_scores(tmp_path, monkeypatch):
    x, usable, births, info = fixture_inputs()
    weights = gate_weights(tmp_path / "gate.pt")
    scored = []

    def occupancy(features, mask, method, temporal_weights=None, calibration=None):
        scored.append(method)
        return features[:, :, 0]

    monkeypatch.setattr(pipeline, "method_scores", occupancy)
    assert len(set(pipeline.METHODS)) == 12
    for method in pipeline.METHODS:
        result = pipeline.method_events(x, usable, births, info, method,
                                        event_gate_weights=weights, decoder=RAW_CONFIG,
                                        record_origin=ORIGIN, model_revision="dispatch-test")
        assert result
        assert all(event.model_revision.startswith("dispatch-test;processing:") for event in result)
        if method == "sn7_retrospective":
            assert all(event.quality["mode"] == "retrospective" for event in result)
    assert set(scored) == set(pipeline.METHODS) - set(pipeline.GATE_METHODS)


def test_full_calibration_namespaces_and_frozen_raw_policy(tmp_path):
    x, usable, births, info = fixture_inputs()
    weights = gate_weights(tmp_path / "gate.pt")
    calibration = {
        "occupancy": {"pairwise": {"scale": 1., "bias": -20.}},
        "decoder": {"threshold": 0.99},
        "event_gates": {"observation_gate": {"scale": 1., "bias": 0., "threshold": 0.99}},
    }
    # Occupancy calibration still applies to old methods.
    assert pipeline.method_events(x, usable, births, info, "pairwise", calibration=calibration) == []
    # The old global occupancy decoder cannot silently change the gate proposal policy.
    events = pipeline.method_events(x, usable, births, info, "observation_gate",
                                    event_gate_weights=weights, calibration=calibration)
    assert events and not any(event.state == "confirmed_by_model" for event in events)
    assert events[0].quality["event_gate_parameters"]["threshold"] == 0.99
    assert events[0].quality["raw_proposal_decoder"]["threshold"] == 0.55
    with pytest.raises(ValueError, match="shared raw proposal"):
        pipeline.method_events(x, usable, births, info, "proposal_tracking",
                               decoder=DecoderConfig(threshold=0.95))


def test_processing_revision_binds_calibration_and_decoder_even_with_custom_label():
    x, usable, births, info = fixture_inputs()
    first = {"occupancy": {"pairwise": {"scale": 1., "bias": 0.}}, "version": "unit"}
    reordered = {"version": "unit", "occupancy": {"pairwise": {"bias": 0., "scale": 1.}}}
    common = dict(method="pairwise", model_revision="custom-label", record_origin=ORIGIN)
    a = pipeline.method_events(x, usable, births, info, calibration=first, **common)
    b = pipeline.method_events(x, usable, births, info, calibration=reordered, **common)
    changed = pipeline.method_events(x, usable, births, info,
                                     calibration={"occupancy": {"pairwise": {"scale": 2., "bias": 0.}}},
                                     **common)
    padded = pipeline.method_events(x, usable, births, info, calibration=first,
                                    decoder=DecoderConfig(timing_padding_days=10), **common)
    assert a[0].model_revision == b[0].model_revision
    assert a[0].model_revision != changed[0].model_revision != padded[0].model_revision
    assert a[0].model_revision.startswith("custom-label;processing:")


def test_gate_only_reconstruct_exports_revisions_and_closes_store_without_truth(tmp_path, monkeypatch):
    x, usable, births, info = fixture_inputs()
    source = tmp_path / "features"
    source.mkdir()
    (source / "sites.json").write_text(json.dumps(info))
    np.savez(source / "features.npz", x=x, usable=usable, births=births)
    weights = gate_weights(tmp_path / "gate.pt")
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps({"event_gates": {"observation_gate": {"threshold": 0.7}}}))
    from terradelta import evaluation

    def forbidden_truth(*args, **kwargs):
        raise AssertionError("truth opened during reconstruction")

    monkeypatch.setattr(evaluation, "ground_truth_events", forbidden_truth)
    original_load = np.load
    opened = []

    def guarded_load(path, *args, **kwargs):
        assert str(path).endswith("features.npz"), "inference attempted a label/cache read"
        opened.append(str(path))
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", guarded_load)
    destination = tmp_path / "history"
    report = pipeline.reconstruct(source, output=destination, event_gate_weights=weights,
                                  calibration_path=calibration)
    assert report["method"] == "observation_gate" and len(opened) == 1
    assert report["event_gate_parameters"]["threshold"] == 0.7
    revisions = [json.loads(line) for line in (destination / "revisions.jsonl").read_text().splitlines()]
    assert any(event["state"] == "confirmed_by_model" for event in revisions)
    assert report["model_revision"] == revisions[0]["model_revision"]
    with duckdb.connect(str(destination / "events.duckdb"), read_only=True) as ledger:
        assert ledger.execute("SELECT count(*) FROM event_revisions").fetchone()[0] == len(revisions)
    independent = gpd.read_parquet(destination / "events.parquet")
    assert independent.crs.to_epsg() == 4326
    assert len(independent) == len({event["event_id"] for event in revisions})
    assert int((independent["state"] != "retracted").sum()) == report["events"]
    before = (destination / "events.geojson").read_bytes()
    with pytest.raises(ValueError, match="preserved"):
        pipeline.reconstruct(source, output=destination, event_gate_weights=weights)
    assert (destination / "events.geojson").read_bytes() == before


def test_cli_defaults_to_gate_and_missing_weights_fail_before_compute(tmp_path, monkeypatch):
    from terradelta import features

    called = []
    monkeypatch.setattr(pipeline, "predict_sequence", lambda *args, **kwargs: called.append("predict"))
    monkeypatch.setattr(features, "prepare_features", lambda *args, **kwargs: called.append("features"))

    def reconstructed(*args, **kwargs):
        called.append(kwargs)
        return {"method": kwargs["method"]}

    monkeypatch.setattr(pipeline, "reconstruct", reconstructed)
    args = ["run", str(tmp_path / "cache"), "--spatial-weights", str(tmp_path / "spatial.pt"),
            "--calibration", str(tmp_path / "calibration.json"), "--output", str(tmp_path / "new")]
    runner = CliRunner()
    result = runner.invoke(app, args + ["--event-gate-weights", str(tmp_path / "gate.pt")])
    assert result.exit_code == 0, result.output
    assert called[-1]["method"] == "observation_gate"
    assert called[-1]["event_gate_weights"] == tmp_path / "gate.pt"
    called.clear()
    result = runner.invoke(app, [*args[:-1], str(tmp_path / "other")])
    assert result.exit_code == 2 and "--temporal-weights" in result.output
    assert not called

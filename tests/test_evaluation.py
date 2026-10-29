import calendar
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import numpy as np
import pytest
from rasterio.transform import Affine
from shapely.geometry import box, mapping

from terradelta.evaluation import (
    evaluate_events,
    ground_truth_events,
    match_events,
    region_area_time,
    summarize_regions,
)
from terradelta.schema import EventRevision, ObservationInterval, SourceIdentity

UTC = timezone.utc
REC = datetime(2026, 10, 29, tzinfo=UTC)


def metadata(valid=None):
    if valid is None:
        valid = np.ones((6, 8, 8), dtype=bool)
    frames = []
    for index in range(len(valid)):
        month = index + 1
        start = datetime(2020, month, 1, tzinfo=UTC)
        end = datetime(2020, month, calendar.monthrange(2020, month)[1], 23, 59, 59, tzinfo=UTC)
        frames.append(
            {
                "source_id": f"frame-{month}",
                "acquisition_start": start.isoformat(),
                "acquisition_end": end.isoformat(),
                "valid_fraction": float(valid[index].mean()),
                "registration": {"usable": index != 2},
            }
        )
    return {
        "region_id": "synthetic-region",
        "crs": "EPSG:4326",
        "width": 8,
        "height": 8,
        "transform": list(Affine(0.001, 0, -80, 0, -0.001, 38))[:6],
        "frames": frames,
    }


def truth(geometry=None, **changes):
    result = {
        "object_id": 1,
        "geometry": geometry or mapping(box(-80, 37, -79.999, 37.001)),
        "change_class": "building_addition",
        "lower": "2020-02-01T00:00:00Z",
        "upper": "2020-03-31T23:59:59Z",
        "first_support": "2020-03-31T23:59:59Z",
        "confirmed_at": "2020-04-30T23:59:59Z",
    }
    result.update(changes)
    return result


def prediction(target=None, **changes):
    target = target or truth()
    source = SourceIdentity(
        source_id="frame-1",
        uri="https://example.org/source.tif",
        sha256="a" * 64,
        acquisition_start="2020-01-01T00:00:00Z",
        acquisition_end="2020-01-31T23:59:59Z",
    )
    value = {
        "event_id": "pred-1",
        "revision": 1,
        "geometry": target["geometry"],
        "change_class": target["change_class"],
        "state": "confirmed_by_model",
        "observation_interval": ObservationInterval(lower=target["lower"], upper=target["upper"]),
        "detected_at": target["confirmed_at"],
        "record_time": REC,
        "sources": [source],
        "model_revision": "unit-test",
        "region_id": "synthetic-region",
        "quality": {"unexpanded_observation_interval": [target["lower"], target["upper"]]},
    }
    value.update(changes)
    return EventRevision(**value)


def test_ground_truth_cloud_gaps_initial_inventory_and_transient_labels(tmp_path):
    labels = np.zeros((6, 8, 8), dtype="int32")
    valid = np.ones_like(labels, dtype=bool)
    # Source ID0 (raster ID1): initial inventory, then persistent removal.
    labels[:2, 1:3, 1:3] = 1
    # Source ID1: absent January, cloud February, present March onward.
    labels[2:, 4:6, 4:6] = 2
    valid[1, 4:6, 4:6] = False
    # Source ID2: a one-observation label flicker cannot establish an event.
    labels[2, 1:3, 5:7] = 3
    # Source ID3: two supports separated by a cloud; cloud is not absence.
    labels[2, 4:6, 1:3] = 4
    labels[4:, 4:6, 1:3] = 4
    valid[3, 4:6, 1:3] = False
    np.save(tmp_path / "instances.npy", labels)
    np.save(tmp_path / "valid.npy", valid)
    meta = metadata(valid)
    (tmp_path / "metadata.json").write_text(json.dumps(meta))
    result = ground_truth_events(tmp_path)
    assert len(result) == 3
    by_id = {item["object_id"]: item for item in result}
    assert by_id[0]["change_class"] == "building_removal"
    assert by_id[0]["lower"] == meta["frames"][1]["acquisition_start"]
    assert by_id[1]["lower"] == meta["frames"][0]["acquisition_start"]
    assert by_id[1]["upper"] == meta["frames"][2]["acquisition_end"]
    assert by_id[1]["confirmed_at"] == meta["frames"][3]["acquisition_end"]
    assert by_id[3]["confirmed_at"] == meta["frames"][4]["acquisition_end"]
    assert 2 not in by_id
    # Registration rejection is a model abstention, not deletion from truth.
    assert meta["frames"][2]["registration"]["usable"] is False
    assert by_id[1]["first_support_index"] == 2


def test_matching_is_one_to_one_and_indices_reference_original_revisions():
    target = truth()
    events = [
        prediction(state="proposed"),
        prediction(revision=2),
        prediction(event_id="duplicate", record_time=REC + timedelta(seconds=1)),
    ]
    matches = match_events(events, [target])
    assert len(matches) == 1
    assert matches[0][0] in {1, 2}
    assert matches[0][1] == 0
    report = evaluate_events(events, [target], metadata())
    assert report["event"]["precision"] == 0.5
    assert report["event"]["recall"] == 1
    assert report["duplicate_confirmed_events"] == 1


def test_sparse_assignment_maximizes_cardinality_before_iou():
    first = truth(mapping(box(0, 0, 0.001, 0.001)))
    second = truth(mapping(box(0.001, 0, 0.002, 0.001)), object_id=2)
    broad = prediction(first, geometry=mapping(box(-0.0002, 0, 0.0022, 0.001)))
    exact = prediction(first, event_id="exact-first")
    matches = match_events([broad, exact], [first, second])
    assert len(matches) == 2
    assert {(i, j) for i, j, _ in matches} == {(0, 1), (1, 0)}


def test_padding_cannot_game_matching():
    target = truth()
    event = prediction(
        target,
        observation_interval=ObservationInterval(
            lower="2019-01-01T00:00:00Z", upper="2020-11-30T23:59:59Z"
        ),
        detected_at="2020-12-31T23:59:59Z",
        quality={
            "unexpanded_observation_interval": ["2020-07-01T00:00:00Z", "2020-08-31T23:59:59Z"]
        },
    )
    assert match_events([event], [target]) == []


def test_late_detection_and_retraction_are_not_hidden():
    target = truth()
    late_time = datetime.fromisoformat(target["first_support"].replace("Z", "+00:00")) + timedelta(
        days=100
    )
    confirmed = prediction(target, revision=2, detected_at=late_time)
    proposed = prediction(target, state="proposed")
    retracted = prediction(
        target,
        revision=3,
        state="retracted",
        detected_at=late_time,
        record_time=REC + timedelta(seconds=1),
    )
    report = evaluate_events([proposed, confirmed, retracted], [target], metadata())
    assert report["event"]["tp"] == 1
    assert report["within_62_days"]["tp"] == 0
    assert report["matched_late_over_62_days"] == 1
    assert report["confirmed_then_retracted_event_ids"] == 1
    assert report["final_retained"]["predictions"] == 0
    assert report["final_retained"]["recall"] == 0
    assert report["detection_delay_from_label_first_support_end_days"]["mean"] == 100


def test_interval_containment_is_not_physical_date_coverage():
    target = truth()
    shorter = prediction(
        target,
        observation_interval=ObservationInterval(
            lower="2020-03-01T00:00:00Z", upper=target["upper"]
        ),
        quality={"unexpanded_observation_interval": ["2020-03-01T00:00:00Z", target["upper"]]},
    )
    report = evaluate_events([shorter], [target], metadata())
    assert report["timing"]["raw_interval_contains_label_interval"] == 0
    assert report["timing"]["raw_interval_overlaps_label_interval"] == 1
    assert report["timing"]["raw_midpoint_absolute_error_days_interval_proxy"]["mean"] > 0
    assert report["timing"]["is_physical_date_coverage"] is False


def test_geodesic_exposure_and_empty_denominators():
    meta = metadata()
    exposure = region_area_time(meta)
    assert 0.60 < exposure["geodesic_area_km2"] < 0.65
    assert exposure["source_valid_area_km2_years"] > exposure["model_usable_area_km2_years"]
    assert exposure["registration_rejected_frames"] == 1
    report = evaluate_events([], [], meta)
    assert report["event"]["precision"] is None
    assert report["event"]["recall"] is None
    assert report["event"]["f1"] is None
    assert report["false_confirmed_alerts_per_source_valid_km2_year"] == 0


def test_region_bootstrap_uses_regions_and_preserves_small_sample_limit():
    target = truth()
    report = evaluate_events([prediction(), prediction(event_id="duplicate")], [target], metadata())
    result = summarize_regions(
        {"region-a": report, "region-b": report, "region-c": report}, bootstrap_samples=50
    )
    assert result["event"]["precision"] == 0.5
    assert result["event"]["recall"] == 1
    assert result["uncertainty"]["regions"] == 3
    assert "unstable" in result["uncertainty"]["limit"]
    interval = result["region_bootstrap_95_percentile_intervals"]["event.precision"]
    assert interval["lower"] == interval["upper"] == 0.5


def test_candidate_coverage_is_separate_from_conditional_head_quality():
    target = truth()
    meta = metadata()
    meta["candidate_geometries"] = {
        "1": target["geometry"],
        "2": mapping(box(-81, 38, -80.999, 38.001)),
    }
    report = evaluate_events([], [target], meta)
    assert report["event"]["recall"] == 0
    assert report["candidate_sites"]["coverage"] == 1
    assert report["candidate_sites"]["sites"] == 2


def test_confirmation_cli_requires_freeze_before_reading_any_data(tmp_path):
    import importlib.util

    script = Path(__file__).parents[1] / "scripts/evaluate.py"
    spec = importlib.util.spec_from_file_location("evaluate_cli", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from test_protocol import boundary

    with pytest.raises(ValueError, match="full freeze"):
        module.validate_experiment(None, **boundary(tmp_path))


def test_confirmation_freeze_rejects_changed_artifact_bytes(tmp_path):
    import importlib.util
    from test_protocol import boundary
    from terradelta.protocol import freeze_experiment

    script = Path(__file__).parents[1] / "scripts/evaluate.py"
    spec = importlib.util.spec_from_file_location("evaluate_cli_hashes", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    kwargs = boundary(tmp_path)
    freeze_path = tmp_path / "freeze.json"
    manifest = freeze_experiment(freeze_path, **kwargs)
    assert module.validate_experiment(freeze_path, **kwargs) == manifest
    kwargs["calibration"].write_text("changed content after freeze")
    with pytest.raises(ValueError, match="calibration_sha256"):
        module.validate_experiment(freeze_path, **kwargs)

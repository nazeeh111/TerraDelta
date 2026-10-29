from datetime import datetime, timedelta, timezone
import json

import pytest
from pydantic import ValidationError
from shapely.geometry import Polygon, box, mapping

from terradelta.schema import EventRevision, ObservationInterval, SourceIdentity, geodesic_area_m2
from terradelta.store import EventStore

UTC = timezone.utc
JAN = datetime(2020, 1, 1, tzinfo=UTC)
FEB = datetime(2020, 2, 1, tzinfo=UTC)
MAR = datetime(2020, 3, 1, tzinfo=UTC)
APR = datetime(2020, 4, 1, tzinfo=UTC)
REC = datetime(2026, 10, 29, tzinfo=UTC)


def source(**changes):
    value = dict(
        source_id="mosaic-2020-02",
        uri="https://example.org/source.tif",
        sha256="a" * 64,
        acquisition_start=FEB,
        acquisition_end=MAR,
        availability_assumption="retrospective_month_end",
        license="CC-BY-SA-4.0",
    )
    value.update(changes)
    return SourceIdentity(**value)


def event(**changes):
    value = dict(
        event_id="event-a",
        revision=1,
        geometry=mapping(box(-80, 37, -79.999, 37.001)),
        change_class="building_addition",
        observation_interval=ObservationInterval(lower=JAN, upper=MAR),
        detected_at=MAR,
        record_time=REC,
        sources=[source()],
        quality={"registration_shift_px": 0.1},
        model_revision="test-sha",
        state="proposed",
        region_id="region-a",
    )
    value.update(changes)
    return EventRevision(**value)


def test_append_idempotence_and_conflicting_revision(tmp_path):
    with EventStore(tmp_path / "events.duckdb") as store:
        assert store.append(event())
        assert not store.append(event())
        with pytest.raises(ValueError, match="different content"):
            store.append(event(confidence=0.9))
        with pytest.raises(ValueError, match="expected revision 2"):
            store.append(event(revision=3, record_time=REC + timedelta(seconds=1)))
        assert store.count_revisions() == 1


def test_stable_history_and_retraction(tmp_path):
    path = tmp_path / "events.duckdb"
    with EventStore(path) as store:
        store.append(event())
        assert store.snapshot_time() == REC
        assert store.snapshot_time(REC + timedelta(days=30)) == REC
        assert store.snapshot_time(REC - timedelta(seconds=1)) is None
        historical = [x.model_dump(mode="json") for x in store.query(as_of=REC)]
        store.append(
            event(
                revision=2,
                state="confirmed_by_model",
                detected_at=APR,
                record_time=REC + timedelta(seconds=1),
            )
        )
        store.append(
            event(
                revision=3,
                state="retracted",
                detected_at=APR,
                reason="new observation contradicts earlier evidence",
                record_time=REC + timedelta(seconds=2),
            )
        )
        assert store.query() == []
        assert store.query(include_retracted=True)[0].revision == 3
        assert store.query(as_of=REC + timedelta(seconds=1))[0].state == "confirmed_by_model"
        assert [x.model_dump(mode="json") for x in store.query(as_of=REC)] == historical
        assert [x.revision for x in store.history("event-a")] == [1, 2, 3]
        assert store.snapshot_time() == REC + timedelta(seconds=2)
        assert store.snapshot_time(REC + timedelta(seconds=1)) == REC + timedelta(seconds=1)
    with EventStore(path) as reopened:
        assert reopened.count_revisions() == 3
        assert reopened.query(as_of=REC)[0].revision == 1


def test_no_backdated_new_event_or_same_time_insert():
    with EventStore() as store:
        store.append(event())
        for timestamp in [REC - timedelta(seconds=1), REC]:
            with pytest.raises(ValueError, match="later than"):
                store.append(event(event_id="event-b", record_time=timestamp))
        assert store.count_revisions() == 1


def test_revision_identity_and_detection_are_monotone():
    with EventStore() as store:
        store.append(event(detected_at=APR))
        with pytest.raises(ValueError, match="identity"):
            store.append(
                event(
                    revision=2,
                    change_class="building_removal",
                    detected_at=APR,
                    record_time=REC + timedelta(seconds=1),
                )
            )
        with pytest.raises(ValueError, match="backwards"):
            store.append(event(revision=2, record_time=REC + timedelta(seconds=1)))
        assert store.count_revisions() == 1


def test_query_spatial_geometry_hole_and_time_boundaries():
    donut = Polygon(
        [(-80, 37), (-79, 37), (-79, 38), (-80, 38), (-80, 37)],
        holes=[[(-79.8, 37.2), (-79.2, 37.2), (-79.2, 37.8), (-79.8, 37.8), (-79.8, 37.2)]],
    )
    with EventStore() as store:
        store.append(event(geometry=mapping(donut)))
        assert store.query(bbox=[-79.7, 37.3, -79.3, 37.7]) == []
        assert len(store.query(bbox=[-79.99, 37.01, -79.9, 37.1])) == 1
        assert len(store.query(start=MAR, end=MAR)) == 1
        assert store.query(start=APR) == []
        assert store.query(end=JAN - timedelta(seconds=1)) == []
        assert store.query(states=["reviewed"]) == []
        with pytest.raises(ValueError, match="end"):
            store.query(start=APR, end=JAN)
        with pytest.raises(ValueError, match="bbox"):
            store.query(bbox=[170, -10, -170, 10])
        with pytest.raises(ValueError, match="state"):
            store.query(states=["approved"])


@pytest.mark.parametrize(
    "changes",
    [
        {"geometry": mapping(box(500, 37, 501, 38))},
        {"geometry": mapping(Polygon([(0, 0), (1, 1), (1, 0), (0, 1), (0, 0)]))},
        {"record_time": datetime(2026, 10, 29)},
        {"detected_at": FEB},
        {"confidence": float("nan")},
        {"sources": []},
        {"unknown": "not allowed"},
    ],
)
def test_schema_rejects_invalid_claims(changes):
    with pytest.raises((ValueError, ValidationError)):
        event(**changes)


def test_source_availability_and_sanitized_identity():
    assert source().available_at == MAR
    assert source(published_at=APR, availability_assumption="known_publication").available_at == APR
    with pytest.raises(ValueError, match="unavailable"):
        event(sources=[source(published_at=APR)])
    with pytest.raises(ValueError, match="publication timestamp"):
        source(availability_assumption="known_publication")
    with pytest.raises(ValueError, match="credentials"):
        source(uri="https://example.org/tif?token=must-not-be-saved")
    with pytest.raises(ValueError, match="positive duration"):
        ObservationInterval(lower=MAR, upper=MAR)


def test_area_uses_ellipsoid():
    area_equator = geodesic_area_m2(mapping(box(0, 0, 0.001, 0.001)))
    area_high_latitude = geodesic_area_m2(mapping(box(0, 60, 0.001, 60.001)))
    assert 12_000 < area_equator < 12_500
    assert 0.49 < area_high_latitude / area_equator < 0.51


def test_revalidates_mutated_nested_model():
    candidate = event()
    candidate.geometry["coordinates"] = []
    with EventStore() as store, pytest.raises(ValueError):
        store.append(candidate)


def test_independent_geojson_and_geoparquet_readback(tmp_path):
    import geopandas as gpd
    import pyarrow.parquet as pq
    import pyogrio

    with EventStore() as store:
        store.append(event())
        json_path = store.export_geojson(tmp_path / "events.geojson", as_of=REC)
        parquet_path = store.export_parquet(tmp_path / "events.parquet", as_of=REC)
        raw = json.loads(json_path.read_text())
        assert "crs" not in raw
        assert raw["terradelta"]["snapshot_time"] == REC.isoformat()
        assert raw["features"][0]["properties"]["area_m2"] > 0
        # GDAL/OGR reads the exported file independently of the writer's geometry path.
        read_json = pyogrio.read_dataframe(json_path)
        read_parquet = gpd.read_parquet(parquet_path)
        assert len(read_json) == len(read_parquet) == 1
        assert read_parquet.crs.to_epsg() == 4326
        assert read_json.geometry[0].equals(read_parquet.geometry[0])
        recovered = EventRevision.model_validate_json(read_parquet.event_json[0])
        assert recovered.sources[0].sha256 == "a" * 64
        assert recovered == event()
        assert b"terradelta" in pq.read_metadata(parquet_path).metadata
        parquet_metadata = json.loads(pq.read_metadata(parquet_path).metadata[b"terradelta"])
        assert parquet_metadata["snapshot_time"] == REC.isoformat()
        empty = store.export_parquet(tmp_path / "empty.parquet", states=["reviewed"])
        assert len(gpd.read_parquet(empty)) == 0

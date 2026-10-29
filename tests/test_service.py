import json

from fastapi.testclient import TestClient
import geopandas as gpd
import numpy as np

from terradelta.schema import EventRevision
from terradelta.service import create_app


def prepare(tmp_path):
    cache = tmp_path / "cache" / "area"
    cache.mkdir(parents=True)
    np.save(cache / "rgb.npy", np.ones((1, 3, 8, 8), dtype="uint8") * 80)
    np.save(cache / "valid.npy", np.ones((1, 8, 8), bool))
    source = {
        "source_id": "month1",
        "uri": "https://example.org/image.tif",
        "sha256": "a" * 64,
        "acquisition_start": "2020-01-01T00:00:00+00:00",
        "acquisition_end": "2020-01-31T23:59:59+00:00",
        "published_at": None,
        "availability_assumption": "retrospective_month_end",
        "valid_fraction": 1.0,
        "registration": {"usable": True, "shift_pixels": [0, 0]},
        "local_image": "/private/not-public/source.tif",
    }
    metadata = {
        "region_id": "area",
        "width": 8,
        "height": 8,
        "crs": "EPSG:3857",
        "bbox": [0, 0, 0.01, 0.01],
        "bounds_native": [0, 0, 1000, 1000],
        "frames": [source],
    }
    (cache / "metadata.json").write_text(json.dumps(metadata))
    app = create_app(tmp_path / "cache", tmp_path / "run")
    return app, source


def event(source, revision=1, state="confirmed_by_model", record="2026-01-01T00:00:00+00:00"):
    src = {
        k: v
        for k, v in source.items()
        if k not in ["valid_fraction", "registration", "local_image"]
    }
    return EventRevision(
        event_id="test-event",
        revision=revision,
        state=state,
        geometry={
            "type": "Polygon",
            "coordinates": [[[0, 0], [0.001, 0], [0.001, 0.001], [0, 0.001], [0, 0]]],
        },
        change_class="building_addition",
        observation_interval={
            "lower": "2020-01-01T00:00:00+00:00",
            "upper": "2020-01-31T23:59:59+00:00",
        },
        detected_at="2020-01-31T23:59:59+00:00",
        record_time=record,
        sources=[src],
        quality={},
        model_revision="test",
        region_id="area",
    )


def test_api_query_export_preserves_revision_and_snapshot(tmp_path):
    app, source = prepare(tmp_path)
    client = TestClient(app)
    first = event(source)
    app.state.store.append(first)
    answer = client.get("/api/events", params={"region_id": "area"}).json()
    snapshot = answer["snapshot_time"]
    second = event(source, 2, "reviewed", "2026-01-02T00:00:00+00:00")
    app.state.store.append(second)
    assert client.get("/api/events", params={"as_of": snapshot}).json() == answer
    result = client.get("/api/export.parquet", params={"as_of": snapshot, "region_id": "area"})
    path = tmp_path / "events.parquet"
    path.write_bytes(result.content)
    frame = gpd.read_parquet(path)
    assert len(frame) == 1 and frame.iloc[0]["revision"] == 1
    assert frame.iloc[0]["state"] == "confirmed_by_model"
    assert frame.crs.to_epsg() == 4326
    current = client.get("/api/export.geojson").json()["features"][0]["properties"]
    assert current["revision"] == 2 and current["state"] == "reviewed"
    app.state.store.close()


def test_api_real_source_empty_snapshot_and_errors(tmp_path):
    app, source = prepare(tmp_path)
    client = TestClient(app)
    initial = client.get("/api/events").json()
    token = initial["snapshot_time"]
    assert token and not initial["features"]
    app.state.store.append(event(source))
    assert not client.get("/api/events", params={"as_of": token}).json()["features"]
    catalog = client.get("/api/catalog").json()
    assert "local_image" not in catalog["regions"][0]["frames"][0]
    assert client.get("/api/regions/area/frames/0.png").headers["content-type"] == "image/png"
    assert client.get("/api/regions/area/frames/-1.png").status_code == 404
    assert client.get("/api/events", params={"bbox": "nan,0,1,1"}).status_code == 422
    assert client.get("/api/events", params={"start": "not-a-date"}).status_code == 422
    assert client.get("/api/events", params={"region_id": "unknown"}).status_code == 404
    assert (
        client.post(
            "/api/events/test-event/review",
            json={"reason": "checked source"},
            headers={"Origin": "https://external.example"},
        ).status_code
        == 403
    )
    app.state.store.close()

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


def test_state_filter_keeps_latest_revision_and_exact_exports(tmp_path):
    app, source = prepare(tmp_path)
    client = TestClient(app)
    app.state.store.append(event(source))
    first = client.get("/api/events", params={"state": "confirmed_by_model"}).json()
    assert len(first["features"]) == 1
    app.state.store.append(event(source, 2, "reviewed", "2026-01-02T00:00:00+00:00"))
    app.state.store.append(event(source, 3, "retracted", "2026-01-03T00:00:00+00:00"))
    assert not client.get("/api/events", params={"state": "confirmed_by_model"}).json()["features"]
    assert (
        client.get(
            "/api/events", params={"state": "confirmed_by_model", "as_of": first["snapshot_time"]}
        ).json()
        == first
    )
    params = {"region_id": "area", "state": "retracted"}
    current = client.get("/api/events", params=params).json()
    assert current["features"][0]["properties"]["revision"] == 3
    params["as_of"] = current["snapshot_time"]
    exported = client.get("/api/export.geojson", params=params).json()
    assert exported["features"] == current["features"]
    assert exported["source_licenses"] == {"area": "unspecified"}
    assert "license" not in exported
    parquet = client.get("/api/export.parquet", params=params)
    assert parquet.status_code == 200
    path = tmp_path / "filtered.parquet"
    path.write_bytes(parquet.content)
    frame = gpd.read_parquet(path)
    assert len(frame) == 1 and frame.iloc[0]["state"] == "retracted"
    assert frame.iloc[0]["revision"] == 3
    assert client.get("/api/events", params={"state": "invented"}).status_code == 422
    app.state.store.close()


def test_native_source_detail_uses_exact_revision_and_pixel_bytes(tmp_path):
    from io import BytesIO
    from PIL import Image

    app, source = prepare(tmp_path)
    app.state.store.close()
    cache = tmp_path / "cache/area"
    y, x = np.indices((128, 128), dtype="uint8")
    rgb = np.stack([x, y, np.full_like(x, 80)])[None]
    valid = np.ones((1, 128, 128), bool)
    valid[:, 80:90, 10:20] = False
    np.save(cache / "rgb.npy", rgb)
    np.save(cache / "valid.npy", valid)
    metadata = json.loads((cache / "metadata.json").read_text())
    metadata.update(width=128, height=128, transform=[10, 0, 0, 0, -10, 1280])
    (cache / "metadata.json").write_text(json.dumps(metadata))
    app = create_app(tmp_path / "cache", tmp_path / "run")
    client = TestClient(app)
    app.state.store.append(event(source))
    params = {"detail_event": "test-event", "detail_revision": 1}
    response = client.get("/api/regions/area/frames/0.png", params=params)
    assert response.status_code == 200
    left, top, width, height = map(int, response.headers["X-TerraDelta-Pixel-Window"].split(","))
    assert width == height == 64
    result = np.asarray(Image.open(BytesIO(response.content)))
    expected = np.concatenate([rgb[0], valid[0][None].astype("uint8") * 255]).transpose(1, 2, 0)
    np.testing.assert_array_equal(result, expected[top : top + height, left : left + width])
    assert not result[..., 3].all()
    full = client.get("/api/regions/area/frames/0.png")
    assert Image.open(BytesIO(full.content)).size == (128, 128)
    assert response.headers["ETag"] != full.headers["ETag"]
    assert response.headers["X-TerraDelta-Source-SHA256"] == source["sha256"]
    assert (
        client.get(
            "/api/regions/area/frames/0.png", params={"detail_event": "test-event"}
        ).status_code
        == 422
    )
    assert (
        client.get(
            "/api/regions/area/frames/0.png", params={**params, "detail_revision": 0}
        ).status_code
        == 422
    )
    assert (
        client.get(
            "/api/regions/area/frames/0.png", params={**params, "detail_revision": 2}
        ).status_code
        == 404
    )
    # A source with the same name but different bytes cannot silently stand in.
    changed = dict(source, sha256="b" * 64)
    app.state.store.append(event(changed, 2, "reviewed", "2026-01-02T00:00:00+00:00"))
    assert (
        client.get(
            "/api/regions/area/frames/0.png", params={**params, "detail_revision": 2}
        ).status_code
        == 409
    )
    assert client.get("/api/regions/area/frames/0.png", params=params).status_code == 200
    app.state.store.close()

import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from scipy.ndimage import gaussian_filter, shift

from terradelta.data import ingest_region, sequence_registration
from terradelta.pipeline import prefix_frames
from terradelta.decoder import decode_events


def write_source(root, month, transform=None, cloud=False):
    directory = root / "images_masked"
    directory.mkdir(exist_ok=True, parents=True)
    name = f"global_monthly_2020_{month:02d}_mosaic_area"
    rgb = np.random.default_rng(0).integers(10, 240, (4, 32, 32), dtype="uint8")
    rgb[3] = 0 if cloud else 255
    with rasterio.open(
        directory / f"{name}.tif",
        "w",
        driver="GTiff",
        width=32,
        height=32,
        count=4,
        dtype="uint8",
        crs="EPSG:3857",
        transform=transform or from_origin(0, 200, 4, 4),
    ) as ds:
        ds.write(rgb)
    labels = root / "labels_match"
    labels.mkdir(exist_ok=True)
    (labels / f"{name}_Buildings.geojson").write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
                "features": [],
            }
        )
    )


def test_empty_labels_and_clouds_do_not_create_observed_background(tmp_path):
    root = tmp_path / "area"
    write_source(root, 1, cloud=True)
    cache = tmp_path / "cache"
    metadata = ingest_region(root, cache, labels=True)
    assert not np.load(cache / "valid.npy").any()
    assert not metadata["frames"][0]["registration"]["usable"]
    assert metadata["frames"][0]["empty_label_collection"]


def test_roundoff_is_allowed_but_a_pixel_grid_shift_is_rejected(tmp_path):
    root = tmp_path / "area"
    write_source(root, 1)
    write_source(root, 2, transform=from_origin(1e-9, 200 + 1e-9, 4, 4))
    assert len(ingest_region(root, tmp_path / "cache")["frames"]) == 2
    write_source(root, 3, transform=from_origin(4, 200, 4, 4))
    with pytest.raises(ValueError, match="Incompatible grids"):
        ingest_region(root, tmp_path / "different")


def test_cumulative_small_shifts_cannot_become_event_support():
    rng = np.random.default_rng(42)
    texture = gaussian_filter(rng.uniform(0, 255, (96, 96)), 0.7)
    rgb = np.stack(
        [np.repeat(shift(texture, (0, t), mode="reflect")[None], 3, axis=0) for t in range(8)]
    ).astype("uint8")
    valid = np.ones((8, 96, 96), bool)
    diagnostics = sequence_registration(rgb, valid)
    assert all(d["usable"] for d in diagnostics[:2])
    assert not any(d["usable"] for d in diagnostics[3:])
    # Perfect unchanged building drifts through frozen geometry; unsupported
    # late probabilities must never count as model-confirmed physical removal.
    from test_temporal_contract import info

    context = info(8)
    for frame, diag in zip(context["metadata"]["frames"], diagnostics):
        frame["registration"] = diag
    p = np.array([[1, 0.75, 0.5, 0.25, 0, 0, 0, 0]], dtype="float32")
    usable = np.array([[d["usable"] for d in diagnostics]])
    events = decode_events(p, usable, np.array([0]), context)
    assert not any(e.state == "confirmed_by_model" for e in events)


def test_midmonth_end_and_known_publication_lags_are_safe():
    frames = [
        {
            "acquisition_start": f"2020-{m:02d}-01T00:00:00+00:00",
            "acquisition_end": f"2020-{m:02d}-28T23:59:59+00:00",
            "published_at": None,
        }
        for m in range(1, 4)
    ]
    assert len(prefix_frames(frames, "2020-02-15")) == 1
    assert len(prefix_frames(frames, "2020-02-28")) == 2
    frames[1]["published_at"] = "2020-04-05T00:00:00+00:00"
    with pytest.raises(ValueError, match="Out-of-order publication"):
        prefix_frames(frames, "2020-03-31")
    with pytest.raises(ValueError, match="Out-of-order publication"):
        prefix_frames(frames, None)

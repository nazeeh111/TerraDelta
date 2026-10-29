import numpy as np
import pytest
import rasterio

from terradelta.pipeline import export_probability_raster


def test_dataset_and_band_masks_preserve_unknown_observations(tmp_path):
    valid = np.ones((8, 8), bool)
    valid[:4] = False
    p = np.ones((2, 8, 8), dtype="float32") * 0.7
    metadata = {
        "height": 8,
        "width": 8,
        "crs": "EPSG:3857",
        "transform": [4, 0, 100, 0, -4, 200],
        "license": "CC-BY-SA-4.0",
    }
    frame = {
        "source_id": "source",
        "sha256": "a" * 64,
        "uri": "https://example.org/source.tif",
        "acquisition_start": "2020-01-01T00:00:00Z",
        "acquisition_end": "2020-01-31T23:59:59Z",
    }
    path = export_probability_raster(
        tmp_path / "probability.tif", p, valid, metadata, frame, "b" * 64
    )
    with rasterio.open(path) as reader:
        np.testing.assert_array_equal(reader.dataset_mask() > 0, valid)
        np.testing.assert_array_equal(~reader.read(1, masked=True).mask, valid)
        assert reader.crs.to_epsg() == 3857 and reader.transform.a == 4
        assert reader.tags()["source_sha256"] == "a" * 64
        assert len(reader.files) == 1  # mask travels inside the GeoTIFF
    with pytest.raises(ValueError, match="exists"):
        export_probability_raster(path, p, valid, metadata, frame, "b" * 64)

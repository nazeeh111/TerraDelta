"""Exact component bookkeeping against the preserved pre-optimization algorithm."""

import numpy as np
import pytest
from scipy import ndimage
from skimage.segmentation import watershed

import terradelta.features as features


def legacy_instances(probability, valid, threshold=0.45, min_pixels=3):
    """The bd33082 whole-image marker assignment, retained as a small oracle."""
    foreground = (probability[0] >= threshold) & valid
    interior = probability[0] * (1 - probability[1])
    seeds = (interior >= max(0.18, threshold * 0.65)) & foreground
    markers, _ = ndimage.label(seeds)
    whole, n = ndimage.label(foreground)
    maxima = ndimage.maximum(markers, whole, np.arange(1, n + 1)) if n else []
    count = int(markers.max())
    for index, maximum in enumerate(maxima, 1):
        if maximum == 0:
            count += 1
            markers[whole == index] = count
    objects = watershed(-interior, markers, mask=foreground).astype("int32")
    areas = np.bincount(objects.ravel())
    reject = (areas < min_pixels) | (areas > 5000)
    reject[0] = False
    objects[reject[objects]] = 0
    return np.unique(objects, return_inverse=True)[1].reshape(objects.shape).astype("int32")


def component_case(name):
    probability = np.zeros((2, 96, 103), dtype="float32")
    probability[1] = 0.98  # Foreground components have no interior seeds by default.
    valid = np.ones(probability.shape[1:], dtype=bool)
    if name == "dense_unseeded":
        for row in range(0, 94, 3):
            for column in range(0, 100, 3):
                probability[0, row:row + 2, column:column + 2] = 0.8
    elif name == "nested_and_seeded":
        probability[0, 6:22, 6:22] = 0.8
        probability[0, 9:19, 9:19] = 0
        probability[0, 12:15, 12:15] = 0.8
        probability[1, 12:15, 12:15] = 0.1  # Seeded island inside an unseeded ring.
        probability[0, :4, :5] = 0.8
        probability[0, -4:, -5:] = 0.8
        probability[1, -4:, -5:] = 0.1
    elif name == "split_edges":
        probability[0, :14, :15] = 0.8
        valid[6, :15] = False
        valid[:3, 4:6] = False
        probability[0, 20:22, 20:22] = 0.8
        probability[0, 22:24, 22:24] = 0.8  # Diagonal contact remains separate.
        probability[1, 20:22, 20:22] = 0.1
    elif name == "size_rejection":
        probability[0, :80, :80] = 0.8  # Existing >5000-pixel rejection.
        probability[0, 85, 85] = 0.8
        probability[0, 90:92, 98:100] = 0.8
        probability[1, 90:92, 98:100] = 0.1
    elif name == "all_seeded":
        probability[0, 3:8, 4:9] = 0.8
        probability[1, 3:8, 4:9] = 0.1
    else:
        assert name == "empty"
    return probability, valid


@pytest.mark.parametrize(
    "name", ["empty", "dense_unseeded", "nested_and_seeded", "split_edges",
             "size_rejection", "all_seeded"]
)
@pytest.mark.parametrize("threshold,min_pixels", [(0.45, 3), (0.6, 4)])
def test_local_component_seeds_keep_exact_instance_ids(name, threshold, min_pixels):
    probability, valid = component_case(name)
    original_probability, original_valid = probability.copy(), valid.copy()
    expected = legacy_instances(probability, valid, threshold, min_pixels)
    actual = features.footprint_instances(probability, valid, threshold, min_pixels)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.int32
    np.testing.assert_array_equal(probability, original_probability)
    np.testing.assert_array_equal(valid, original_valid)


def test_local_component_seeds_accept_noncontiguous_input():
    probability, valid = component_case("nested_and_seeded")
    probability, valid = probability[:, ::-1, ::2], valid[::-1, ::2]
    assert not probability.flags.c_contiguous
    np.testing.assert_array_equal(features.footprint_instances(probability, valid),
                                  legacy_instances(probability, valid))


@pytest.mark.parametrize("boundary_probability", [0.0, 0.98])
def test_entirely_foreground_raster_keeps_its_first_component(boundary_probability):
    probability = np.ones((2, 1, 3), dtype="float32")
    probability[1] = boundary_probability
    valid = np.ones((1, 3), dtype=bool)

    actual = features.footprint_instances(probability, valid)

    np.testing.assert_array_equal(actual, np.ones((1, 3), dtype="int32"))


def test_entirely_foreground_raster_keeps_multiple_watershed_components():
    probability = np.ones((2, 2, 6), dtype="float32")
    probability[1] = 0
    probability[1, :, 2:4] = 1
    valid = np.ones((2, 6), dtype=bool)

    actual = features.footprint_instances(probability, valid)

    assert (actual > 0).all()
    np.testing.assert_array_equal(np.unique(actual), [1, 2])
    assert np.bincount(actual.ravel())[1:].min() >= 3


def test_entirely_foreground_initial_inventory_keeps_its_causal_site():
    probability = np.zeros((3, 2, 1, 3), dtype="float32")
    probability[0, 0] = 1
    valid = np.ones((3, 1, 3), dtype=bool)
    usable = np.ones(3, dtype=bool)

    sites, births = features.candidate_sites(probability, valid, usable)
    prefix_sites, prefix_births = features.candidate_sites(
        probability[:1], valid[:1], usable[:1]
    )

    np.testing.assert_array_equal(sites, np.ones((1, 3), dtype="int32"))
    np.testing.assert_array_equal(births, [0])
    np.testing.assert_array_equal(sites, prefix_sites)
    np.testing.assert_array_equal(births, prefix_births)


def test_candidate_births_features_and_future_prefix_remain_exact(monkeypatch):
    probability, validity = component_case("nested_and_seeded")
    probabilities = np.repeat(probability[None], 6, axis=0)
    probabilities[2:, 0, 25:28, 28:31] = 0.9
    probabilities[2:, 1, 25:28, 28:31] = 0.1
    probabilities[4:, 0, :3, 35:38] = 0.9
    probabilities[4:, 1, :3, 35:38] = 0.1
    valid = np.repeat(validity[None], 6, axis=0)
    valid[3, 5:8, 5:8] = False
    good_frames = np.array([True, True, False, True, True, True])
    rng = np.random.default_rng(2901)
    rgb = rng.integers(0, 256, size=(6, 3, *validity.shape), dtype="uint8")
    metadata = {"frames": [
        {"acquisition_end": f"2020-{month:02d}-28T23:59:59+00:00",
         "registration": {"usable": bool(good_frames[month - 1]),
                          "shift_magnitude_pixels": 0.25}}
        for month in range(1, 7)
    ]}
    actual_sites, actual_births = features.candidate_sites(probabilities, valid, good_frames)
    with monkeypatch.context() as context:
        context.setattr(features, "footprint_instances", legacy_instances)
        old_sites, old_births = features.candidate_sites(probabilities, valid, good_frames)
    np.testing.assert_array_equal(actual_sites, old_sites)
    np.testing.assert_array_equal(actual_births, old_births)
    x, usable = features.feature_cube(probabilities, rgb, valid, metadata, actual_sites)
    old_x, old_usable = features.feature_cube(probabilities, rgb, valid, metadata, old_sites)
    np.testing.assert_array_equal(x, old_x)
    np.testing.assert_array_equal(usable, old_usable)

    cutoff = 4
    sites, births = features.candidate_sites(probabilities[:cutoff], valid[:cutoff],
                                            good_frames[:cutoff])
    np.testing.assert_array_equal(np.where(actual_sites <= len(births), actual_sites, 0), sites)
    np.testing.assert_array_equal(actual_births[:len(births)], births)
    prefix_x, prefix_usable = features.feature_cube(
        probabilities[:cutoff], rgb[:cutoff], valid[:cutoff],
        {"frames": metadata["frames"][:cutoff]}, sites
    )
    np.testing.assert_array_equal(x[:len(births), :cutoff], prefix_x)
    np.testing.assert_array_equal(usable[:len(births), :cutoff], prefix_usable)

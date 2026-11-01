"""CPU/synthetic only: no study pixels, labels, downloads or accelerator use."""
import copy
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import terradelta.embeddings as embedding
from terradelta.data import sha256
from terradelta.features import FEATURE_NAMES
from terradelta.protocol import semantic_metadata_hash
from terradelta.decoder import DecoderConfig, decode_events


def test_actual_resnet_cpu_forward_is_frozen_and_pixel_dependent():
    torchvision = pytest.importorskip("torchvision")
    torch.set_num_threads(2)
    torch.manual_seed(4)
    model = embedding.FrozenResNet18(encoder=torchvision.models.resnet18(weights=None))
    before = {key: value.clone() for key, value in model.encoder.state_dict().items()}
    model.train()
    a = model(torch.rand(1, 3, 31, 43))
    b = model(torch.zeros(1, 3, 31, 43))
    assert a.shape == (1, 64, 8, 11)
    assert torch.isfinite(a).all() and torch.max(torch.abs(a-b)) > 1e-4
    assert not a.requires_grad and not model.encoder.training
    assert all(not p.requires_grad and p.grad is None for p in model.encoder.parameters())
    assert all(torch.equal(value, model.encoder.state_dict()[key]) for key, value in before.items())


def test_pooling_exact_native_centers_valid_pixels_and_future_sites():
    y, x = np.mgrid[:5, :9]
    sites = np.where(x < 4, 1, np.where(x < 8, 2, 3)).astype(np.int32)
    valid = np.ones_like(sites, bool)
    valid[0, 0] = False
    # A linear lattice field: bilinear samples must equal x/4+10*y/4.
    field = torch.tensor([[[[0., 1., 2.], [10., 11., 12.]]]])
    pooled, coverage = embedding.pool_sites(field, sites, valid, np.array([True, False, False]),
                                            chunk_pixels=3)
    selected = (sites == 1) & valid
    np.testing.assert_allclose(pooled[0, 0], (x/4+10*y/4)[selected].mean(), atol=1e-6)
    assert coverage[0] == np.float32(19/20)
    np.testing.assert_array_equal(pooled[1:], 0)
    old_sites = np.where(sites == 1, sites, 0)
    prefix, _ = embedding.pool_sites(field, old_sites, valid, np.array([True]), chunk_pixels=3)
    np.testing.assert_array_equal(prefix, pooled[:1])
    full_inventory, _ = embedding.pool_sites(field, sites, valid, np.ones(3, bool), chunk_pixels=3)
    np.testing.assert_array_equal(prefix, full_inventory[:1])
    empty, fraction = embedding.pool_sites(field, sites, np.zeros_like(valid),
                                           np.ones(3, bool))
    np.testing.assert_array_equal(empty, 0)
    np.testing.assert_array_equal(fraction, 0)


def test_causal_distance_and_state_are_prefix_exact_with_gaps_and_births():
    vectors = np.zeros((2, 7, 64), np.float32)
    vectors[:, :2, 0] = 1
    vectors[:, 2:5, 1] = 1
    vectors[:, 5:, 0] = 1
    usable = np.ones((2, 7), bool)
    usable[0, 3] = False
    births = np.array([0, 4], np.int32)
    probability = np.array([[.1, .1, .9, .9, .9, .1, .1]]*2, np.float32)
    score, available, distance = embedding.distance_gated_scores(
        probability, vectors, usable, births, .2)
    short = embedding.distance_gated_scores(probability[:1, :4], vectors[:1, :4],
                                            usable[:1, :4], births[:1], .2)
    for full, prefix in zip((score, available, distance), short):
        np.testing.assert_array_equal(full[:1, :4], prefix)
    changed = vectors.copy()
    changed[:, 4:] = 20
    later = embedding.distance_gated_scores(probability, changed, usable, births, .2)
    for full, altered in zip((score, available, distance), later):
        np.testing.assert_array_equal(full[:, :4], altered[:, :4])
    assert distance[0, 2] == 1 and not available[0, 3]
    # Archived observations remain available when this site is discovered later.
    assert available[1, :4].all()
    np.testing.assert_array_equal(score[1, :4], probability[1, :4])
    zeros = embedding.distance_gated_scores(probability, np.zeros_like(vectors), usable, births, 0)
    assert not zeros[1].any()
    np.testing.assert_array_equal(zeros[0], .5)


def test_zero_distance_threshold_reduces_to_current_spatial_probability_on_usable_frames():
    rng = np.random.default_rng(19)
    vectors = rng.uniform(.1, 1, (3, 5, 64)).astype(np.float32)
    probability = rng.uniform(0, 1, (3, 5)).astype(np.float32)
    score, available, _ = embedding.distance_gated_scores(
        probability, vectors, np.ones((3, 5), bool), np.zeros(3, np.int32), 0)
    assert available.all()
    np.testing.assert_array_equal(score, probability)


def test_undefined_median_reference_abstains_but_new_usable_images_can_recover():
    vectors = np.zeros((1, 6, 64), np.float32)
    vectors[0, 0, 0] = vectors[0, 1, 1] = vectors[0, 2, 2] = 1
    vectors[0, 3:, 0] = 1
    distance, available = embedding.causal_distance(vectors, np.ones((1, 6), bool),
                                                    np.array([0], np.int32))
    assert not available[0, 3] and not available[0, 4] and available[0, 5]
    assert np.isfinite(distance).all() and distance[0, 5] == 0


def sidecar_fixture(tmp_path):
    directory, feature_dir, cache_dir = [tmp_path / name for name in ("sidecar", "features", "cache")]
    for path in (directory, feature_dir, cache_dir):
        path.mkdir()
    meta = {"region_id": "synthetic", "frames": [{"source_id": "a"}, {"source_id": "b"}]}
    (cache_dir / "metadata.json").write_text(json.dumps(meta))
    np.save(cache_dir / "rgb.npy", np.zeros((2, 3, 4, 4), np.uint8))
    np.save(cache_dir / "valid.npy", np.ones((2, 4, 4), bool))
    np.save(feature_dir / "sites.npy", np.ones((4, 4), np.int32))
    births, usable = np.array([0], np.int32), np.ones((1, 2), bool)
    np.savez(feature_dir / "features.npz", x=np.zeros((1, 2, 17), np.float32),
             usable=usable, births=births)
    binding = {"spatial_weights_sha256": "a"*64, "spatial_backbone_sha256": None,
               "candidate_threshold": .55, "spatial_inference": {
                   "tile_size": 128, "context_halo_native_pixels": 64,
                   "native_alignment": 8, "internal_input_scale": 2,
                   "implementation_sha256": {"spatial.py": "c"*64}}}
    info = {"region_id": "synthetic", "feature_protocol": "calendar_quality_v2",
            "feature_names": FEATURE_NAMES, "metadata_semantic_sha256":
            semantic_metadata_hash(cache_dir / "metadata.json"), **binding}
    (feature_dir / "sites.json").write_text(json.dumps(info))
    arrays = {"vectors": np.ones((1, 2, 64), np.float32), "usable": usable,
              "coverage": np.ones((1, 2), np.float32), "births": births}
    np.savez(directory / "embeddings.npz", **arrays)
    paths = {"rgb.npy": cache_dir / "rgb.npy", "valid.npy": cache_dir / "valid.npy",
             "features.npz": feature_dir / "features.npz", "sites.npy": feature_dir / "sites.npy",
             "sites.json": feature_dir / "sites.json"}
    manifest = {"format": embedding.FORMAT, "complete": True, "region_id": "synthetic",
                "descriptor": embedding.descriptor_plan(), "config_sha256": "d"*64,
                "spatial_provenance": binding, "input_sha256": {k: sha256(v) for k, v in paths.items()},
                "metadata_semantic_sha256": info["metadata_semantic_sha256"],
                "implementation_sha256": {"src/terradelta/embeddings.py": sha256(Path(embedding.__file__)),
                    "scripts/extract_embeddings.py": sha256(Path(__file__).parents[1] / "scripts/extract_embeddings.py")},
                "source_frames": [{k: frame.get(k) for k in (
                    "source_id", "sha256", "uri", "acquisition_start", "acquisition_end",
                    "published_at", "availability_assumption")} for frame in meta["frames"]],
                "embeddings_npz_sha256": sha256(directory / "embeddings.npz"),
                "array_sha256": {k: embedding.array_sha256(v) for k, v in arrays.items()}}
    (directory / "embedding.json").write_text(json.dumps(manifest))
    return directory, feature_dir, cache_dir


def test_image_only_loader_rejects_source_site_and_array_edits(tmp_path):
    directory, feature, cache = sidecar_fixture(tmp_path)
    result = embedding.load_embedding_data(directory, feature_dir=feature, cache_dir=cache,
                                           expected_config_sha256="d"*64)
    assert result["vectors"].shape == (1, 2, 64) and result["usable"].all()
    sites = feature / "sites.json"
    original = sites.read_bytes()
    sites.write_bytes(original+b" ")
    with pytest.raises(ValueError, match="input pixels/masks/sites/features"):
        embedding.load_embedding_data(directory, feature_dir=feature, cache_dir=cache)
    sites.write_bytes(original)
    with pytest.raises(ValueError, match="config"):
        embedding.load_embedding_data(directory, feature_dir=feature, cache_dir=cache,
                                       expected_config_sha256="e"*64)
    with (directory / "embeddings.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="array file changed"):
        embedding.load_embedding_data(directory, feature_dir=feature, cache_dir=cache)


@pytest.mark.parametrize("damage", ["missing_extractor", "wrong_extractor", "frame_source"])
def test_loader_rejects_missing_or_stale_extractor_and_frame_identity(tmp_path, damage):
    directory, feature, cache = sidecar_fixture(tmp_path)
    path = directory / "embedding.json"
    value = json.loads(path.read_text())
    if damage == "missing_extractor":
        del value["implementation_sha256"]["scripts/extract_embeddings.py"]
    elif damage == "wrong_extractor":
        value["implementation_sha256"]["scripts/extract_embeddings.py"] = "f"*64
    else:
        value["source_frames"][0]["source_id"] = "a different actual image"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="implementation|frame/source"):
        embedding.load_embedding_data(directory, feature_dir=feature, cache_dir=cache)


def test_missing_backbone_refused_before_any_network_or_deserialization(tmp_path):
    path = tmp_path / "wrong.pth"
    path.write_bytes(b"not official weights")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        embedding.FrozenResNet18(path)


def test_protocol_does_not_select_a_layer_or_grid_from_outcomes():
    plan = embedding.descriptor_plan()
    assert plan["layer"] == "layer1" and plan["dimensions"] == 64
    assert plan["reference_lookback"] == 3
    assert plan["distance_threshold_grid"] == [0, .01, .025, .05, .1, .2, .4]
    changed = copy.deepcopy(plan)
    changed["distance_threshold_grid"].append(.8)
    assert embedding.descriptor_plan() != changed


def test_empty_site_inventory_has_typed_hashes_and_no_invented_scores():
    vectors = np.zeros((0, 3, 64), np.float32)
    assert len(embedding.array_sha256(vectors)) == 64
    result = embedding.distance_gated_scores(np.zeros((0, 3), np.float32), vectors,
                                            np.zeros((0, 3), bool), np.zeros(0, np.int32), .1)
    assert all(value.shape == (0, 3) for value in result)


def toy_info(times):
    frames = [{"source_id": f"month-{m}", "uri": f"https://example.org/{m}.tif",
               "sha256": f"{m:064x}", "acquisition_start": f"2020-{m:02d}-01T00:00:00Z",
               "acquisition_end": f"2020-{m:02d}-28T23:59:59Z", "published_at": None,
               "availability_assumption": "retrospective_month_end", "registration": {"usable": True}}
              for m in range(1, times+1)]
    geometry = {"type": "Polygon", "coordinates": [[[0, 0], [.001, 0], [.001, .001],
                                                     [0, .001], [0, 0]]]}
    return {"region_id": "synthetic-newborn", "metadata": {"frames": frames},
            "geometries": {"1": geometry, "2": geometry}}


def decode(scores, usable, births):
    return decode_events(scores, usable, births, toy_info(scores.shape[1]),
                         DecoderConfig(threshold=.65, confirmations=2, minimum_dwell_days=20),
                         record_origin=datetime(2026, 1, 1, tzinfo=timezone.utc))


def test_newborn_control_and_genuine_nn_change_propose_confirm_without_prebirth_events():
    torchvision = pytest.importorskip("torchvision")
    torch.set_num_threads(2)
    torch.manual_seed(2901)
    model = embedding.FrozenResNet18(encoder=torchvision.models.resnet18(weights=None)).eval()
    sites = np.zeros((31, 43), np.int32)
    sites[8:24, 10:30] = 1
    valid = np.ones(sites.shape, bool)
    prior, _ = embedding.encode_scene(model, np.zeros((3, *sites.shape), np.uint8),
                                      sites, valid, np.array([True]))
    current, _ = embedding.encode_scene(model, np.full((3, *sites.shape), 255, np.uint8),
                                        sites, valid, np.array([True]))
    vectors = np.stack((prior, prior, current, current), axis=1)
    probability = np.array([[.1, .1, .9, .9]], np.float32)
    births = np.array([2], np.int32)
    for threshold in (0, .01):
        scores, mask, distances = embedding.distance_gated_scores(
            probability, vectors, np.ones((1, 4), bool), births, threshold)
        assert distances[0, 2] > .01  # Actual neural encoder output changes, not a mock distance.
        events = decode(scores, mask, births)
        assert [e.state for e in events] == ["proposed", "confirmed_by_model"]
        assert all(e.change_class == "building_addition" for e in events)
        assert [e.quality["support_current_index"] for e in events] == [2, 3]
        assert events[0].quality["support_before_index"] == 1
        assert all(e.quality["support_first_index"] >= births[0] for e in events)
        # Without observed prebirth images, missing data must not invent absence.
        no_history = np.array([[False, False, True, True]])
        missing = embedding.distance_gated_scores(probability, vectors, no_history, births, threshold)
        assert decode(missing[0], missing[1], births) == []
        inventory = embedding.distance_gated_scores(np.full((1, 4), .9, np.float32),
                                                    vectors, np.ones((1, 4), bool),
                                                    np.array([0], np.int32), threshold)
        assert decode(inventory[0], inventory[1], np.array([0], np.int32)) == []


def test_full_prefix_records_and_old_pool_bits_survive_added_future_site_pixels():
    rng = np.random.default_rng(8)
    sites = rng.integers(0, 4, (13, 17), dtype=np.int32)
    valid = rng.random(sites.shape) > .1
    grid = torch.from_numpy(rng.random((1, 64, 4, 5)).astype(np.float32))
    full, _ = embedding.pool_sites(grid, sites, valid, np.ones(3, bool), chunk_pixels=31)
    earlier_sites = np.where(sites == 3, 0, sites)
    prefix, _ = embedding.pool_sites(grid, earlier_sites, valid, np.ones(2, bool), chunk_pixels=31)
    np.testing.assert_array_equal(full[:2], prefix)
    vectors = np.zeros((2, 6, 64), np.float32)
    vectors[0, :2, 0] = vectors[1, :4, 0] = 1
    vectors[0, 2:, 1] = vectors[1, 4:, 1] = 1
    p = np.array([[.1, .1, .9, .9, .9, .9], [.1, .1, .1, .1, .9, .9]], np.float32)
    births = np.array([2, 4], np.int32)
    full_scores = embedding.distance_gated_scores(p, vectors, np.ones((2, 6), bool), births, .01)
    short_scores = embedding.distance_gated_scores(p[:1, :4], vectors[:1, :4],
                                                   np.ones((1, 4), bool), births[:1], .01)
    full_events = [e for e in decode(full_scores[0], full_scores[1], births)
                   if e.quality["support_current_index"] < 4]
    short_events = decode(short_scores[0], short_scores[1], births[:1])
    assert [e.model_dump(mode="json") for e in full_events] == [e.model_dump(mode="json")
                                                               for e in short_events]
    assert len(full_events) == 2

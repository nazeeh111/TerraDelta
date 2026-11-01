from datetime import datetime, timezone
import numpy as np
import pytest
import torch

from terradelta.decoder import DecoderConfig, decode_events
from terradelta.features import candidate_sites, feature_cube
from terradelta.temporal import (
    TEMPORAL_NUMERICS,
    ObservationTransition,
    WatchLSTM,
    causal_smooth,
    hmm_filter,
    make_temporal,
    predict_temporal,
    load_temporal,
)


def test_temporal_checkpoint_rejects_mixed_spatial_models_before_loading_parameters(tmp_path):
    binding = {
        "spatial_weights_sha256": "a" * 64,
        "candidate_threshold": 0.55,
        "spatial_backbone_sha256": "b" * 64,
        "spatial_inference": {
            "tile_size": 128, "context_halo_native_pixels": 64,
            "native_alignment": 16, "internal_input_scale": 2,
            "implementation_sha256": {"spatial.py": "c" * 64, "pretrained_spatial.py": "e" * 64},
        },
    }
    path = tmp_path / "temporal.pt"
    torch.save({"format": "terradelta-temporal-v1", "spatial_provenance": binding}, path)
    with pytest.raises(ValueError, match="requires spatial feature provenance"):
        load_temporal(path)
    with pytest.raises(ValueError, match="spatial model/backbone/inference"):
        load_temporal(path, spatial_info=binding | {"spatial_weights_sha256": "d" * 64})
    model = make_temporal("observation")
    torch.save({"format": "terradelta-temporal-v1", "spatial_provenance": binding,
                "models": {"observation": model.state_dict()}}, path)
    loaded = load_temporal(path, spatial_info=binding)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value, rtol=0, atol=0)
    torch.save({"format": "terradelta-temporal-v1"}, path)
    with pytest.raises(ValueError, match="Legacy temporal"):
        load_temporal(path, spatial_info=binding)


def info(months=8):
    frames = []
    for month in range(1, months + 1):
        frames.append(
            {
                "source_id": f"m{month}",
                "uri": f"https://example.org/m{month}.tif",
                "sha256": str(month) * 64,
                "acquisition_start": f"2020-{month:02d}-01T00:00:00+00:00",
                "acquisition_end": f"2020-{month:02d}-28T23:59:59+00:00",
                "published_at": None,
                "availability_assumption": "retrospective_month_end",
                "registration": {
                    "usable": True,
                    "shift_pixels": [0, 0],
                    "shift_magnitude_pixels": 0,
                },
            }
        )
    return {
        "region_id": "unit",
        "metadata": {"frames": frames},
        "geometries": {
            "1": {
                "type": "Polygon",
                "coordinates": [[[0, 0], [0.001, 0], [0.001, 0.001], [0, 0.001], [0, 0]]],
            }
        },
    }


def decode(p, good=None, context=None):
    p = np.array([p], dtype="float32")
    return decode_events(
        p,
        np.ones_like(p, bool) if good is None else np.array([good], bool),
        np.array([0]),
        context or info(p.shape[1]),
        DecoderConfig(),
        record_origin=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def test_cloud_gap_is_not_removal_or_an_exact_date():
    events = decode([0.05, 0.05, 0.99, 0.99, 0.95, 0.95], [1, 1, 0, 0, 1, 1])
    assert [e.state for e in events] == ["proposed", "confirmed_by_model"]
    event = events[-1]
    assert event.change_class == "building_addition"
    assert event.observation_interval.lower.month == 2
    assert event.observation_interval.upper.month == 5
    assert event.detected_at.month == 6
    assert len(event.sources) == 3
    assert all(s.acquisition_end.month not in [3, 4] for s in event.sources)


def test_initial_presence_and_unknown_are_not_change():
    assert decode([0.95, 0.95, 0.95, 0.95]) == []
    assert decode([0.05, 0.05, 0.05, 0.05], [1, 0, 0, 1]) == []
    assert decode([0.5, 0.5, 0.5, 0.5]) == []


def test_correction_preserves_earlier_proposal_and_confirmation():
    events = decode([0.05, 0.05, 0.95, 0.95, 0.05, 0.05])
    key = events[0].event_id
    history = [e for e in events if e.event_id == key]
    assert [e.state for e in history] == ["proposed", "confirmed_by_model", "retracted"]
    assert history[-1].revision == 3
    assert len({e.event_id for e in events if e.state == "confirmed_by_model"}) == 1


def test_online_decoder_future_edit_invariance():
    a = decode([0.05, 0.05, 0.95, 0.95, 0.95, 0.95])
    b = decode([0.05, 0.05, 0.95, 0.95, 0.05, 0.05])
    prefix_a = [e.model_dump(mode="json") for e in a if e.detected_at.month <= 4]
    prefix_b = [e.model_dump(mode="json") for e in b if e.detected_at.month <= 4]
    assert prefix_a == prefix_b
    first = decode([0.05, 0.05, 0.95, 0.95])
    assert [e.model_dump(mode="json") for e in first] == prefix_a


def test_learned_methods_causal_and_skip_unknown():
    torch.manual_seed(3)
    torch.set_num_threads(1)
    x = torch.rand(2, 8, 17)
    good = torch.ones(2, 8, dtype=torch.bool)
    for model in [
        ObservationTransition(),
        ObservationTransition(spatial_residual=False),
        WatchLSTM(),
    ]:
        if isinstance(model, ObservationTransition):
            torch.nn.init.normal_(model.head.weight, std=0.2)
        model.eval()
        with torch.inference_mode():
            full = model(x, good)
            prefix = model(x[:, :4], good[:, :4])
        torch.testing.assert_close(full[:, :4], prefix, atol=1e-6, rtol=1e-5)
    model = ObservationTransition().eval()
    torch.nn.init.normal_(model.head.weight, std=0.2)
    good[:, 3:6] = False
    with torch.inference_mode():
        out = model(x, good)
    torch.testing.assert_close(out[:, 2], out[:, 5])


@pytest.mark.parametrize("kind", ["observation", "no_quality", "watch_lstm"])
@pytest.mark.parametrize("sites,prefix_sites", [(17, 11), (513, 500), (1025, 519)])
def test_fixed_site_batches_preserve_exact_learned_prefixes(kind, sites, prefix_sites):
    """Future site births must not change earlier arithmetic or event confidence."""
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        torch.manual_seed(2901)
        model = make_temporal(kind, spatial_residual=False)
        # Exercise nonzero learned emissions and recurrent states. The initial
        # zero GRU head would hide the terminal-batch rounding regression.
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.normal_(0, 0.2)

        def require_fixed_site_batch(_model, inputs):
            # Assert the versioned numerical contract even when a particular
            # set of weights happens to round two batch heights identically.
            assert inputs[0].shape[0] == inputs[1].shape[0] == 512

        model.register_forward_pre_hook(require_fixed_site_batch)
        rng = np.random.default_rng(2901)
        x = rng.normal(size=(sites, 8, 17)).astype("float32")
        usable = np.ones((sites, 8), bool)
        usable[::7, 2] = False
        cutoff = 4
        full = predict_temporal(model, x, usable)
        short = predict_temporal(model, x[:prefix_sites, :cutoff],
                                 usable[:prefix_sites, :cutoff])
        edited_x, edited_usable = x.copy(), usable.copy()
        edited_x[:, cutoff:] *= -3
        edited_usable[:, cutoff::2] = False
        edited = predict_temporal(model, edited_x, edited_usable)
        assert full.shape == (sites, 8)
        np.testing.assert_array_equal(full[:prefix_sites, :cutoff], short)
        np.testing.assert_array_equal(full[:, :cutoff], edited[:, :cutoff])

        context = info(8)
        context["geometries"] = {
            str(site): context["geometries"]["1"] for site in range(1, sites + 1)
        }
        births = np.where(np.arange(sites) < prefix_sites, 0, cutoff)
        prefix_context = {
            **context,
            "metadata": {"frames": context["metadata"]["frames"][:cutoff]},
            "geometries": {str(site): context["geometries"][str(site)]
                           for site in range(1, prefix_sites + 1)},
        }
        settings = DecoderConfig(threshold=0.5, confirmations=1, minimum_dwell_days=0)
        kwargs = dict(config=settings, record_origin=datetime(2026, 1, 1,
                                                            tzinfo=timezone.utc))
        long_events = decode_events(full, usable, births, context, **kwargs)
        short_events = decode_events(short, usable[:prefix_sites, :cutoff],
                                     births[:prefix_sites], prefix_context, **kwargs)
        edited_events = decode_events(edited, edited_usable, births, context, **kwargs)
        def historical(events):
            return [event.model_dump(mode="json") for event in events
                    if event.detected_at.month <= cutoff]

        before = historical(long_events)
        assert before  # Include real proposals/confidences, not only empty histories.
        assert before == historical(short_events)
        assert before == historical(edited_events)
    finally:
        torch.set_num_threads(previous_threads)


def test_fixed_site_batch_protocol_keeps_empty_shape_and_rejects_other_sizes():
    assert TEMPORAL_NUMERICS == "fixed512-site-batch-v2"
    x = np.empty((0, 8, 17), dtype="float32")
    usable = np.empty((0, 8), dtype=bool)
    result = predict_temporal(ObservationTransition(), x, usable)
    assert result.shape == (0, 8)
    assert result.dtype == np.float32
    with pytest.raises(ValueError, match="requires batch_size=512"):
        predict_temporal(ObservationTransition(), x, usable, batch_size=256)


def test_smoothing_and_hmm_do_not_update_on_invalid_pixels():
    p = np.array([[0.1, 0.9, 0.1, 0.8]], dtype="float32")
    mask = np.array([[1, 0, 0, 1]], bool)
    for method in [causal_smooth, hmm_filter]:
        out = method(p, mask)
        assert out[0, 0] == out[0, 1] == out[0, 2]


def test_candidate_geometry_and_features_cannot_use_later_images():
    p = np.zeros((6, 2, 24, 24), dtype="float32")
    p[:, 0, 3:7, 3:7] = 0.9
    p[4:, 0, 15:19, 15:19] = 0.9
    valid = np.ones((6, 24, 24), bool)
    frames = np.ones(6, bool)
    sites, births = candidate_sites(p, valid, frames)
    short, short_births = candidate_sites(p[:4], valid[:4], frames[:4])
    assert len(births) > len(short_births)
    assert np.array_equal(sites == 1, short == 1)
    rgb = np.ones((6, 3, 24, 24), dtype="uint8") * 80
    fx, good = feature_cube(p, rgb, valid, info(6)["metadata"], sites)
    sx, sg = feature_cube(p[:4], rgb[:4], valid[:4], info(4)["metadata"], short)
    np.testing.assert_allclose(fx[0, :4], sx[0])
    assert np.array_equal(good[0, :4], sg[0])


def test_elapsed_time_and_seasonal_reference_use_calendar_not_frame_positions():
    context = info(3)["metadata"]
    context["frames"][2]["acquisition_start"] = "2021-01-01T00:00:00+00:00"
    context["frames"][2]["acquisition_end"] = "2021-01-28T23:59:59+00:00"
    p = np.zeros((3, 2, 8, 8), dtype="float32")
    p[:, 0] = np.array([0.1, 0.8, 0.3])[:, None, None]
    rgb = np.ones((3, 3, 8, 8), dtype="uint8") * 80
    valid = np.ones((3, 8, 8), bool)
    sites = np.ones((8, 8), dtype="int32")
    features, _ = feature_cube(p, rgb, valid, context, sites)
    assert np.isclose(features[0, 2, 12], 0.2)
    valid[1] = False
    features, usable = feature_cube(p, rgb, valid, context, sites)
    assert not usable[0, 1]
    assert 11.7 < features[0, 2, 15] < 12
    assert np.isclose(features[0, 2, 12], 0.2)

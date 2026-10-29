from datetime import datetime, timezone
import numpy as np
import torch

from terradelta.decoder import DecoderConfig, decode_events
from terradelta.features import candidate_sites, feature_cube
from terradelta.temporal import ObservationTransition, WatchLSTM, causal_smooth, hmm_filter


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
    for model in [ObservationTransition(), WatchLSTM()]:
        model.eval()
        with torch.inference_mode():
            full = model(x, good)
            prefix = model(x[:, :4], good[:, :4])
        torch.testing.assert_close(full[:, :4], prefix, atol=1e-6, rtol=1e-5)
    model = ObservationTransition().eval()
    good[:, 3:6] = False
    with torch.inference_mode():
        out = model(x, good)
    torch.testing.assert_close(out[:, 2], out[:, 5])


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

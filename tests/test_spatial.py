from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from terradelta.spatial import (
    BuildingUNet,
    load_model,
    masked_segmentation_loss,
    predict_scene,
)


def test_native_odd_shape_and_checkpoint(tmp_path: Path):
    torch.set_num_threads(2)
    torch.manual_seed(3)
    model = BuildingUNet(base_channels=2).eval()
    x = torch.rand(1, 3, 27, 39)
    with torch.no_grad():
        expected = model(x)
    assert expected.shape == (1, 2, 27, 39)
    path = tmp_path / "weights.pt"
    torch.save(
        {
            "format": "terradelta-spatial-v1",
            "architecture": {"base_channels": 2},
            "state_dict": model.state_dict(),
        },
        path,
    )
    actual = load_model(path)(x)
    torch.testing.assert_close(actual, expected)


def test_invalid_pixels_never_change_loss_or_gradient():
    logits = torch.zeros((1, 2, 4, 4), requires_grad=True)
    truth = torch.zeros_like(logits)
    valid = torch.zeros((1, 4, 4), dtype=torch.bool)
    valid[:, :2] = True
    loss = masked_segmentation_loss(logits, truth, valid)
    changed = truth.clone()
    changed[:, :, 2:] = 1
    torch.testing.assert_close(loss, masked_segmentation_loss(logits, changed, valid))
    loss.backward()
    assert torch.count_nonzero(logits.grad[:, :, 2:]) == 0
    assert torch.count_nonzero(logits.grad[:, :, :2]) > 0


def test_fully_unobserved_patch_has_zero_differentiable_loss():
    logits = torch.randn((2, 2, 8, 8), requires_grad=True)
    loss = masked_segmentation_loss(
        logits, torch.ones_like(logits), torch.zeros((2, 8, 8), dtype=torch.bool)
    )
    assert loss.item() == 0
    loss.backward()
    assert torch.count_nonzero(logits.grad) == 0


class PointwiseModel(nn.Module):
    def forward(self, x):
        return torch.stack((x[:, 0] * 3 - 1, x[:, 2] - 0.5), dim=1)


def test_overlap_stitching_preserves_all_pixels_without_seams():
    rng = np.random.default_rng(33)
    rgb = rng.integers(0, 256, (3, 83, 109), dtype=np.uint8)
    model = PointwiseModel()
    with torch.no_grad():
        expected = model(torch.from_numpy(rgb.astype(np.float32) / 255)[None]).sigmoid()[0].numpy()
    actual = predict_scene(model, rgb, tile_size=32)
    assert actual.dtype == np.float32
    assert actual.shape == (2, 83, 109)
    np.testing.assert_allclose(actual, expected, atol=2e-7)
    np.testing.assert_allclose(
        predict_scene(model, rgb[:, :7, :9], tile_size=32), expected[:, :7, :9], atol=2e-7
    )


def test_inference_rejects_ambiguous_band_contract():
    with pytest.raises(ValueError, match="uint8"):
        predict_scene(PointwiseModel(), np.zeros((3, 10, 10), dtype=np.float32))


@pytest.mark.parametrize("input_scale", [1, 2])
def test_unet_context_tiles_match_single_scene_on_pooling_grid(input_scale):
    torch.set_num_threads(2)
    torch.manual_seed(49)
    model = BuildingUNet(base_channels=2, input_scale=input_scale).eval()
    rgb = np.random.default_rng(5).integers(0, 256, (3, 149, 173), dtype=np.uint8)
    with torch.no_grad():
        expected = model(torch.from_numpy(rgb.astype(np.float32) / 255)[None]).sigmoid()[0].numpy()
    tiled = predict_scene(model, rgb, tile_size=64)
    np.testing.assert_allclose(tiled, expected, rtol=1e-5, atol=2e-6)


def test_scaled_native_probabilities_are_arithmetic_cell_means(monkeypatch):
    model = BuildingUNet(base_channels=2, input_scale=2)
    cells = torch.tensor(
        [
            [
                [[0.1, 0.3, 0.2, 0.6], [0.5, 0.7, 0.4, 0.8]],
                [[0.9, 0.7, 0.8, 0.4], [0.5, 0.3, 0.6, 0.2]],
            ]
        ]
    )
    captured = []

    def synthetic_grid(image):
        captured.append(image.detach())
        return torch.logit(cells)

    monkeypatch.setattr(model, "_forward_model_grid", synthetic_grid)
    image = torch.tensor([[[[0.0, 1.0]], [[0.0, 1.0]], [[0.0, 1.0]]]])
    probabilities = model(image).sigmoid()
    torch.testing.assert_close(probabilities, torch.tensor([[[[0.4, 0.5]], [[0.6, 0.5]]]]))
    assert captured[0].shape == (1, 3, 2, 4)
    torch.testing.assert_close(captured[0][0, 0, 0], torch.tensor([0.0, 0.25, 0.75, 1.0]))


def test_scaled_checkpoint_and_changed_input(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(73)
    model = BuildingUNet(base_channels=2, input_scale=2).eval()
    path = tmp_path / "scaled.pt"
    torch.save(
        {
            "format": "terradelta-spatial-v2",
            "architecture": {"base_channels": 2, "input_scale": 2},
            "state_dict": model.state_dict(),
        },
        path,
    )
    loaded = load_model(path)
    assert loaded.input_scale == 2
    rng = np.random.default_rng(81)
    image = rng.integers(0, 256, (3, 31, 39), dtype=np.uint8)
    a = predict_scene(loaded, image, tile_size=32)
    b = predict_scene(loaded, np.zeros_like(image), tile_size=32)
    assert a.shape == (2, 31, 39)
    assert np.isfinite(a).all() and a.min() >= 0 and a.max() <= 1
    assert np.max(np.abs(a - b)) > 1e-7


def test_tiled_encoder_keeps_native_pooling_phase_on_odd_scene():
    from torch import nn
    from terradelta.spatial import predict_scene

    class PhaseProbe(nn.Module):
        native_alignment = 16

        def __init__(self):
            super().__init__()
            self.origins = []

        def forward(self, x):
            self.origins.append(
                (round(float(x[0, 0, 0, 0]) * 255), round(float(x[0, 1, 0, 0]) * 255))
            )
            return torch.zeros((len(x), 2, *x.shape[-2:]))

    rows, cols = np.indices((149, 173), dtype="uint8")
    image = np.stack([cols, rows, np.zeros_like(rows)])
    model = PhaseProbe()
    prediction = predict_scene(model, image, tile_size=64)
    assert len(model.origins) > 1
    assert all(x % 16 == y % 16 == 0 for x, y in model.origins)
    np.testing.assert_array_equal(prediction, np.full((2, 149, 173), 0.5, dtype="float32"))

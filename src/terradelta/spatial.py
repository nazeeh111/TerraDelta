"""Compact native-resolution RGB building segmentation, trained without external weights."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class _DoubleConv(nn.Sequential):
    def __init__(self, inputs: int, outputs: int):
        super().__init__(
            nn.Conv2d(inputs, outputs, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(outputs, outputs, 3, padding=1), nn.ReLU(inplace=True),
        )


class BuildingUNet(nn.Module):
    """Three downsampling levels; returns native-grid footprint/boundary logits.

    Optional 2x interpolation is an internal model operation, not increased
    physical image resolution. Source pixels, targets and output grids stay native.
    """

    def __init__(self, base_channels: int = 16, input_scale: int = 1):
        super().__init__()
        if base_channels < 1:
            raise ValueError("base_channels must be positive")
        if input_scale not in (1, 2):
            raise ValueError("input_scale must be 1 or 2")
        self.base_channels = base_channels
        self.input_scale = int(input_scale)
        self.encoders = nn.ModuleList([
            _DoubleConv(3, base_channels),
            _DoubleConv(base_channels, base_channels * 2),
            _DoubleConv(base_channels * 2, base_channels * 4),
        ])
        self.bottleneck = _DoubleConv(base_channels * 4, base_channels * 8)
        self.ups = nn.ModuleList([
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2),
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2),
            nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2),
        ])
        self.decoders = nn.ModuleList([
            _DoubleConv(base_channels * 8, base_channels * 4),
            _DoubleConv(base_channels * 4, base_channels * 2),
            _DoubleConv(base_channels * 2, base_channels),
        ])
        self.head = nn.Conv2d(base_channels, 2, 1)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError("expected RGB tensor [batch,3,height,width]")
        if self.input_scale == 1:
            return self._forward_model_grid(rgb)
        model_input = F.interpolate(rgb, scale_factor=2, mode="bilinear", align_corners=False)
        high_logits = self._forward_model_grid(model_input)
        # A native pixel contains exactly four interpolated model-grid cells.
        # Aggregate probabilities, not logits, then expose equivalent finite
        # logits so the existing native-mask loss/API retains its contract.
        probabilities = F.avg_pool2d(high_logits.sigmoid(), kernel_size=2, stride=2)
        epsilon = torch.finfo(probabilities.dtype).eps
        return torch.logit(probabilities.clamp(epsilon, 1 - epsilon))

    def _forward_model_grid(self, rgb: torch.Tensor) -> torch.Tensor:
        height, width = rgb.shape[-2:]
        rgb = F.pad(rgb, (0, (-width) % 8, 0, (-height) % 8), mode="replicate")
        skips = []
        for encoder in self.encoders:
            rgb = encoder(rgb)
            skips.append(rgb)
            rgb = F.max_pool2d(rgb, 2)
        rgb = self.bottleneck(rgb)
        for up, decoder, skip in zip(self.ups, self.decoders, reversed(skips)):
            rgb = decoder(torch.cat((up(rgb), skip), dim=1))
        return self.head(rgb)[..., :height, :width]


def masked_segmentation_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    positive_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """BCE plus soft Dice; invalid pixels contribute neither positives nor negatives."""
    if logits.shape != target.shape or logits.ndim != 4 or logits.shape[1] != 2:
        raise ValueError("logits and target must both be [batch,2,height,width]")
    if valid.ndim == 3:
        valid = valid[:, None]
    if valid.shape != (logits.shape[0], 1, *logits.shape[-2:]):
        raise ValueError("valid must be [batch,height,width] or [batch,1,height,width]")
    mask = valid.to(dtype=logits.dtype).expand_as(logits)
    if not bool(torch.any(valid)):
        return logits.sum() * 0.0
    weight = None if positive_weights is None else positive_weights.reshape(2, 1, 1)
    bce = F.binary_cross_entropy_with_logits(
        logits, target, reduction="none", pos_weight=weight
    )
    bce = (bce * mask).sum() / mask.sum()
    probability = torch.sigmoid(logits) * mask
    truth = target * mask
    axes = (0, 2, 3)
    dice = (2 * (probability * truth).sum(axes) + 1) / (
        probability.sum(axes) + truth.sum(axes) + 1
    )
    return bce + (1 - dice).mean()


def load_model(weights: str | Path, device: str = "cpu") -> BuildingUNet:
    """Load our tensor-only checkpoint with its explicit architecture metadata."""
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    if checkpoint.get("format") not in {"terradelta-spatial-v1", "terradelta-spatial-v2"}:
        raise ValueError("unsupported spatial checkpoint format")
    model = BuildingUNet(
        base_channels=int(checkpoint["architecture"]["base_channels"]),
        input_scale=checkpoint["architecture"].get("input_scale", 1),
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval()


def _anchors(size: int, tile: int) -> list[int]:
    if size <= tile:
        return [0]
    positions = list(range(0, size - tile + 1, max(1, tile // 2)))
    if positions[-1] != size - tile:
        positions.append(size - tile)
    return positions


@torch.inference_mode()
def predict_scene(
    model: nn.Module,
    rgb: np.ndarray,
    device: str = "cpu",
    tile_size: int = 256,
) -> np.ndarray:
    """Predict two native-grid probability rasters with overlapping contextual tiles.

    The source/output pixel grid is unchanged. A checkpoint may enable internal
    model interpolation; this adds no observations or physical resolution. Each
    tile gets a 64-native-pixel context halo aligned to the model's pooling grid.
    Only its native-grid core is accumulated with a positive blend window.
    No validity mask is inferred from color; callers must retain their real mask.
    """
    if rgb.ndim != 3 or rgb.shape[0] != 3 or rgb.dtype != np.uint8:
        raise ValueError("rgb must be uint8 [3,height,width]")
    if tile_size < 16 or min(rgb.shape[-2:]) < 1:
        raise ValueError("tile_size must be >=16 and scene dimensions positive")
    model.eval()
    height, width = rgb.shape[-2:]
    total = np.zeros((2, height, width), dtype=np.float32)
    denominator = np.zeros((height, width), dtype=np.float32)
    for top in _anchors(height, tile_size):
        for left in _anchors(width, tile_size):
            bottom, right = min(top + tile_size, height), min(left + tile_size, width)
            y0 = max(0, (top - 64) // 8 * 8)
            x0 = max(0, (left - 64) // 8 * 8)
            y1 = min(height, ((bottom + 64 + 7) // 8) * 8)
            x1 = min(width, ((right + 64 + 7) // 8) * 8)
            patch = np.ascontiguousarray(rgb[:, y0:y1, x0:x1], dtype=np.float32) / 255
            tensor = torch.from_numpy(patch)[None].to(device)
            pred = model(tensor).sigmoid()[0].cpu().numpy()
            pred = pred[:, top - y0:bottom - y0, left - x0:right - x0]
            y_window = np.maximum(np.hanning(bottom - top), 0.05)
            x_window = np.maximum(np.hanning(right - left), 0.05)
            blend = (y_window[:, None] * x_window[None, :]).astype(np.float32)
            total[:, top:bottom, left:right] += pred * blend[None]
            denominator[top:bottom, left:right] += blend
    if np.any(denominator <= 0):
        raise RuntimeError("internal tiling error: uncovered scene pixels")
    return total / denominator[None]

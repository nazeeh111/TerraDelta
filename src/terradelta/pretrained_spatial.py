"""Frozen local ResNet18 features and a native-grid building decoder.

This module never downloads weights. Torchvision's code license and the
pretraining-data/weight terms are separate provenance records.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

FORMAT = "terradelta-pretrained-spatial-v1"
BACKBONE_ENUM = "ResNet18_Weights.IMAGENET1K_V1"
BACKBONE_URI = "https://download.pytorch.org/models/resnet18-f37072fd.pth"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _block(inputs: int, outputs: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(inputs, outputs, 3, padding=1, bias=False),
        nn.GroupNorm(8, outputs), nn.ReLU(),
        nn.Conv2d(outputs, outputs, 3, padding=1, bias=False),
        nn.GroupNorm(8, outputs), nn.ReLU(),
    )


class PretrainedBuildingUNet(nn.Module):
    """RGB [0,1] to native [N,2,H,W] footprint/boundary equivalent logits.

    ``encoder`` injection is for unit tests only. Actual fits require a complete
    local official checkpoint and full SHA256, checked before deserialization.
    """

    def __init__(
        self, backbone_path: str | Path | None = None,
        backbone_sha256: str | None = None, input_scale: int = 2,
        *, encoder: nn.Module | None = None,
    ):
        super().__init__()
        if input_scale not in (1, 2):
            raise ValueError("input_scale must be 1 or 2")
        self.input_scale = input_scale
        self.test_encoder = encoder is not None
        if encoder is None:
            if (backbone_path is None or backbone_sha256 is None
                    or len(backbone_sha256) != 64
                    or any(c not in "0123456789abcdef" for c in backbone_sha256)):
                raise ValueError("local backbone path and full lowercase SHA256 are required")
            if sha256_file(backbone_path) != backbone_sha256:
                raise ValueError("backbone SHA256 mismatch")
            from torchvision.models import resnet18

            encoder = resnet18(weights=None)  # explicitly disables network access
            encoder.load_state_dict(
                torch.load(backbone_path, map_location="cpu", weights_only=True), strict=True
            )  # Includes classifier: no silently omitted keys.
            encoder.fc = nn.Identity()  # Excluded only AFTER successful complete load.
        self.backbone_sha256 = backbone_sha256
        self.encoder = encoder.requires_grad_(False).eval()
        self.decoder = nn.ModuleDict({
            "blocks": nn.ModuleList([
                _block(a, b) for a, b in
                ((768, 128), (256, 64), (128, 32), (96, 16), (19, 16))
            ]),
            "head": nn.Conv2d(16, 2, 1),
        })
        self.register_buffer("mean", torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([.229, .224, .225])[None, :, None, None])

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()  # Frozen parameters alone do not freeze BatchNorm statistics.
        return self

    def _forward_model_grid(self, rgb: torch.Tensor) -> torch.Tensor:
        height, width = rgb.shape[-2:]
        rgb = F.pad(rgb, (0, (-width) % 32, 0, (-height) % 32), mode="replicate")
        normalized = (rgb - self.mean) / self.std
        with torch.no_grad():
            stem = self.encoder.relu(self.encoder.bn1(self.encoder.conv1(normalized)))
            value = self.encoder.maxpool(stem)
            stages = [rgb, stem]
            for layer in (self.encoder.layer1, self.encoder.layer2,
                          self.encoder.layer3, self.encoder.layer4):
                value = layer(value)
                stages.append(value)
        for block, skip in zip(self.decoder["blocks"], reversed(stages[:-1])):
            value = block(torch.cat((F.interpolate(
                value, size=skip.shape[-2:], mode="bilinear", align_corners=False
            ), skip), dim=1))
        return self.decoder["head"](value)[..., :height, :width]

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        if rgb.ndim != 4 or rgb.shape[1] != 3 or min(rgb.shape[-2:]) < 1:
            raise ValueError("expected nonempty RGB tensor [batch,3,height,width]")
        if self.input_scale == 1:
            return self._forward_model_grid(rgb)
        high = F.interpolate(rgb, scale_factor=2, mode="bilinear", align_corners=False)
        probability = F.avg_pool2d(self._forward_model_grid(high).sigmoid(), 2, 2)
        epsilon = torch.finfo(probability.dtype).eps
        return torch.logit(probability.clamp(epsilon, 1 - epsilon))


def make_checkpoint(model: PretrainedBuildingUNet, training: dict) -> dict:
    if model.test_encoder or model.backbone_sha256 is None:
        raise ValueError("test/random encoder cannot produce a research checkpoint")
    return {
        "format": FORMAT,
        "architecture": {"name": "FrozenResNet18GroupNormDecoder", "input_scale": model.input_scale,
                         "group_norm_groups": 8, "outputs": ["footprint", "boundary"],
                         "native_output": "mean internal-cell probabilities; equivalent logits"},
        "backbone": {"uri": BACKBONE_URI, "sha256": model.backbone_sha256,
                     "enum": BACKBONE_ENUM, "torchvision": "0.24.1",
                     "classifier": "strictly loaded then excluded as Identity",
                     "terms": {"code": "https://github.com/pytorch/vision/blob/v0.24.1/LICENSE",
                               "weights_notice": "https://github.com/pytorch/vision/blob/v0.24.1/README.md",
                               "dataset": "https://image-net.org/accessagreement"}},
        "decoder_state_dict": {k: v.detach().cpu().clone()
                               for k, v in model.decoder.state_dict().items()},
        "training": training,
    }


def load_model(weights: str | Path, *, backbone_path: str | Path,
               device: str = "cpu") -> PretrainedBuildingUNet:
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != FORMAT:
        raise ValueError("unsupported pretrained spatial checkpoint format")
    backbone = checkpoint["backbone"]
    if backbone["enum"] != BACKBONE_ENUM or backbone["uri"] != BACKBONE_URI:
        raise ValueError("unsupported backbone identity")
    architecture = checkpoint["architecture"]
    if (architecture["name"] != "FrozenResNet18GroupNormDecoder"
            or architecture["group_norm_groups"] != 8):
        raise ValueError("unsupported decoder architecture")
    model = PretrainedBuildingUNet(backbone_path, backbone["sha256"], architecture["input_scale"])
    model.decoder.load_state_dict(checkpoint["decoder_state_dict"], strict=True)
    return model.to(device).eval()

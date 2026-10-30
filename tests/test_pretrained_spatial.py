"""Unit data are random; no study cache, network weights or training run is used."""
from pathlib import Path

import pytest
import torch
from torch import nn

from terradelta.pretrained_spatial import (
    PretrainedBuildingUNet, load_model, make_checkpoint, sha256_file,
)
from terradelta.spatial import masked_segmentation_loss


class TinyEncoder(nn.Module):
    """Only the skip channel/stride contract; never used by the fit command."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = nn.Conv2d(64, 64, 3, padding=1)
        self.layer2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.layer3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.layer4 = nn.Conv2d(256, 512, 3, stride=2, padding=1)


@pytest.fixture
def model():
    torch.set_num_threads(2)
    torch.manual_seed(319)
    return PretrainedBuildingUNet(encoder=TinyEncoder())


def test_native_odd_outputs_and_changed_pixels(model):
    model.eval()
    with torch.no_grad():
        a = model(torch.rand(1, 3, 31, 47)).sigmoid()
        b = model(torch.zeros(1, 3, 31, 47)).sigmoid()
        tiny = model(torch.rand(1, 3, 1, 3))
    assert a.shape == (1, 2, 31, 47)
    assert tiny.shape == (1, 2, 1, 3)
    assert torch.isfinite(a).all() and torch.isfinite(tiny).all()
    assert 0 < a.min() < a.max() < 1
    assert (a - b).abs().max() > 1e-5


def test_encoder_parameters_and_bn_frozen_across_optimizer_step(model):
    before = {k: v.clone() for k, v in model.encoder.state_dict().items()}
    decoder_before = {k: v.clone() for k, v in model.decoder.state_dict().items()}
    optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=.001)
    model.train()
    assert model.training and model.decoder.training and not model.encoder.training
    assert not model.encoder.bn1.training
    truth = torch.randint(0, 2, (2, 2, 19, 27)).float()
    loss = masked_segmentation_loss(model(torch.rand(2, 3, 19, 27)), truth,
                                    torch.ones(2, 19, 27, dtype=torch.bool))
    loss.backward()
    optimizer.step()
    for key, value in before.items():
        assert torch.equal(model.encoder.state_dict()[key], value), key
    assert all(not p.requires_grad and p.grad is None for p in model.encoder.parameters())
    assert any(not torch.equal(value, model.decoder.state_dict()[key])
               for key, value in decoder_before.items())


def test_invalid_mask_yields_zero_loss_and_decoder_gradient(model):
    logits = model(torch.rand(1, 3, 17, 23))
    loss = masked_segmentation_loss(logits, torch.ones_like(logits),
                                    torch.zeros(1, 17, 23, dtype=torch.bool))
    assert loss.item() == 0
    loss.backward()
    assert all(p.grad is None or not torch.count_nonzero(p.grad)
               for p in model.decoder.parameters())


def test_native_probability_aggregation_and_internal_interpolation(model, monkeypatch):
    cells = torch.tensor([[[[.1, .3, .2, .6], [.5, .7, .4, .8]],
                           [[.9, .7, .8, .4], [.5, .3, .6, .2]]]])
    captured = []

    def high_grid(x):
        captured.append(x)
        return torch.logit(cells)

    monkeypatch.setattr(model, "_forward_model_grid", high_grid)
    x = torch.tensor([[[[0., 1.]], [[0., 1.]], [[0., 1.]]]])
    torch.testing.assert_close(model(x).sigmoid(),
                               torch.tensor([[[[.4, .5]], [[.6, .5]]]]))
    torch.testing.assert_close(captured[0][0, 0, 0], torch.tensor([0., .25, .75, 1.]))


def test_random_injection_cannot_be_serialized_as_research_checkpoint(model):
    with pytest.raises(ValueError, match="test/random"):
        make_checkpoint(model, {})


def test_backbone_hash_rejected_before_import_or_deserialization(tmp_path):
    path = tmp_path / "not-weights.pth"
    path.write_bytes(b"intentionally invalid checkpoint")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        PretrainedBuildingUNet(path, "0" * 64)
    with pytest.raises(ValueError, match="full lowercase SHA256"):
        PretrainedBuildingUNet(path, "f37072fd")


def test_complete_backbone_load_decoder_only_roundtrip(tmp_path: Path, monkeypatch):
    torchvision = pytest.importorskip("torchvision")
    torch.set_num_threads(2)
    torch.manual_seed(89)

    def network_forbidden(*args, **kwargs):
        raise AssertionError("loader attempted a network download")

    monkeypatch.setattr(torch.hub, "download_url_to_file", network_forbidden)
    backbone = tmp_path / "random-unit-test-backbone.pth"
    state = torchvision.models.resnet18(weights=None).state_dict()
    torch.save(state, backbone)
    model = PretrainedBuildingUNet(backbone, sha256_file(backbone)).eval()
    checkpoint = make_checkpoint(model, {"unit_test_only": True})
    assert "state_dict" not in checkpoint
    assert set(checkpoint) == {"format", "architecture", "backbone",
                               "decoder_state_dict", "training"}
    assert set(checkpoint["decoder_state_dict"]) == set(model.decoder.state_dict())
    assert not any("encoder" in key or "running_" in key or "num_batches_tracked" in key
                   for key in checkpoint["decoder_state_dict"])
    assert sum(p.numel() for p in model.parameters()) == 12_461_330
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 1_284_818
    path = tmp_path / "decoder.pt"
    torch.save(checkpoint, path)
    loaded = load_model(path, backbone_path=backbone)
    x = torch.rand(1, 3, 35, 41)
    with torch.no_grad():
        assert torch.equal(model(x), loaded(x))
    # A full state is required, including classifier keys before its exclusion.
    del state["fc.weight"]
    incomplete = tmp_path / "incomplete.pth"
    torch.save(state, incomplete)
    with pytest.raises(RuntimeError, match="fc.weight"):
        PretrainedBuildingUNet(incomplete, sha256_file(incomplete))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        load_model(path, backbone_path=incomplete)

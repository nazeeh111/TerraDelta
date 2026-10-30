# Bounded frozen-backbone spatial revision

This is a new research revision after the preserved failed v2 study. Its accuracy
is unknown until the prospective split is fitted and evaluated. Old confirmation
regions are audit-only. They must not enter new training, model selection or
operating-point calibration. No new confirmation labels or image outcomes are
needed by this model implementation or trainer.

## Model and native-grid contract

`PretrainedBuildingUNet` uses Torchvision 0.24.1's ResNet18 architecture and the
explicit `ResNet18_Weights.IMAGENET1K_V1` identity. A caller supplies a local
backbone file and its complete SHA256. The code hashes it before deserialization,
constructs `resnet18(weights=None)` and strictly loads the **complete** state,
including classifier keys. Only then is the unused classifier replaced with
`Identity`. There is no automatic download, alternate checkpoint, partial load or
random-backbone fallback in the training route.

All encoder parameters have gradients disabled. The model's `train()` override
keeps the encoder, including BatchNorm running statistics and counters, in
evaluation mode. Forward extraction also uses `no_grad`. Only the skip decoder
is optimized. Its five two-convolution blocks have widths 128, 64, 32, 16, 16,
eight GroupNorm groups, bilinear upsampling, early stem and dense RGB skips.
The classifier-free model has 12,461,330 parameters, of which 1,284,818 train.

Input is RGB float32 `[N,3,H,W]` in `[0,1]`. The encoder alone receives ImageNet
mean `[.485,.456,.406]` and standard deviation `[.229,.224,.225]` normalization.
Classification resize/crop transforms are not applied. The configured internal
2x bilinear interpolation is a model operation. It supplies no new observations
or physical resolution. The internal grid is replicate-padded to multiples of 32
for pooling; padding is cropped before each native 2x2 cell's probabilities are
averaged. Returned finite equivalent logits have shape `[N,2,H,W]`, footprint
then instance boundary, including odd native dimensions. Geotransforms, CRS,
nodata and source masks remain the caller's responsibility and are unchanged.

The existing native-grid overlapping `spatial.predict_scene` API accepts this
model. GroupNorm depends on each tile's spatial context, so fixed tiling is part
of the processing protocol; full-frame and tiled outputs are not asserted to be
identical. Inference tiling must stay fixed across compared methods and prefixes.

## Checkpoint and API

```python
from terradelta.pretrained_spatial import load_model
from terradelta.spatial import predict_scene

model = load_model("decoder.pt", backbone_path="resnet18-f37072fd.pth")
probabilities = predict_scene(model, rgb_uint8, tile_size=256)
```

`make_checkpoint(model, training)` stores `decoder_state_dict`, explicit
architecture, backbone URI/full hash/enum/package and training provenance.
It stores no encoder tensors, running statistics or optimizer state. Unit-test
encoder injection is available through `encoder=...`; such a model cannot
produce a research checkpoint. `load_model` requires the separate local backbone
and strictly restores decoder keys. A backbone hash mismatch is an error.

The backbone is intentionally separate from decoder artifacts. Its official URI
is [PyTorch's ResNet18 file](https://download.pytorch.org/models/resnet18-f37072fd.pth).
The model definition is [Torchvision's pinned ResNet source](https://github.com/pytorch/vision/blob/v0.24.1/torchvision/models/resnet.py).
Keep [Torchvision's BSD code license](https://github.com/pytorch/vision/blob/v0.24.1/LICENSE),
its [pretrained-model notice](https://github.com/pytorch/vision/blob/v0.24.1/README.md)
and the [ImageNet data agreement](https://image-net.org/accessagreement) as separate
terms. Decoder-only packaging avoids including the backbone bytes; it is not a
determination of commercial, redistribution or derived-weight rights. The model
does not claim a BSD license for pretrained weights or unrestricted commercial
rights.

## Fit protocol and limits

`scripts/train_spatial_pretrained.py` reads only the config's nine TRAIN and four
DEV caches. The three prospective confirmation regions and old audit-only
regions are excluded. It checks disjoint roles and the complete configured
backbone identity. `spatial_backbone` requires `uri`, `sha256`, `enum`.
`resource_limits` requires `patch_size=128`, `development_patch_size=128`,
`batch_size=4`, `input_scale=2`, `cpu_threads=2`,
`spatial_training_steps<=4000` and `spatial_wall_seconds<=10800`.

Training reuses the existing sampler: half of sampled crops seek buildings,
with at most five attempts for valid coverage and foreground; exact rotations
and reflection transform RGB, both targets and masks together. RGB receives
per-channel gain in `[.85,1.15]` and scalar offset in `[-.05,.05]`, clipped to
`[0,1]`. There is no geometric resizing of labels. The existing native valid-mask
BCE plus Dice loss uses footprint/boundary positive weights 3/6. Invalid pixels
contribute no loss, and an entirely invalid batch is rejected in actual fits.
AdamW updates decoder parameters only, learning rate .001, decay .0001, gradient
norm clipping at 5. There is no consistency loss or further objective change.

DEV selection uses the same fixed sampling helper: four evenly spaced frames
and four spatial positions per frame, sixteen native128 crops per AOI. The
selected score is mean footprint/boundary Dice at .5. Evaluation occurs at step
0, step 1, every 100 steps and the requested last step. The step0 fallback is
explicitly recorded if no trained decoder improves it. Selection uses no event
threshold or confirmation result.

Run the actual TRAIN-only pilot first, with a distinct output stem:

```sh
PYTHONPATH=src python scripts/train_spatial_pretrained.py \
  --cache /path/to/cache --config configs/experiment-v5.json \
  --backbone /path/to/resnet18-f37072fd.pth --backbone-sha256 FULL_SHA256 \
  --output /path/to/spatial-pretrained-pilot.pt --pilot-only --device mps
```

The pilot performs one actual masked TRAIN update, verifies every encoder tensor
and BatchNorm statistic unchanged, and saves only `.pilot.json`. Its decoder
update is discarded; a subsequent full fit starts from the frozen seed. It
reads no DEV cache. The full-fit command omits `--pilot-only` and uses a fresh
output stem. Training is not launched by importing the module.

The trainer refuses preexisting output/checkpoint/trace paths. Atomic best-state
saves retain complete input-array/metadata, config, source and backbone hashes.
No optimizer copies are serialized. A final trace records every selection point,
actual completed updates, stop reason and wall time. An alarm limits optimization
and validation to the frozen wall budget, including initialization time; Python
can handle it only after a current native operation returns. Final frozen-state
audit and artifact serialization occur after the alarm is disarmed. Source or
config mutation during fitting is an error. CPU fallback is explicit in the
trace; it must obey the same step and wall limits.

## Acceptance

`pytest tests/test_pretrained_spatial.py` checks an optimizer step cannot change
encoder tensors or BatchNorm state, odd native outputs, changed-input response,
invalid-mask zero loss/gradient, native probability averaging, rejection of
random-injection research exports, full-hash rejection before loading, complete
strict backbone loading, decoder-only keys and exact roundtrip. The roundtrip
uses a random local official-architecture state strictly for a unit test, never
a research fit. Tests perform no downloads. Local package/checkpoint loading,
TRAIN pilot and actual fit are separate execution evidence; unit tests do not
establish useful building geometry or event accuracy.

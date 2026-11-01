# Data and trained artifact terms

SpaceNet 7 imagery and labels are by SpaceNet Partners, including Planet Labs
imagery, distributed under **Creative Commons Attribution-ShareAlike 4.0**.
See [the original dataset](https://spacenet.ai/sn7-challenge/) and
[license terms](https://creativecommons.org/licenses/by-sa/4.0/).

Permitted dataset copies, probability rasters, image previews, exported event
geometries and trained artifacts distributed with this project retain
**CC-BY-SA-4.0** attribution and share-alike terms. Code is separately MIT licensed.
This conservative artifact designation does not claim to resolve general legal
questions about learned weights. Third-party code retains its original notices.

The experimental `spatial-v3.pt` contains only TerraDelta decoder parameters.
Its frozen ImageNet ResNet18 encoder is a separate official input and is not
redistributed. Reproduction requires that input and its applicable terms.
[Torchvision's BSD code license](https://github.com/pytorch/vision/blob/v0.24.1/LICENSE)
does not establish blanket rights for the pretrained weights or ImageNet data;
see the [upstream weights notice](https://github.com/pytorch/vision/blob/v0.24.1/README.md)
and [ImageNet access terms](https://image-net.org/accessagreement).
This project's conservative CC designation does not override upstream terms or
grant unrestricted commercial rights to that separate input.

Cite Adam Van Etten, Daniel Hogan, Jesus Martinez Manso, Jacob Shermeyer, Nicholas
Weir and Ryan Lewis, *The Multi-Temporal Urban Development SpaceNet Dataset*,
CVPR 2021, [paper](https://arxiv.org/abs/2102.04420).

Accessed October 29, 2026 through unsigned public AWS HTTPS. Every acquired source
is identified by its object key and SHA-256 in the frozen data manifest. AWS
LastModified is distribution metadata and is not treated as the original image
publication time. Calendar bounds in the source contract denote the mosaic's
filename reporting month. Individual contributing scene and per-pixel acquisition
timestamps are unavailable; these bounds are not a verified timestamp for every
pixel. Matched building labels are postprocessed and can exceed the visible
precision of the approximately 4 m imagery. No label-assisted alignment is used.

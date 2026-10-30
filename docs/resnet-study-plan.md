# Next spatial revision: bounded feasibility assessment

2026-10-29. Independent research and synthetic-runtime assessment. Repository, weights and study data were not modified. No fits, package installations, new label-content reads, new image-pixel interpretation or confirmation-performance reads were performed. Parent-reported first-study evidence is a geometry ceiling around .24–.29 and event-gate precision 3.9%, recall 1.8%; those numbers motivate this assessment but were not independently recomputed here.

## Recommended single path

Proceed with one research revision: an **official ImageNet ResNet18 encoder, frozen including BatchNorm statistics, plus a small trainable GroupNorm skip decoder**. Use native 128-pixel crops, optional internal 256-pixel interpolation, native-grid footprint/boundary outputs, and the existing physical-coverage truth protocol. Cap the first fit at **4,000 updates or three hours**, whichever arrives first. The local synthetic MPS evidence supports feasibility; it does not establish useful accuracy. Keep the failed first study and its original confirmation results intact. Its three confirmation AOIs are now audit-only, excluded from new fitting and model/threshold selection.

This is the best bounded path among the four reviewed options because it has the smallest external download, narrowest dependency change, measured local operator compatibility and lower risk of having been trained directly on the new SN7 confirmation regions. Its weight-rights provenance must be described accurately: Torchvision's BSD code license does **not** prove that its pretrained weights are BSD licensed. A decoder-only release avoids bundling the encoder but does not settle every possible right in derived weights.

## Evidence and alternatives

| Option | Official weight/provenance evidence | Fit to this revision | Decision |
|---|---|---|---|
| ResNet18 ImageNet1K V1 | Official Torchvision enum; 11.69M classifier parameters, 44.7 MB documented weight size; direct public PyTorch download | Small dependency change; measured compatible PyTorch operators; generic natural-image features, so transfer to tiny roofs is uncertain | Select for research; retain separate terms and provenance |
| Satlas aerial | Explicit ODC-BY checkpoint notice; released aerial variant is Swin-v2-Base; **504 MB** public checkpoint | Appropriate high-resolution pretraining, but larger than presumed; no local runtime measurement; exceeds this nine-AOI download budget | Do not implement in this bounded revision |
| TerraMind tiny | Apache-2.0 model card; **212 MB** checkpoint; global Sentinel-derived pretraining | More setup, coarser imagery and temporal/geographic overlap uncertainty; not locally executed | Reserve for a separately justified study |
| Original SN7 winning ensemble | Five folds collectively cover all 60 public training AOIs | Direct training-domain contamination for prospective public-train confirmation AOIs | Exclude |

### ResNet18 rights, source and dependency limits

Pin `torchvision==0.24.1` and `ResNet18_Weights.IMAGENET1K_V1`, rather than the movable `DEFAULT` alias. The official checkpoint is `https://download.pytorch.org/models/resnet18-f37072fd.pth`. The filename hash prefix is `f37072fd`; the complete SHA256 has **not** been independently measured because no model was downloaded. Compute and freeze it on the first authorized download. Torchvision's loader requests hash checking. The classification transform includes resize and center crop, which must not be applied to georeferenced dense labels; use its RGB normalization, explicit internal interpolation and native-grid output instead. [Official ResNet documentation](https://docs.pytorch.org/vision/0.24/models/generated/torchvision.models.resnet18.html), [pinned model source](https://github.com/pytorch/vision/blob/v0.24.1/torchvision/models/resnet.py).

Torchvision's code is BSD-3-Clause, while its own pretrained-model notice says that weights may have training-dataset terms. ImageNet's dataset agreement permits noncommercial research/education and does not claim ownership of underlying image copyrights. Neither source supplies a clean, separate blanket permissive license for this particular `.pth`. This supports a carefully documented research experiment, **not a claim of unrestricted commercial or redistribution rights**. Keep the official URL, full downloaded SHA, exact enum/package, attribution, code license and dataset notice separately. Release new decoder tensors only if a release is later authorized; inspect exported keys to exclude encoder parameters, encoder running statistics and optimizer copies. Such packaging is technical non-redistribution of the backbone, not a legal determination about derived decoders. [Torchvision notice and compatibility table](https://raw.githubusercontent.com/pytorch/vision/v0.24.1/README.md), [code license](https://raw.githubusercontent.com/pytorch/vision/v0.24.1/LICENSE), [ImageNet agreement](https://image-net.org/accessagreement), [ImageNet copyright explanation](https://www.image-net.org/about.php).

The model is ImageNet-2012 classification pretraining. Exact 2018–2020 SN7 Planet acquisitions would not normally occur in that older dataset; this is an inference from provenance, not an image-deduplication certificate or proof that no landmark/geography overlaps. [Official ImageNet dataset documentation](https://docs.pytorch.org/vision/0.24/generated/torchvision.datasets.ImageNet.html).

Existing environment: Python 3.12.13, Torch 2.9.1, macOS arm64; torchvision, timm, transformers, huggingface_hub, safetensors and terratorch were absent. The official 2.9/0.24 compatibility table covers this Python version. PyPI supplies a CPython-3.12/macOS-arm64 torchvision 0.24.1 wheel around 1.9 MB, SHA256 `e48bf6a8ec95872eb45763f06499f87bd2fb246b9b96cb00aae260fda2f96193`. Install only this compatible wheel without dependency replacement after main coordinates the change; preserve Torch 2.9.1. Actual import and official checkpoint loading remain untested. [Official wheel listing](https://pypi.org/project/torchvision/0.24.1/).

### Satlas: better explicit checkpoint terms, wrong presumed size

The official model table and runtime registry contain **Aerial_SwinB_SI/MI only** for aerial RGB; their smaller ResNet50/SwinTiny entries are Sentinel-2 models. There is no currently advertised official NAIP ResNet50/SwinTiny checkpoint in these two sources. Choosing the Sentinel-2 ResNet50 would not deliver the requested high-resolution NAIP-pretrained alternative. The aerial single-image file is public without an authentication gate, 504 MB, official SHA256 `f8abf21cadaab2a71859e9d172bec4e5d6b7cc9710882a2f6a4d8aae826db036`. No weight bytes were downloaded. [Registry](https://raw.githubusercontent.com/allenai/satlaspretrain_models/main/satlaspretrain_models/utils.py), [official checkpoint metadata](https://huggingface.co/allenai/satlas-pretrain/blob/main/aerial_swinb_si.pth).

The official library explicitly says checkpoints use ODC-BY and that aerial models use RGB NAIP **and other high-resolution images**. Multi-image models pool across time; a complete-sequence temporal pool must never enter an online system. The single-image variant avoids that integration issue. The repository's HF Apache badge conflicts with the more specific checkpoint notice, so preserve the explicit ODC-BY notice rather than relabeling weights Apache. [Model documentation](https://raw.githubusercontent.com/allenai/satlaspretrain_models/main/README.md).

ODC-BY explicitly grants royalty-free uses including commercial use, with notices/attribution conditions; it also distinguishes database rights from separate rights in contents. This is substantially clearer affirmative checkpoint permission than Torchvision's generic disclaimer, without proving every downstream right. [Official data/model page](https://raw.githubusercontent.com/allenai/satlas/main/SatlasPretrain.md), [license text, sections 2–4](https://raw.githubusercontent.com/allenai/satlas/main/DataLicense).

NAIP is US coverage, historically around 1m and later finer; dense buildings are a reasonable transfer motivation for 4m Planet imagery, but a resolution gap remains. Excluding US confirmation sites would reduce NAIP geographic overlap, not prove no overlap with the model's other imagery or global labels. No complete pretraining-tile intersection audit was performed. Satlas is not selected, so no post hoc country filter was added to the hash-based split. [USGS NAIP coverage/resolution](https://www.usgs.gov/centers/eros/science/usgs-eros-archive-aerial-photography-national-agriculture-imagery-program-naip).

Its source requires Torch, torchvision, requests and matplotlib. The loader defaults to CUDA and can use CPU before a deliberate move to MPS; this is source feasibility, not an executed MPS compatibility claim. The 504 MB full checkpoint must be acquired even if only its backbone is retained. Substituting it for the 47 MB ResNet reserve gives roughly **3.164 GB** total, before any additional dependencies, over the 3 GB cap. A larger budget or fewer prospectively selected sequences would require a different study. [Loader source](https://raw.githubusercontent.com/allenai/satlaspretrain_models/main/satlaspretrain_models/model.py), [package requirements](https://raw.githubusercontent.com/allenai/satlaspretrain_models/main/setup.py).

### TerraMind and SN7 checkpoints

TerraMind tiny has an explicit Apache-2.0 model card and a 212 MB public checkpoint, SHA256 `e56ea9ebcd4451078b9ca4893d5cd8a89bbee376ae16829c3e7fbbbc76de0eba`. Its RGB pathway is Sentinel-2 RGB; TerraMesh contains global 2017–2024 imagery at a 10m grid. These dates overlap and can postdate SN7's event windows. Without intersecting pretraining locations/time, do not call prospective SN7 sites unseen. [Model card](https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-tiny), [checkpoint metadata](https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-tiny/blob/main/TerraMind_v1_tiny.pt), [pretraining data description](https://huggingface.co/datasets/ibm-esa-geospatial/TerraMesh).

TerraTorch brings a broader dependency surface. Its tiny registration uses 12 transformer blocks, width 192, and Sentinel-RGB-specific normalization; the loader fills missing shape-compatible coverage with initialized model values, so a future integration must inspect actual loaded keys, not trust `pretrained=True`. IBM also documents a UNetDecoder/MPS normalization issue. No TerraMind install or runtime test was performed. [Registration source](https://raw.githubusercontent.com/terrastackai/terratorch/main/terratorch/models/backbones/terramind/model/terramind_register.py), [official setup/runtime notes](https://raw.githubusercontent.com/IBM/terramind/main/README.md).

The reviewed SN7 winner explicitly splits **all 60 public training AOIs** into five folds and ensembles ten models at two upscales. Its full-sequence mask averaging is retrospective. The ensemble would therefore train on the proposed new public-train confirmation regions. A single fold could only be reconsidered with exact training-membership evidence excluding every heldout site; that is not this bounded path. Code-license permissiveness alone does not fix scientific contamination. [Pinned winner description, commit ff1241f](https://raw.githubusercontent.com/SpaceNetChallenge/SpaceNet7_Multi-Temporal_Solutions/ff1241f01e391248a9d6111a60c97973fc5a9643/4-motokimura/motokimura-solution-description.md).

## Measured local compute budget

Evidence: `next-spatial-feasibility-evidence/runtime_probe.py`, `cpu-frozen-probe.json`, `mps-frozen-probe.json` and unfrozen comparison JSON. This uses a PyTorch ResNet18-shaped encoder and a light decoder, random tensors, **zero optimizer updates**, no study imagery/labels. It is not actual torchvision checkpoint execution. MPS was unavailable in the default sandbox but available in the approved process outside that restriction; finite forward/backward was verified.

Batch 4, RGB 128→internal 256, native 128 output with two channels. Full parameter count 12,461,330; frozen-encoder decoder trainable count 1,284,818. Eight passes, three warmup and five measured:

| Probe | Median seconds per forward/backward | Measured range | Implication |
|---|---:|---:|---|
| Frozen encoder, MPS | .397 | .147–.518 | 4,000 passes ~26.4min raw operator time |
| Frozen encoder, CPU two threads | 2.632 | 2.393–5.140 | 4,000 passes ~2.93h before overhead |
| Unfrozen comparison, MPS | 1.114 | .307–1.445 | More cost; not recommended first fit |

Frozen MPS driver allocation was 1.262GB, tensor allocation 63.9MB, recommended device working set 26.8GB. These timings exclude raster I/O, augmentation, masked BCE/Dice, optimizer and validation. Therefore the three-hour cap is a **hard stop**, not a predicted completion time. First verify actual checkpoint loading, fixed encoder statistics and one real TRAIN-only masked batch in FP32; then measure a bounded pilot before committing the remaining budget. CPU fallback requires shorter fitting or a revised budget; do not silently spend beyond the cap.

## Metadata-only fresh split proposal

Retain the old five TRAIN and two DEV AOIs. Add four TRAIN, two DEV and three CONFIRMATION. Old confirmation AOIs remain audit-only. Source: unsigned public S3 key/size listings under `spacenet/SN7_buildings/train/`; neither label contents nor image pixels were read. Catalog files and the complete selection algorithm are retained in `next-spatial-feasibility-evidence/`.

Rank unused AOIs by SHA256 of `terradelta-resnet-fresh-v1|2901|` + exact region ID. Require at least 20 monthly images and complete image/label/UDM filename sets. Choose three confirmation first and two development next, each at least 500km from all original ten AOIs and previously chosen heldouts. Choose four train at least 500km from every old/new DEV and new CONF, and at least 50km from one another. Distances use WGS84 geodesics between encoded WebMercator tile centers. No scores, labels, event counts or visible development outcomes informed selection. `selection-summary.json` records complete old+new memberships.

| New role | Region | Months | Full required sequence bytes |
|---|---|---:|---:|
| CONF | L15-1172E-1306N_4688_2967_13 | 24 | 131,135,942 |
| CONF | L15-1204E-1204N_4819_3372_13 | 25 | 118,401,880 |
| CONF | L15-0358E-1220N_1433_3310_13 | 24 | 147,555,771 |
| DEV | L15-1716E-1211N_6864_3345_13 | 24 | 190,040,280 |
| DEV | L15-0614E-0946N_2459_4406_13 | 21 | 232,669,594 |
| TRAIN | L15-1669E-1160N_6678_3548_13 | 26 | 129,697,310 |
| TRAIN | L15-1335E-1166N_5342_3524_13 | 25 | 111,687,971 |
| TRAIN | L15-0487E-1246N_1950_3207_13 | 25 | 113,103,335 |
| TRAIN | L15-1298E-1322N_5193_2903_13 | 24 | 119,414,458 |

Total new required source objects 1,293,706,541B. Existing source 1,361,753,089B plus 52,000,000B reserved for backbone/wheel/metadata gives **2,707,459,630B**, leaving 292,540,370B under a decimal 3 GB cap. Counts include matched building labels, UDM and masked imagery, not just RGB files. This is a size estimate from public object metadata, not actual acquired hashes. Additional source downloads, retries or alternative checkpoints must remain in an explicit cumulative ledger; deletion does not reset bytes spent.

Minimum TRAIN–DEV distance is 1192.18km, TRAIN–CONF 731.43km, DEV–CONF 3103.23km. First-image HTTP GeoTIFF headers were inspected for all nine candidates, without reading pixel arrays/masks: all have four uint8 channels and explicit north-up georeferencing. Their georeferenced center differs from encoded-center inference by at most 2.773m. `source-header-audit.json` contains exact keys/bounds/transforms. This validates first-image location, not every monthly grid, registration quality or eventual source usability. The new config and metadata/header hashes must be frozen **before any new label interpretation**; ingestion must still validate every acquired observation.

## Scientific scope and stopping decision

Train only on TRAIN footprints and native boundary targets using source-valid masks. Keep the physical-coverage 80/20 semantics and causal registration/site-birth policy. Freeze encoder tensors **and BatchNorm running statistics** even when the outer model is switched to training. Full-resolution RGB/early feature skips matter: tiny roofs can disappear in a stride 32 backbone. Internal 2x interpolation adds no new ground detail.

Use the enlarged DEV set for spatial checkpoint selection and object-level proposal recall/IoU with explicit false-positive burden. Compare ordinary per-frame geometry with frozen first-birth site geometry; if only the former improves, the remaining ceiling implicates the site-geometry policy. Do not hide that limit by retroactively rewriting old footprints. Do not tune thresholds on fresh confirmation. Record the new source/model/config identity, rebuild downstream features, and retrain/recalibrate gates only after spatial features are fixed; old gate weights are not automatically valid on changed features.

The outcome can still be negative. This experiment tests a specific failure hypothesis: whether stable pretrained visual features and a native-grid decoder improve usable physical proposals enough for the temporal method to matter. A future confirmation result must be reported as the new frozen study, while retaining the old failed study and all exclusions. No better metric, pretrained license badge or synthetic speed test alone establishes analyst readiness.

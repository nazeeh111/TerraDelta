# TerraDelta foundation review

Inspected 2026-10-29. Read-only source review, not a claim that these models were run. Public source checkouts are disposable at `/private/tmp/terradelta-foundations/`. No weights or imagery downloaded by this review. No paid API needed.

## Decision

A small RGB segmentation model plus object tracking and an observation-aware temporal head is feasible to test locally. Do not claim novelty for temporal persistence alone: SpaceNet7 already implements it. Compare on the same image predictions against (1) pairwise/spectral evidence, (2) ordinary causal persistence/HMM, (3) a clearly labeled adapted SpaceNet7 tracker, and (4) an actual reproduced temporal model such as WATCH's LSTM. Freeze region splits and availability assumptions first. Full winning-system reproduction and causal adaptations must have different names in results.

## SpaceNet7 solutions

Observed revision `ff1241f01e391248a9d6111a60c97973fc5a9643`, root Apache-2.0 license. The fourth-place Motoki Kimura solution supplies especially relevant, concrete prior art:

- [Written solution](https://github.com/SpaceNetChallenge/SpaceNet7_Multi-Temporal_Solutions/blob/ff1241f01e391248a9d6111a60c97973fc5a9643/4-motokimura/motokimura-solution-description.md): RGB EfficientNet-B3 U-Net, footprint/edge/contact channels, five AOI folds and two image scales, ten-model ensemble, 300 epochs per model on Tesla V100, watershed. This full setup is substantial compute; a compact locally trained segmentation network is an adaptation, not its reproduction.
- [Actual mask refinement](https://github.com/SpaceNetChallenge/SpaceNet7_Multi-Temporal_Solutions/blob/ff1241f01e391248a9d6111a60c97973fc5a9643/4-motokimura/code/tools/refine_pred_mask.py): averages predictions over configured past/future frames. The report uses the full sequence mean. This is retrospective and unsuitable as evidence of online warning ability.
- [Actual tracking and persistence](https://github.com/SpaceNetChallenge/SpaceNet7_Multi-Temporal_Solutions/blob/ff1241f01e391248a9d6111a60c97973fc5a9643/4-motokimura/code/spacenet7_model/utils/utils.py#L641): retains global master polygons, matches IoU, rejects occupied candidates and crowded-neighbor artifacts, inserts missing polygons between first and last occurrence. Important detail: `__poly_exists_consistently_in_ahead_frames` accepts a match in **at least one** future frame, not every future frame. With no future frames it returns true, so sequence-end behavior needs explicit treatment.
- [Defaults](https://github.com/SpaceNetChallenge/SpaceNet7_Multi-Temporal_Solutions/blob/ff1241f01e391248a9d6111a60c97973fc5a9643/4-motokimura/code/spacenet7_model/configs/defaults.py#L130): matching IoU 0.1, next-three-frame support IoU 0.25, no occupied overlap, radius 12 pixels, >4 neighbors rejected, gap interpolation enabled. These are upstream defaults, not automatically correct thresholds for TerraDelta.
- [Deployment](https://github.com/SpaceNetChallenge/SpaceNet7_Multi-Temporal_Solutions/blob/ff1241f01e391248a9d6111a60c97973fc5a9643/4-motokimura/code/docs/FINAL_SCORING.md) assumes NVIDIA Docker. Direct Mac reproduction is not established. [Third-party attributions](https://github.com/SpaceNetChallenge/SpaceNet7_Multi-Temporal_Solutions/blob/ff1241f01e391248a9d6111a60c97973fc5a9643/4-motokimura/code/licenses/README.md) must accompany reused code where required.

Inference: a dependency-light port of this tracker over TerraDelta's common predictions is a useful controlled temporal comparison. Label it an adapted SN7 temporal tracker and enumerate deviations. It is not the challenge-winning ensemble and does not establish removal performance.

## WATCH

Observed revision `ec6e1539673da46437c276c3ce42534982d2faff`, MIT code license. Data/checkpoints are not included. Archaeological-site/month localization differs from building object tracking.

- [TED implementation](https://github.com/microsoft/WATCH/blob/ec6e1539673da46437c276c3ce42534982d2faff/temporal_embedding_distance/infer_all_months.py) computes RMS-L2 or cosine distance to the median of previous K feature vectors. However it first uses `np.interp` over all months to fill missing observations.
- [Utilities](https://github.com/microsoft/WATCH/blob/ec6e1539673da46437c276c3ce42534982d2faff/self_supervised_change_detection/utils.py#L67) use a symmetric +/-12 window for robust scores, and sigmoid/minmax/softmax normalization depends on the full score series. Thus the maintained TED pipeline is not prefix invariant despite a past-only raw distance. Do not silently call its adapted form an exact reproduction.
- [Weakly supervised model](https://github.com/microsoft/WATCH/blob/ec6e1539673da46437c276c3ce42534982d2faff/weakly_supervised/model.py): ordinary two-layer MLP + two-layer LSTM with 128 hidden units, default unidirectional, monthly scalar head. [Training](https://github.com/microsoft/WATCH/blob/ec6e1539673da46437c276c3ce42534982d2faff/weakly_supervised/train.py#L151) explicitly falls back to CPU. This is a plausible actual reproduced architecture on common object features, with task-label adaptation documented.
- [README](https://github.com/microsoft/WATCH/blob/ec6e1539673da46437c276c3ce42534982d2faff/README.md) references `unsupervised_monthly/` runner paths, but the checkout stores these under `self_supervised_change_detection/`. Its environment is not a verified drop-in install; use a minimal reviewed dependency set and record the difference.

Inference: compare learned-head benefit to this small LSTM and causal TED using identical spatial features and masks. Neither generic feature pooling nor a bidirectional retrospective model establishes an observation-aware causal contribution.

## TerraMind

Observed IBM example revision `53d61da09213c2c3629931b6322f94cce617886a`, Apache-2.0 code. The [tiny weight card](https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-tiny) declares Apache-2.0, supports RGB, and separates generated modalities from observations. No weights were fetched or tested.

Actual [tiny registration source](https://github.com/torchgeo/terratorch/blob/main/terratorch/models/backbones/terramind/model/terramind_register.py#L909), inspected through GitHub connector (blob `7b5ec6fbaa42bbcaf33a4ac3b0924639c36c6df1`), defines a 12-block, dimension-192, three-head transformer. IBM's [tiny segmentation configuration](https://github.com/IBM/terramind/blob/53d61da09213c2c3629931b6322f94cce617886a/configs/terramind_v1_tiny_sen1floods11.yaml) freezes its backbone and uses a trainable UNet decoder. The [multitemporal configuration](https://github.com/IBM/terramind/blob/53d61da09213c2c3629931b6322f94cce617886a/configs/terramind_v1_base_multitemporal_crop.yaml) pools/concatenates timestamps for crop classification. Neither is a building event decoder. The README warns of MPS batch-normalization failures with UNetDecoder and suggests FCN or corrected TerraTorch versions.

Inference: tiny RGB frozen embeddings are a plausible optional feature baseline, but require real runtime/memory verification and appropriate Planet normalization. Generating absent SAR or land-cover inputs cannot count as observed supporting evidence. Starting with a small U-Net avoids this extra dependency surface.

## ChangeMamba

Observed revision `9ce9cec13f9ea14bc0ad91f071577ec9b3a97983`, Apache-2.0 root license. [ChangeMambaBCD.forward](https://github.com/ChenHongruixuan/ChangeMamba/blob/9ce9cec13f9ea14bc0ad91f071577ec9b3a97983/changedetection/models/ChangeMambaBCD.py) accepts exactly pre/post images and outputs a two-class change map. It supplies no persistent event identity or uncertain timing history.

[VMamba source](https://github.com/ChenHongruixuan/ChangeMamba/blob/9ce9cec13f9ea14bc0ad91f071577ec9b3a97983/changedetection/models/vmamba.py#L255) calls CUDA selective-scan functions; [extension setup](https://github.com/ChenHongruixuan/ChangeMamba/blob/9ce9cec13f9ea14bc0ad91f071577ec9b3a97983/kernels/selective_scan/setup.py) builds `.cu` sources with `CUDAExtension` and CUDA >=11.6. README states Linux-only testing. Native Apple execution is not established; a fallback rewrite would require correctness validation. Weight redistribution rights at the linked Zenodo record were not verified because that record was inaccessible to the web tool. Do not ship those weights under an assumed code license.

## Consequential gates

1. Region holdout means whole AOIs, and imagery/model pretraining contamination must be disclosed. Pretrained SN7 winner weights may have used public training AOIs selected as TerraDelta test regions.
2. A shared compact segmenter supports controlled comparison of temporal methods but not a full published-system reproduction claim.
3. Cloud gaps and monthly mosaics give interval-censored support, not exact construction dates. Missing image publication metadata prevents a live availability claim.
4. A learned quality head only earns a benefit claim if it improves held-out event metrics at a matched delay and thresholds selected on development AOIs. Preserve negative results.
5. Tiny baselines and tracker ports are plausible on CPU/MPS, but runtime, memory, training convergence and export correctness remain empirical gates for the implementation owner.

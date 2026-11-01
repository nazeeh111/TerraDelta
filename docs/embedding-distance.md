# Frozen neural embedding comparison

The new study adds a genuine neural descriptor comparison. The earlier `ted`
adaptation uses RGB and spatial scores; it is retained under that description and
does not stand in for neural embedding distance.

The descriptor uses the separate official ImageNet ResNet18 checkpoint already
declared for this study. Its encoder parameters and BatchNorm state stay frozen.
Native RGB passes through layer1, producing 64 channels at stride4 with a
43-pixel receptive field. Bilinear sampling at native pixel centers and valid-site
mean pooling add no observations or physical resolution. Source-valid pooled
centers do not guarantee that the entire surrounding receptive field is valid.
The backbone is not redistributed; [upstream terms](../DATA_LICENSE.md) remain
separate from the source-code license.

At each image-derived site's discovery, its fixed geometry may pool archived
images already available at that prefix. This matches the common spatial-feature
history and permits observed absence before a first appearance. It does not
claim the site geometry was known earlier. The shared decoder prevents all public
events before discovery. Missing prior imagery remains unknown; first observed
presence alone never invents an addition. Fixed native-index sampling chunks keep
earlier-site arithmetic unchanged when later sites appear.

Cosine distance compares the current unit descriptor with the renormalized
coordinate median of its last three usable descriptors. Distance gates updates
to ordinary spatial occupancy; the held value is a filter state, not another
observation. The ordinary decoder retains two supporting observations and a
20-day dwell. One global development operating point is selected from the
distance and decoder grids in [experiment-v8](../configs/experiment-v8.json), by
mean regional event F1 within62days, then ordinary F1, then lower distance and
decoder thresholds. No region-specific selection is allowed.

Per-region `embeddings.npz` and `embedding.json` bind descriptors, usability,
coverage, site births, pixels, source frames, geometry, feature arrays, config,
encoder and implementation hashes. Extraction reads no annotations. The loader
checks identities before calibration/evaluation opens truth. Extra encoder,
pooling, compression, storage and process peak memory are reported separately
from temporal decoding cost.

```sh
uv run --frozen --extra ml python scripts/extract_embeddings.py \
  --cache-root /path/to/cache-v3 --features-root /path/to/features \
  --config configs/experiment-v8.json --backbone /path/to/resnet18-f37072fd.pth \
  --output /path/to/embeddings-v8 --split development --device cpu
```

After all fresh confirmation image features are ready, append its disjoint
regions with `--split confirmation` and the same root/config/source. Existing
regions and receipts cannot be overwritten. Freeze all sidecars, models,
calibration, source and data identities before scoring confirmation outcomes.
The v5 freeze rechecks every frozen source, image-only input and complete sidecar
file identity before hashing any confirmation truth, including replacements
whose own manifests are internally consistent.

The v8 comparison amendment retains the actual v7 head-training config identity.
Both head trainers ignore the embedding comparator. The resolver checks the
referenced sibling file's exact SHA and compares the entire remaining config,
including numeric types, after removing only the explicitly listed comparator
and version metadata. Splits, data policy, spatial/candidate binding, truth,
resources, optimizer/selection settings and unknown fields cannot change. Actual
checkpoint headers and fit-input receipts retain v7; study and calibration retain
v8 plus that explicit training identity. No checkpoint metadata is relabeled.

The first descriptor implementation masked all prebirth history and consequently
missed newborn additions. Independent synthetic review caught this before any
real descriptor extraction or performance selection. Its original plan and
evidence were preserved. The corrected version passes newborn, missing-history,
initial-inventory and exact-prefix controls with an actual frozen encoder. These
checks establish behavior, not useful event accuracy. Development selection and
fresh geographic confirmation remain required.

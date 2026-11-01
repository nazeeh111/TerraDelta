# Event transition gate

This alternative learns whether an ordinary persistent spatial proposal is compatible with a physical building-change annotation. It does not learn another occupancy sequence. A useful outcome requires event/object precision, recall and false-alert burden at a matched detection horizon on geographically held-out data. A trained classifier or better development pixel score is insufficient.

## Shared candidate policy and causal evidence

All three methods call the same image-derived site inventory, birth gating and ordinary raw decoder: occupancy threshold0.55, two supporting usable observations, minimum20-day dwell, with the decoder's common gap/revision limits. `proposal_tracking` returns that ordinary history unchanged. `observation_gate` and `gate_no_quality` filter its eligible confirmations. The model never supplies candidate geometry or site identities.

The78 input values are current17features, mean of the last≤3usable pre-change observations ending at the last stable support, mean/std of the latest≤3usable observations since first opposite support, and10direction/dwell/count/margin/contrast/area values. Source dates, frozen image-derived geometry and current usable-prefix features are allowed. Ground-truth overlap, event matching, later survival/retraction and future windows are absent from inference.

Quality inputs are explicit: source-valid fraction, registration displacement/usability, cadence, seasonal phase/reference and color-change context. The paired ablation zeroes columns3,4,5,6,7,12,13,15,16 in each17feature block after train-only standardization. It retains the identical hard usability mask and ordinary persistence/dwell inputs. Both MLPs have the same78→64→32→1architecture, initialization, actual example indices and training budget.

A first opposite observation creates the same proposed event ID for every method. Only after ordinary persistence eligibility can the gate confirm it. A low score keeps a proposed revision; each later usable observation may update that proposal and eventually confirm the same ID. Gate confirmation stays until an observed raw reversal retracts it, avoiding repeated alerts due to score jitter. Consecutive revisions preserve all prior answers. A raw correction retracts even a never-confirmed proposal. Full usable-prefix source identities cover seasonal feature dependencies; missing observations are never counted as absence.

## Labels, class balance and calibration

Training is blocked until the configuration declares reviewed physical truth ready and pins both truth and supervision source hashes. Only the configured TRAIN and DEVELOPMENT regions are opened. The historical v4 study used five TRAIN and two DEVELOPMENT regions; later studies declare their own frozen geographic roles. No confirmation array, label or metric is read.

Every raw first-confirmed proposal is assigned through the evaluator's maximum-cardinality, then maximum-IoU one-to-one matching against physical ground-truth events (IoU≥0.25, first-support tolerance93days). Later retractions do not remove or redefine earlier training examples. Each proposal contributes its first eligible prefix and up to three later usable prefixes before a raw correction. Later rows inherit physical-event compatibility; labels are not invented exact event dates or timely-alert labels. Matched detection-delay evaluation remains separate.

Train-only mean/std uses each proposal's total unit mass. Real positive and negative TRAIN rows are sampled in equal128+128minibatches; sampling within each class gives every proposal equal mass. This is explicit class balance, not fabricated examples. Training uses the final checkpoint of a fixed bounded budget, avoiding occupancy-loss checkpoint selection.

Each method then fits one global positive-slope Platt transform on actual DEVELOPMENT proposals. Rows from a proposal sum to unit weight, preserving actual proposal prevalence despite repeated prefixes. A one-class development set yields an explicitly labeled constant-prevalence transform because a discrimination slope is not identifiable. The score means development-calibrated compatibility under this matching protocol, not legal evidence, arbitrary-shift calibration or a timing-coverage guarantee. Threshold and time padding are selected/frozen by the main study, then evaluated with ordinary and matched-delay baselines.

## API and artifacts

`event_gate.generate_events(x, usable, births, info, weights, kind='observation_gate', calibration={}, record_origin=None)` returns `EventRevision` history. Kinds: `observation_gate`, `gate_no_quality`, `proposal_tracking`. `weights` may be a checkpoint path or `load_gate(path)` result; the raw baseline acceptsNone. Optional calibration is `{kind: {scale, bias, threshold, timing_padding_days}}`; omitted learned parameters use the checkpoint's development Platt fit, threshold0.5 and padding0. Raw candidate thresholds are never changed through this dictionary.

The raw decoder must provide1-based `site_id` and0-based `support_before_index`, `support_first_index`, `support_current_index` quality fields. Feature order and frozen spatial provenance are enforced. Inputs/outputs remain native-grid, source-linked histories.

## Explicit spatial provenance for a new study

New training configurations require `event_gate.spatial_provenance` with exactly
these keys:

```json
{
  "spatial_weights_sha256": "<audited complete lowercase checkpoint SHA256>",
  "spatial_backbone_sha256": "<audited complete lowercase backbone SHA256>",
  "candidate_threshold": 0.55,
  "spatial_inference": {
    "tile_size": 128,
    "context_halo_native_pixels": 64,
    "native_alignment": 16,
    "internal_input_scale": 2,
    "implementation_sha256": {
      "spatial.py": "<complete source SHA256>",
      "pretrained_spatial.py": "<complete source SHA256>"
    }
  }
}
```

This is a schema example, not an approved checkpoint identity or a launch-ready
configuration. Use values from the audited prediction protocol and selected
artifact after their bytes are stable. `spatial_backbone_sha256` may be explicit
JSON `null` only for a model with no separate backbone; omission is an error.
When the study also declares `spatial_backbone.sha256`, the two identities must
agree. The inference dictionary must include all five displayed fields; grid
parameters are integers and implementation identities are complete hashes.
The implementation set is exactly `spatial.py` plus `pretrained_spatial.py` for
a separate encoder, and only `spatial.py` for an ordinary U-Net. Candidate
threshold is also bound because it changes geometry and proposal prevalence.

Before any feature arrays, proposal generation or physical labels are read, the
trainer checks **all** TRAIN and DEVELOPMENT `sites.json` files against this
binding, feature/supervision protocols, region identity and source metadata.
A later DEV mismatch therefore cannot occur after earlier TRAIN labels were
already opened. Missing, mixed or unequal model, backbone, tiling, internal
scale, alignment, context or implementation hashes fail closed. Confirmation
manifests are not part of this preflight and are not opened by the trainer.

New gate checkpoints store the same `spatial_provenance` and retain the existing
top-level `spatial_weights_sha256`; disagreement is rejected at load. Learned
inference checks the complete feature binding **before** decoding raw proposals.
This protects tile-context-dependent spatial features from accidental mixing.
The pure helpers `validate_spatial_provenance(binding)` and
`assert_spatial_provenance(info, binding)` are shared with other preflight/freeze
callers; they perform no file access.

Backward compatibility is limited and explicit. Only the exact historical
configuration version `sn7-regions-v4-event-gate`, with no new backbone
declaration, may omit the new binding. It retains the original hardcoded
spatial-v2 SHA. Historical feature records may omit the newly forwarded optional
backbone/inference fields or contain JSON `null`. Current inference also accepts
the independently checked spatial-v2 replay grid: exact original checkpoint,
no backbone, tile1024/halo64/alignment8/internal2/candidate0.45 and the fixed
reviewed `spatial.py` implementation SHA in `assert_legacy_spatial_compatibility`.
Other non-null provenance is rejected. This exception does not change the
historical study's source identity or erase new metadata. Original frozen
retraining/reproduction uses commit `bd33082`; a current reconstruction has a
new processing identity. Raw `proposal_tracking`
does not use learned weights. The provenance revision changes no examples,
matching rule, optimizer, loss, training budget, Platt fit or checkpoint selection.

```sh
python scripts/train_event_gate.py \
  --features-root /path/to/new-study/features \
  --cache-root /path/to/new-study/cache \
  --config configs/experiment-v6.json \
  --output work/retrained/event-gate.pt
```

The trainer exclusively creates `.pt`, `.json` and `.examples.npz` files; existing outputs are rejected. Checkpoints record actual source/data/config/model hashes, spatial provenance, label protocol, per-region proposal audits, paired training traces and prevalence calibration. The examples file preserves the actual78-dimensional rows, labels, proposal weights, event IDs, region IDs and prefix indices used. Model/data versions and earlier failed studies remain distinct.

Unit fixtures exercise causality, source masks, proposal retention, later confirmation, correction history, ablation and calibration weighting. These fixtures are not training data or performance evidence.

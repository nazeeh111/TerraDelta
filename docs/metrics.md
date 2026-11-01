# What the evaluation measures

The unit of evaluation is a persistent, label-supported building-footprint
visibility transition in a complete geographic region. The current study uses
`configs/experiment-v4.json`; it preserves the five training, two development and
three confirmation regions originally fixed in experiment-v1. No threshold is
fitted separately for a confirmation region. Adjacent crops are not independent
regions. This guide describes inspected source behavior, not measured results.

## Physical-coverage truth and its audit

`ground_truth_events(cache)` defaults to `physical_coverage_v2`. It reads the
native-grid `instances.npy`, source-valid `valid.npy` and acquisition metadata.
Each matched source ID's first rasterized footprint supplies an evaluation
reference geometry. Those future label identities/geometries never create model
candidates or inference features. Subpixel labels without any rasterized pixel
are outside this native-grid evaluation. SpaceNet's matched IDs and footprints
are postprocessed annotation products, not exact physical event-date observations.

For each reference footprint and month:

1. At least 80% of the full reference footprint must be source-valid. Otherwise
   its state is unknown.
2. Among that source-valid portion, **any building annotation** covering at least
   80% means present. Coverage at most 20% means absent. Intermediate coverage is
   unknown and cannot update the label state.
3. Two usable opposite-state observations establish a persistent transition.
   Masked or ambiguous observations neither count as absence nor supply a second
   support. A single opposite-state flicker followed by the stable state is not
   an event.

Presence is geometric building coverage, not continued membership of one source
ID. An ID reassignment alone therefore cannot establish demolition or a new
building. Initial usable inventory is baseline, not an addition. A building
present initially may later have a supported removal. Source-valid truth events
remain eligible when the model rejects imagery for registration quality; that
model abstention can lower recall.

Overlapping source references can describe the same physical transition.
Deduplication groups only events of the same class whose geometries have
intersection-over-union (IoU) at least 0.5 **against every group member**, with
first-support span at most 62 days. It does not grow arbitrary overlap/time
chains. The first deterministic representative supplies the scored geometry;
`reference_truth_ids` and `reference_members` retain every member's complete
identity, geometry and timing. No synthetic union replaces those references.

`ground_truth_audit(cache)` retains the optional `matched_id_v1` identity-based
comparison alongside the physical protocol. It is an audit, not the primary
performance target. The audit includes ID/coverage conflicts, duplicate merges,
legacy and physical events, and observation/exclusion counts for **all**
references, including references with zero events. Counts distinguish source
unusable frames, ambiguous coverage, initial inventory/absence, no usable state,
and pending changes with insufficient later support. An empty label raster has
an explicit zero-reference audit. These cases must stay visible instead of
vanishing behind a count of successful events.

The label interval begins at the **start** of the last confidently opposite-state
month and ends at the **end** of the first supporting-state month. The second
support establishes persistence without moving that first month. Cloud gaps
widen support. No precise physical event date is annotated inside the interval.

## One-to-one event and footprint comparisons

Each event ID contributes its first `confirmed_by_model` or `reviewed` revision
to ever-confirmed alert evaluation. Multiple revisions of the same ID are not
multiple predictions. A prediction and label are eligible to match when:

1. Their supported change classes agree.
2. Footprint IoU is at least 0.25, using WGS84 geodesic polygon areas. This relaxed
   native-resolution threshold is explicit, not substituted for stricter accuracy.
3. First-support month-end proxies differ by at most 93 days.

Predicted first support comes from
`quality.unexpanded_observation_interval[1]`, before timing padding. Expanded
intervals cannot manufacture match eligibility. Sparse one-to-one assignment
first maximizes the number of matches, then total footprint IoU. Returned
`match_events` indices refer to the original revision list and original truth
list. Precision is matched/predicted events, recall is matched/label events, and
F1 is their harmonic mean. Empty denominators are `null`, not invented perfect
or zero scores. Zero-event regions still report their predictions, exposure and
false-alert burden.

`spatial_object` repeats class/IoU assignment without the 93-day timing limit.
`matched_footprint_iou` summarizes geometry among temporal matches. Candidate
coverage reports how many truth events overlap any image-derived site at
IoU >= 0.25, independently of the temporal model. It exposes the spatial proposal
recall ceiling and candidate count/area rather than presenting conditional gate
accuracy as end-to-end coverage.

An unmatched confirmation eligible for an already matched truth event counts as
a duplicate and remains a false positive. Retraction cannot erase an earlier
confirmed alert from precision or false-alert burden.
`confirmed_then_retracted_event_ids` retains that correction cost.
`final_retained` evaluates the latest confirmed/reviewed hypotheses; retracting
a true event can lower final recall. Proposed-only hypotheses are reported as a
separate review burden and are not confirmed alerts.

## Twelve controlled methods

All methods share native imagery, frozen spatial-v2 probabilities and causal
image-derived sites. The comparison is controlled adaptation, not replication of
published full-system scores.

| Method key | Role and availability |
| --- | --- |
| `pairwise` | Immediate spatial occupancy change, with one support |
| `spatial_tracking` | Spatial occupancy with ordinary persistent tracking |
| `smoothing` | Causal temporal smoothing with the shared decoder |
| `hmm` | Causal two-state hidden Markov filtering |
| `ted` | Causal temporal embedding distance on shared spatial/RGB features |
| `sn7_retrospective` | SN7-inspired mean/persistence adaptation using future evidence |
| `observation` | Learned observation-conditioned GRU occupancy model |
| `no_quality` | Paired GRU quality/context-covariate ablation |
| `watch_lstm` | WATCH's unidirectional LSTM architecture adapted to these features/labels |
| `proposal_tracking` | Shared raw event proposals without learned compatibility filtering |
| `observation_gate` | Learned available-prefix event-compatibility gate |
| `gate_no_quality` | Paired event-gate quality/context-covariate ablation |

The last three share the same raw proposal policy, candidate identities and
geometry. Their first proposal uses occupancy threshold 0.55; confirmation
eligibility requires two usable supports and at least 20 days of dwell.
Gate calibration does not alter that raw policy. The no-quality gate keeps the
same hard usability mask, proposal set, examples and training budget. See
[event-gate.md](event-gate.md) for exact inputs and revision behavior.

The retrospective SN7 method includes full-sequence availability in detection
time. It cannot claim online latency or prefix invariance. It is not the original
ten-model EfficientNet-B3 ensemble. WATCH is an architecture/task adaptation,
not its archaeological-site study. No unmodified CUDA-dependent method is claimed.

## Timing, availability and detection delay

Raw and public expanded intervals are tested for containing the **entire label
interval**. Raw overlap and raw/expanded widths are also reported. These are
support-interval diagnostics, not confidence coverage for an unknown physical
construction date. Absolute interval-midpoint difference is explicitly named an
interval proxy error. Timing summaries are conditional on matched events, so
recall must accompany them and misses cannot disappear from the assessment.

Detection delay is first-confirmation `detected_at` minus the label's first-support
month end. Original publication timestamps are unavailable, so this is a
retrospective month-end-availability experiment, not measured live-warning
latency. Negative delays are retained as early alerts. Bounds relative to the
entire label interval are reported separately. Full-sequence methods cannot
claim detection before their evidence became available.

`within_62_days` repeats one-to-one assignment with maximum detection delay of
62 days. Its precision denominator includes all ever-confirmed predictions; its
recall denominator includes all supported label events. Late alerts do not earn
timely detections. Ordinary matched late counts and measured delay distributions
remain visible. A common horizon supports comparison but does not make actual
delay distributions identical. Global thresholds and timing padding are selected
only on development regions and frozen before confirmation.

## Occupancy and event-compatibility calibration are different

Occupancy calibration estimates building presence on common sites. Its target
uses the physical 80/20/unknown coverage rules and actual source usability. That
reliability assessment is not event correctness.

The event gate instead learns compatibility of a raw first-confirmed proposal
with a physical event under the one-to-one class/IoU/time matching protocol.
Labels include subsequently retracted proposals; later survival/retraction
cannot erase or relabel an earlier example. The first eligible prefix plus up
to three later usable prefixes before observed raw retraction provide at most
four score rows per proposal. Every raw proposal has **unit total mass** across
its available prefix rows. Longer histories therefore do not dominate fitting
or reported calibration.

Training balances actual positive and negative examples. One global positive-slope
Platt transform is then fitted on the actual development proposal prevalence,
using those unit-mass weights. No region-specific fitting occurs. Compatibility
is not a timely-alert label, so detection-delay evaluation remains separate.
A one-class development set uses an explicitly labeled constant-prevalence
transform instead of claiming an identifiable discrimination slope.

`scripts/evaluate_gate_calibration.py` performs **no fitting**. After the full
freeze guard, it scores every held-out raw eligible proposal with the frozen gate
and development transform, using the same at-most-four-prefix/unit-mass policy.
It reports row counts, proposal mass, positive prevalence, Brier error, reliability
bins and expected calibration error separately by region. Brier is mean squared
score error. The constant-prevalence oracle Brier is a diagnostic using held-out
prevalence, not a deployable fitted baseline. Empty sets have no Brier/prevalence;
an empty accumulated calibration error cannot demonstrate calibration. Always
read denominators alongside error summaries.

These scores condition on the common proposal distribution. They cannot measure
objects omitted upstream or guarantee calibration under arbitrary geographic
shift. Event precision/recall, candidate coverage, false-alert burden and timing
support remain necessary alongside score reliability.

## Area-time burden, costs and geographic uncertainty

AOI area uses the WGS84 ellipsoid and transformed native-grid boundary. Primary
observed area-time is that area multiplied by each month's source-valid pixel
fraction and acquisition duration, summed in km²-years. The pixel fraction
approximates valid-area fraction within a compact AOI; the denominator is not
Web Mercator square metres. A second denominator also applies model registration
usability. The primary denominator retains all source-valid observations.
Neither treats masked months as observed no change.

False confirmed alerts per source-valid km²-year include all unmatched
ever-confirmed events, duplicates and later retractions. Proposed-only candidates
are separate review work. Runtime for scoring/decoding/evaluation is separate
from recorded spatial prediction time; both must be reported.

`analyst_cost_sensitivity` illustrates possible tradeoffs in **dimensionless work
units**, not currency, measured analyst time or user preferences. It combines
false-alert weights of 1, 10 or 100, miss weight 10, positive matched detection
delay weight 1 per building-month, and weight 5 per confirmed-then-retracted ID.
The same false alert can incur a later correction cost. These assumed weights
are a sensitivity analysis, not a demonstrated business-value ranking.

Pooled precision/recall aggregate counts across complete regions. Bootstrap
intervals resample entire geographic regions, never pixels or adjacent tiles.
With three confirmation regions, percentile intervals are unstable, can be
degenerate and do not establish broad generalization. Undefined resamples are
excluded per metric and their defined count is disclosed. Complete ordinary and
difficult regions, exclusions and per-region reports must accompany pooled scores.

## Full confirmation freeze

The published first physical study uses `terradelta-full-freeze-v3` in
`artifacts/freeze-v2.json`, reproduced from commit `bd33082`. Format and
study-artifact version are distinct. The newer source uses v4 for legacy-input
studies and v5 when the neural descriptor comparison is declared. V5 also binds
the complete descriptor sidecars and explicit actual head-training config.
The freeze binds actual spatial, temporal and gate checkpoint hashes, calibration,
configuration, confirmation-region identities and consequential source files.
Per-region bindings include `features.npz`, `sites.json`, the `sites.npy` raster,
semantic acquisition/quality metadata, RGB and source-valid arrays, and truth
instance arrays. Probability-product hashes and source identities are bound
through recorded site provenance. Host-local image paths are excluded from the
semantic metadata hash; scientific metadata is not.

Consequential source bindings include acquisition, ingestion, models, features,
decoders/gates, schema/store/service/CLI, evaluation, calibration/training,
stress/invariance/raster scripts, this calibration evaluator, `pyproject.toml`
and `uv.lock`. Freeze creation hashes prepared truth bytes without interpreting
outcomes. Confirmation evaluation reconstructs and validates the boundary before
interpreting its truth. A mutation invalidates the freeze. New results use a new
declared version; the comparison never fits from confirmation outcomes.

The first study's controlled-comparison commands use spatial-v2, temporal-v2, event-gate-v1,
calibration-v2 and experiment-v4:

```sh
uv run python scripts/evaluate.py --features-root work/study/features \
  --cache-root work/cache --weights artifacts/temporal-v2.pt \
  --spatial-weights artifacts/spatial-v2.pt --event-gate-weights artifacts/event-gate-v1.pt \
  --calibration artifacts/calibration-v2.json --config configs/experiment-v4.json \
  --split confirmation --freeze-manifest artifacts/freeze-v2.json \
  --output work/confirmation-v2
uv run python scripts/evaluate_gate_calibration.py --features-root work/study/features \
  --cache-root work/cache --weights artifacts/temporal-v2.pt \
  --spatial-weights artifacts/spatial-v2.pt --event-gate-weights artifacts/event-gate-v1.pt \
  --calibration artifacts/calibration-v2.json --config configs/experiment-v4.json \
  --freeze-manifest artifacts/freeze-v2.json --output work/gate-calibration-v2.json
```

Paths must identify the exact prepared data bound by that manifest. Regenerated
products require their own local freeze as described in [reproduce.md](reproduce.md).
Use fresh output paths. The [frozen confirmation result](results-v2.md) is now
available and negative. This guide defines the inspected source protocol; the
linked result provides measurements, executed verification and remaining limits.

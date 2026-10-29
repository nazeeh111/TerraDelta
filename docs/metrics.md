# What the evaluation measures

The unit of evaluation is a persistent building-visibility transition in a
complete geographic region. Training, development and confirmation regions are
fixed in `configs/experiment-v1.json`. No threshold is fitted separately for a
confirmation region. Adjacent crops are not treated as independent regions.

## Label-supported events

`ground_truth_events(cache)` reads the native-grid `instances.npy` matched source
IDs, `valid.npy` source masks and monthly acquisition metadata. It uses each ID's
first rasterized footprint as an evaluation reference. Those future label
identities and reference geometries never create model candidates or features.
Subpixel labels without any rasterized pixel are outside this native-grid
evaluation. Source IDs and footprints are postprocessed SpaceNet matched labels,
not independent measurements of exact physical construction dates.

A reference site is observable in a month only when at least 80% of its reference
footprint is source-valid. Two usable observations of an opposite state establish
a transition. A masked observation is unknown, neither an absence nor a second
support. One usable opposite-state flicker followed by the original state does
not establish an event. First usable inventory is baseline, so buildings already
present at the beginning are not newly detected additions. Existing inventory
can later have a supported removal. Source-valid events remain in the truth set
when the model rejects an image for registration quality; that rejection is an
abstention that can lower recall.

The label interval starts at the **start** of the last confidently opposite-state
month and ends at the **end** of the first supporting-state month. The second
support establishes persistence but does not move the first supporting month.
Cloud gaps widen this support. There is no annotation of a precise physical
event date inside the interval. Label intervals are therefore not ground truth
for an exact construction or demolition date.

## One-to-one event and footprint comparisons

Each event ID contributes its first `confirmed_by_model` or `reviewed` revision
to the ever-confirmed alert evaluation. Multiple revisions of that ID do not
become multiple predictions. Predicted/label events are eligible to match when:

1. Their supported change classes agree.
2. Their footprint intersection-over-union is at least 0.25, measured with WGS84
   geodesic polygon areas. This relaxed footprint threshold reflects the native
   resolution and is stated explicitly, not substituted for stricter accuracy.
3. Their first support month-end proxies differ by at most 93 days.

The predicted first support comes from
`quality.unexpanded_observation_interval[1]`, before timing-calibration padding.
Expanded intervals cannot manufacture match eligibility. Sparse bipartite
assignment first maximizes matched event count, then total footprint IoU. Each
prediction and label can be matched once. Precision is matched/predicted events;
recall is matched/label events; F1 is their harmonic mean. Empty denominators are
`null`, not invented perfect or zero performance. Returned `match_events` indices
refer to the original event-revision list and the original truth list.

`spatial_object` applies the same class/IoU assignment without the 93-day timing
limit. It exposes footprint/object detection separately from event timing.
`matched_footprint_iou` reports IoU among temporal matches. Candidate coverage
reports the fraction of truth events covered by any image-derived candidate at
IoU >=0.25, independently of the temporal head. This exposes the upstream recall
ceiling and candidate count/area rather than presenting conditional head accuracy
as end-to-end coverage.

An unmatched confirmation eligible for a truth event already matched to another
prediction counts as a duplicate. It remains a false positive. A retraction does
not erase the alert from ever-confirmed precision or false-alert burden.
`confirmed_then_retracted_event_ids` records that cost separately.
`final_retained` evaluates the latest confirmed/reviewed hypotheses after all
revisions, so retracting a real supported event can lower final recall. Proposed
candidates are reported but do not count as confirmed alerts.

## Timing, availability and delay

The public/expanded and raw intervals are evaluated for containing the **entire
label interval**. Raw interval overlap and raw/expanded widths are also reported.
These are support-interval diagnostics, not confidence coverage for an unknown
physical event date. The absolute difference of interval midpoints is explicitly
named an interval proxy error. It does not establish exact event-time accuracy.
Timing metrics are conditional on matched events; report event recall alongside
them so misses cannot disappear from the assessment.

Detection delay uses the first confirmation's `detected_at` minus the label's
first-support month end. Source publication times are unavailable, so this is a
retrospective month-end-availability experiment, not a measured live-warning
latency. Negative delays are retained and counted as early alerts, because
uncertain monthly timing can produce them. Bounds relative to the full label
interval are reported separately. A method inspecting the full sequence has its
actual full-sequence evidence availability and cannot claim an earlier online
detection.

`within_62_days` repeats the one-to-one assignment with a maximum 62-day delay.
Its precision denominator includes all ever-confirmed predictions and its recall
denominator includes all supported label events. Late alerts therefore do not
earn timely detections. The ordinary matched-event late count is also reported.
This common horizon allows a fixed-delay comparison; it does not imply that
methods have identical measured delay distributions. Threshold selection belongs
only to the development split and must be frozen before confirmation.

## Area-time burden and geographic uncertainty

The AOI area is measured on the WGS84 ellipsoid from its transformed native grid
boundary. Observed area-time is that geodesic area multiplied by each month's
source-valid pixel fraction and acquisition duration, summed in square-kilometre
years. The pixel fraction approximates valid-area fraction within a compact AOI;
the denominator is not Web Mercator square metres. A second denominator applies
registration usability, while the primary denominator retains all source-valid
observations. Neither counts masked months as observed no change.

False confirmed alerts per source-valid km²-year use all unmatched
ever-confirmed events, including duplicates and later retractions. Proposed-only
candidates are a separate review burden. Runtime includes temporal scoring,
decoding and evaluation; spatial prediction runtime must be reported separately
from its recorded prediction artifact.

Aggregated precision/recall pool counts across complete regions. Bootstrap
intervals resample complete geographic regions with replacement, never pixels
or adjacent tiles. With only three confirmation regions, percentile intervals
are unstable, can be degenerate, and do not establish broad geographic
generalization. Undefined metric resamples are excluded from that metric's
percentile and their defined count is disclosed. Ordinary and difficult full
regions, exclusions and per-region results must accompany pooled scores.

## Confirmation guard and reproducible comparison

`scripts/evaluate.py` reads confirmation features and labels only after validating
an explicitly supplied freeze manifest containing `frozen_at`,
`temporal_weights_sha256`, `calibration_sha256` and `config_sha256`. Preserve
spatial weights, source hashes, data manifests and processing revisions alongside
those artifacts. The comparison never fits calibration or thresholds.
Calibration supports `occupancy[method]`, `decoders[method]`, and a fallback
`decoder`; settings are recorded per method and region.

```bash
python scripts/evaluate.py --features-root work/features --cache-root work/cache \
  --weights artifacts/temporal-v1.pt --calibration artifacts/calibration-v1.json \
  --config configs/experiment-v1.json --split development --output work/dev-comparison
```

Use a new output directory for each run. Confirmation additionally requires
`--split confirmation --freeze-manifest artifacts/freeze-v1.json`. The shared
candidate/spatial comparison is a controlled local adaptation comparison. It
does not reproduce the published winner's full ensemble, WATCH's original task,
or an unmodified CUDA-dependent temporal network.

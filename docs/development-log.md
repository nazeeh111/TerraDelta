# Development decisions before confirmation scoring

This record preserves unsuccessful development results and the reasons for later
versions. The later [frozen confirmation result](results-v2.md) failed geographic
transfer. Training completion, lower development loss and interface polish remain
separate from a useful held-out event detector.

## Fixed data and the initial study

Five training, two development and three confirmation regions were fixed before
label inspection. The download contains 714 source objects, 238 monthly mosaics
and associated matched labels/UDMs, about 1.36 GB. Complete regions and sequences,
not adjacent crops, define the split. The minimum training-to-confirmation centre
distance is about 1,489 km.

The first 482,050-parameter spatial model trained 3,000 steps in 1,606.84 seconds
on Apple MPS. Step 1,900 was selected on fixed development crops: footprint Dice
0.4220 and boundary Dice 0.3088. The 4,962-parameter observation-conditioned GRU
and its quality-covariate ablation trained 1,000 steps each. The adapted WATCH
LSTM has 291,329 parameters.

Development event results were poor. At the selected observation-head setting,
the two development regions had 3/72 and 8/315 correct first-confirmed event
alerts respectively, with only 2 and 4 inside the 62-day horizon. The quality head
was nearly identical to ordinary segmentation/tracking. The adapted WATCH LSTM
had better ordinary event F1 but longer delay. These are failed development
results under the original protocol, not evidence of a successful new method.

The version-1 models, calibration, source hashes and traces remain in `artifacts/`.
Its confirmation freeze was created, but confirmation scoring was deferred.
That old freeze is not valid for changed source or the later truth protocol.
The original figures therefore must not be directly compared with revised
physical-truth event counts as if only the model changed.

## Spatial-v2 revision

An inspected SpaceNet7 technique suggested internal enlargement for small
footprints. Spatial-v2 warm-started from spatial-v1 and used 2× internal
interpolation, 64 native-pixel training crops and the same fixed 128 native-pixel
development crops. Source and export grids retained their original resolution.
Interpolation created no additional observations.

The bounded run completed 5,000 additional steps in 913.49 seconds on Apple MPS.
Selected step 4,500 had fixed-crop footprint Dice 0.4395237 and boundary Dice
0.3567405. Its checkpoint SHA-256 is
`7ab95679a2009918dbab62fca64244476df519080cb8c0d376c00d1494983886`.
Warm start, scale and extra optimization changed together, so this is a
development revision, not a clean ablation of interpolation alone. A modest
development pixel improvement does not establish improved event performance.

## Physical supervision and source-quality repairs

The revised physical-coverage protocol examines any-building coverage inside
reference footprints: at least 80% means present, at most 20% absent, and the
middle band unknown. At least 80% must be source-valid. Two usable supports
establish a persistent transition. Matched ID reassignment alone cannot create
a demolition. Bounded all-member duplicate grouping retains its audit. The two
development regions have 124 and 194 supported reference events under this
protocol, rather than the earlier ID-presence target.

Source correctness repairs preceded confirmation scoring: calendar-aware seasonal
and elapsed-time features, explicit rejection of unsupported publication ordering,
anchor/accepted-reference registration checks that bound cumulative drift,
required RGB/alpha semantics and source-linked local imports without invented
S3 provenance. GeoTIFF products use shared internal validity masks so original
grids and masks survive independent readback. Earlier products and failures are
retained separately.

Temporal-v2 trained each occupancy model for a bounded 2,000 steps with
development selection by 62-day event F1. The observation-conditioned occupancy
model remained weak. Its WATCH architecture adaptation was more promising in
development but did not establish a generalizable operating point.

## Event-gate-v1 alternative

The alternative directly learns compatibility of an already persistent raw
proposal with a supported physical event. It shares raw geometry, birth gating,
occupancy threshold 0.55, two supports and a 20-day minimum dwell with ordinary
`proposal_tracking`. It never creates geometry from labels. `observation_gate`
and `gate_no_quality` differ only in the declared quality/context covariates;
both retain actual hard usability masks and the same raw proposal lifecycle.

Each MLP has 78 inputs and a 64/32 hidden architecture. The paired models trained
2,000 steps each on actual labeled prefixes, with equal positive/negative
minibatches and per-proposal weighting. The complete paired run took 41.95 seconds
on two CPU threads. Its training set contains 19,891 eligible raw proposals,
including 249 physical matches; development contains 5,505 proposals with
99 matches. The saved examples retain 66,351 training and 18,077 development
prefix rows, whose per-proposal weights prevent longer histories from dominating.

The model uses available-prefix evidence, not later retraction/survival or future
windows. Later eligible prefixes can confirm a retained proposal; a raw correction
retracts the same identity. All prior revisions remain in the ledger.

One global development Platt transform adjusts scores to actual proposal
prevalence. The development Brier scores are 0.0171253 for the quality-input gate
and 0.0168013 for its no-quality-input ablation. Brier is mean squared score error;
lower is better. These calibration diagnostics favor the ablation and do not
support a quality-input benefit. They are not event precision/recall, a
timing-coverage guarantee or evidence of calibration under geographic shift.

## Frozen comparison and outcome

Global development threshold and timing-padding selection precedes the final
freeze and untouched confirmation evaluation. The comparison includes twelve
methods: pairwise, spatial tracking, smoothing, HMM, causal temporal embedding
distance, retrospective SN7 adaptation, observation GRU, its ablation, WATCH
LSTM adaptation, raw proposal tracking, observation event gate and the paired
gate ablation. Online and retrospective availability remain distinct.

After development, version-2 calibration and the full freeze preceded untouched
three-region confirmation scoring. The [result](results-v2.md) is negative, with
poor transfer, severe missed-event counts and no overall quality-input benefit.
The complete record includes matched-delay results, timing, burden, stress and
independent wheel/API checks. Public release and remote verification remain
pending. A further scientific revision requires fresh geographic confirmation
data; no successful future model is promised.

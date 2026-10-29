# Development decisions before confirmation scoring

## Initial bounded experiment

Five training, two development and three confirmation regions were fixed before
label inspection. The download contains 714 source objects, 238 monthly mosaics
and associated matched labels/UDMs, about 1.36 GB. No confirmation performance was
computed during this development stage.

The first spatial model trained 3,000 steps. Step 1,900 was selected on fixed
development crops (footprint Dice 0.4220; boundary 0.3088). The 4,962-parameter
observation-conditioned temporal head and its quality-covariate ablation trained
1,000 steps each. The adapted WATCH LSTM has 291,329 parameters.

Development event results were poor. At the selected observation-head setting,
the two development regions had 3/72 and 8/315 correct first-confirmed event
alerts respectively, with only 2 and 4 inside the 62-day horizon. The quality head
was nearly identical to ordinary segmentation/tracking. The adapted WATCH LSTM
had better ordinary event F1 but longer detection delay. These are failed
development results, not evidence of a successful new ML method.

The initial models, calibration, source hashes and training traces are preserved
as version 1. Its confirmation freeze was created, but confirmation scoring was
deliberately deferred. Any source/model revision invalidates that freeze and
requires a new version before confirmation evaluation.

## Bounded spatial revision

The next development revision follows an inspected SpaceNet7 technique for small
footprints: enlarge images internally for the network, then return probabilities
to the original grid. It uses 2x internal interpolation, 64 native-pixel training
crops, the same fixed 128 native-pixel development crops, a warm start from the
first model and at most 5,000 additional steps. Source imagery and exported grids
retain their original resolution; interpolation creates no new observations.

Warm start, changed training scale and extra steps make this a development
revision, not a clean ablation identifying the effect of upsampling alone. Regions
and confirmation data remain fixed. The old result will not be replaced by an
unrecorded favorable run, and failure will remain explicit if the revision fails.

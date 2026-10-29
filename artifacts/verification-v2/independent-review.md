# TerraDelta independent event-gate review v5

2026-10-29. **Ready for a final experiment freeze: no open material correctness defect was found within this review scope after the verified repairs below.** This approves the reviewed gate/data/calibration contracts for the next frozen evaluation step. It is not evidence that the learned method improves performance, that scores generalize across arbitrary regions, or that release requirements have been met.

Read-only scope: `event_gate.py`, `train_event_gate.py`, config4, gate documentation, the actual trained checkpoint/examples/metadata, the added held-out calibration evaluator, relevant integration/protocol source and tests. Actual artifact access was restricted to TRAIN/DEVELOPMENT rows and the trained weights. No confirmation imagery, labels, examples or performance were opened or computed. No source, model or experiment inputs were changed by this reviewer.

## Findings and verified repairs

**P2, repaired: checkpoint/runtime policy compatibility.** The initial loader recorded but ignored the checkpoint's `raw_decoder` and quality-drop contract. With the actual checkpoint trained at threshold 0.55, changing the running raw threshold to 0.95 in an isolated Python patch still loaded the model under the same checkpoint revision. `event_gate.py:151-154` now rejects stored raw-decoder or ablation-drop mismatches before model-state loading. The independent reproduction now raises the intended `ValueError`; current actual artifact policy values match the running contract.

**P2, repaired: effective calibration identity in standalone events.** Initially, the documented external scale/bias overrides changed confidence and confirmation decisions without recording those effective parameters or changing `model_revision`. A synthetic fixture using the actual trained weights and bias overrides -100/+100 produced proposed/confirmed states under one identical revision. Current `event_gate.py:209-217`, `:230-255` records effective scale, bias, threshold, timing padding and raw decoder, and adds a canonical calibration hash to the revision. The same independent reproduction now records the effective bias and yields different revision identities. The pipeline also binds full calibration, active decoder and relevant model weights into its processing revision, including custom display labels; targeted integration tests passed. No learned parameters, truth semantics or feature rows were refitted by this repair.

**P2, repaired: calibration evaluator's spatial compatibility guard.** The new calibration evaluator originally scored features directly without the spatial-checkpoint equality enforced by `generate_events`. It now checks each region's spatial identity against the gate checkpoint before loading feature arrays or interpreting truth (`evaluate_gate_calibration.py:102-104`). An independent fixture deliberately supplies mismatching identities, no NPZ/cache arrays, and a truth-reader trap. It rejects before truth interpretation, needs no array files, and writes no output. Full freeze validation still precedes this check.

The gate source changed during review to make these bounded compatibility/provenance repairs. Initial evidence and final source snapshots are retained. The checkpoint's original training source hashes remain historical provenance and were not rewritten.

## Actual trained artifact checks

Actual weights SHA256: `9bae4c22a5fc6dca64b270a0e37a8d899424477ed7b2c4f6ac5c2e6e811e62ba`.

Actual examples SHA256: `88294adb63dac8486603c7a9058aa44e3bf19af4f99bb1364805f6a3928e1e07`.

Checkpoint, examples and config hashes match their metadata. At the start of review, all six recorded training source hashes matched the corresponding source bytes. Current changes to gate inference add the repairs described above; training source, truth and supervision source remain the recorded versions.

| Split | Prefix rows | Distinct proposals / total proposal mass | Positive proposal mass | Regions |
|---|---:|---:|---:|---:|
| TRAIN | 66,351 | 19,891 | 249 | 5 |
| DEVELOPMENT | 18,077 | 5,505 | 99 | 2 |

Independently verified: each proposal's row masses sum exactly to one; labels are constant within a proposal; at most four rows are retained per proposal; every feature value is finite; recorded region sets match the configured TRAIN/DEVELOPMENT sets and exclude every confirmation region. Recomputed TRAIN-only weighted means/stds exactly match the checkpoint. Both methods' stored DEVELOPMENT prevalence and weighted Brier diagnostics reproduce from the actual examples, models and stored transforms. Both stored Platt fits report success. These are artifact-consistency checks and fitted development diagnostics, not held-out performance evidence.

## Causality, labels and lifecycle

- Inference receives prediction-derived site features, usability, births, geometry and source metadata. No truth labels or matching results enter `generate_events`. Pre-change pooling stops at the last stable support; post-change pooling stops at the current usable observation. Existing synthetic tests and an additional actual-weight synthetic future-edit/prefix reproduction preserve the earlier answer exactly.
- Raw proposal IDs, geometries, births and persistence policy are shared with the ungated proposal baseline. Raw eligibility precedes gate confirmation. A low gate score retains a proposed revision; later usable evidence may confirm the same event ID. Confirmation is sticky until a raw correction. Raw retractions preserve earlier revisions, including never-confirmed proposals. Missing rows do not score or become absence. Store-history and source-availability tests passed.
- Training labels use full annotation histories only as supervised targets, under the reviewed physical-coverage protocol. Matching includes all raw first confirmations even if later retracted. Retractions stop rows only at their observed prefix; earlier examples are not discarded or relabeled by eventual survival. Full-sequence one-to-one matching can assign a positive to one of multiple compatible proposals, so the label means **compatibility under that frozen assignment**, not an independent proof of physical truth for every unmatched proposal.
- Positive/negative minibatches are balanced from actual TRAIN examples, with conditional sampling that gives each proposal unit total mass. Standardization uses TRAIN only. Final weights follow a fixed training budget. DEVELOPMENT supplies the global Platt transforms; there is no per-confirmation-region fitting. No held-out split was opened by the trainer, and the split-overlap guard runs before collection.
- Sources in emitted learned events cover the usable prefix required by pooled/seasonal evidence, and each revision records its current availability. Target-derived row labels stay in the separate examples artifact. The added site/support indices are prediction-derived provenance.

No target leakage or future-input leakage was found in these inspected paths. This does not replace final freeze verification of the actual files consumed by the eventual held-out run.

## Held-out calibration evaluator assessment

`scripts/evaluate_gate_calibration.py` uses the same IoU 0.25 / first-support tolerance 93 days / no delay-cap compatibility assignment as training. It uses the same first eligible plus at most three later usable rows, stops at observed retraction, and gives each proposal total mass one. It loads frozen TRAIN standardization and fitted DEVELOPMENT scale/bias; it performs no fitting. Full-freeze validation precedes truth interpretation, and the new spatial guard is verified above. The evaluator is included in the source boundary.

This is calibration on a **raw-proposal, bounded-prefix distribution**. It is not calibration of only accepted first alerts under the gate's repeated-decision stopping rule, and it says nothing about objects omitted by spatial proposals. Report that distinction. Inference can wait beyond the four sampled prefixes; performance and confidence behavior on such delayed decisions should be described by the frozen event/delay results rather than inferred from fitted development Brier values. The stated three-region limits and absence of arbitrary-shift guarantees are appropriate. A negative eventual result is valid evidence and does not require redefining targets or selecting a different confirmation protocol.

## Executed evidence and limits

- Initial selected run during integration: 45 passed, two evaluator import failures because `pipeline.method_events` had not landed. After integration: **49 passed**, four upstream warnings, 3.62 seconds.
- After final gate provenance repair: **11 gate tests passed**, 1.39 seconds; **10 pipeline/protocol tests passed**, six upstream warnings, 2.16 seconds. A mistakenly requested nonexistent integration-test filename collected no tests; it was corrected to the actual `test_gate_pipeline.py` and the successful check is the one reported here.
- Independent actual-artifact, weighting, standardizer, calibration-consistency and prefix checks: `independent-review-v5-evidence/artifact_checks.py`; final output `artifact-check-after-fixes.json`.
- Calibration spatial-guard failure ordering: `calibration_guard_check.py`, `calibration-guard-output.json`.
- Earlier calibration-provenance failure: `artifact-check-before-provenance-fix.json`.
- Reviewed final sources: `source-snapshot-final/`; hashes in `source-sha256-final.json`.

The main agent's broader full-suite results are separate from these independently executed checks. This review did not inspect a final freeze manifest because it has not yet been created. Proceed by freezing final code/config/artifact bytes before any confirmation scoring; review the actual resulting comparisons and calibration afterward without tuning on those outcomes.

## Final reviewed hashes

- `src/terradelta/event_gate.py`: `6a4c0431c47b6977f863a6ee8ef3844dff7693040a05b79fc2e46e21b9d51267`
- `scripts/train_event_gate.py`: `91533ef9e5c86b41b40909ad56ded674892c2701a89af6ac97ff8f7f6fbd64dc`
- `scripts/evaluate_gate_calibration.py`: `d9d9e5ce0ad53099978417699d55d8fb645f14589739bbe8d6fc4d8101e64a0d`
- `src/terradelta/protocol.py`: `4ec510607dafca176d289130291c86ee71a73ce12d5cb2ee55467f9e504d0944`
- `src/terradelta/pipeline.py`: `fcbefe4b5b0a973c465081d01a4ed2d35df74316c11af08231d71f1577bae4c4`
- `src/terradelta/evaluation.py`: `66fd8b765a9d4dc552efefc4b4728858e0370b40808946ae1573121b4d2b3c7b`
- `src/terradelta/features.py`: `cb0ef4acf31e23ffac7d79d374ef24120a2e75a9c01cbb68bc4b2764ca41a31f`
- `configs/experiment-v4.json`: `71dfe56a1ff2e7d6cb75547e2bc81b02a1ca6b96f8ffed61367fc8ebbc7899b3`

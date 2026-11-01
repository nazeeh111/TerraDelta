# Versioned repair complete

Current source implements TEMPORAL_NUMERICS = fixed512-site-batch-v2. predict_temporal keeps its signature and [site,time] float32 output, pads each terminal batch to 512 with zero features and unavailable masks, then trims scores to the actual site count. Non-512 explicit batch sizes now raise ValueError rather than silently claim this numerical protocol. Existing repository callers use the default 512.

Only src/terradelta/temporal.py and tests/test_temporal_contract.py were changed by this agent. No model architecture, trained weights, calibration, thresholds, decoder or gate inference changed. The main agent separately owns processing-v3 engine identity in pipeline.py. Original commit bd33082 and artifacts/freeze-v2.json remain preserved for the historical study and its documented negative findings.

Verification:

- 18 focused temporal tests pass (27.80 seconds), including nonzero random recurrent/emission parameters across site counts 17, 513, 1025 and changing prefix counts 11, 500, 519. All three learned methods compare exact scores and nonempty complete revision histories for prefixes and future feature/mask edits. The model input hook enforces the fixed 512 numerical contract even when particular random weights happen to round different batch heights identically.
- Targeted Ruff and git diff --check pass.
- The new regression contract fails against the preserved original terminal-batch implementation.
- The actual patched function and final processing-v3 pipeline pass all 9 method/confirmation-region combinations for bitwise calibrated prefix/future-edit scores and every historical revision field. Source hashes in padding-patched-all-regions.json match current source. Prior native candidate/feature recreation was reused only after verifying exact feature-file bytes against its receipt. No duplicate native-image reconstruction, label read, training or refit occurred.
- Full-sequence comparison with preserved original inference changes small score values, at most 8.344650268554688e-7. Event IDs, states, decisions, geometry, sources, times and revision counts are identical across all 9 combinations. Exactly one confidence changes by 1.1920928955078125e-7, as documented in final-evidence.json: no_quality, region 1049, event td-d14ba86ae0ec722e05d4, revision index 1880, 0.6155861616134644 to 0.6155862808227539. This new receipt compares scores through the same current dispatcher identity so confidence differences can be isolated; it does not relabel original published revisions.

Current receipts: padding-repair-summary.json, padding-patched-all-regions.json and padding-unit-tests.log. Earlier RESULTS.md/final-evidence.json describe the preceding in-memory read-only phase; retain them as historical evidence. No event-gate batch optimization was applied because its same-current-frame probe changed logits.

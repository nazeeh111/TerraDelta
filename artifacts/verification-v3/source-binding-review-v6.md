# Source binding review v6

**Clear for the bounded compatibility and early-failure scope at the hashes below. No unresolved consequential finding in this recheck.** The zero-scale regression from v5 is fixed, and the earlier v4/v5 guard failures remain rejected. All 13 source/config hashes were stable before/after 41 small independent probes. V4 and v5 evidence is preserved. This is not a scientific-performance approval or a validation of future modified bytes.

Only synthetic manifests, tiny synthetic inputs/checkpoints and the actual immutable first-study artifacts were opened. No actual AOI metadata, imagery, feature/supervision arrays, ground-truth geometry or truth cache was opened; no MPS, source changes, model overrides, publications or commits occurred. The one successful freeze control hashes only a temporary synthetic zero-array `instances.npy`.

**Early failures observed.** Eight bad-head/calibration freeze variants reject with zero confirmation-truth hash calls: missing gate Platt fit, negative gate scale, missing selected-head SHA, wrong candidate policy, NaN event bias, negative occupancy scale, fractional decoder confirmation count and missing event operating point. A fully valid bound control freezes successfully and reaches its one synthetic truth hash, showing the checks are not blanket rejection.

The common dispatcher rejects a partially versioned calibration without the selected gate-head SHA. The CLI rejects a separate backbone with calibration `{}` before any spatial prediction call. Explicit legacy candidate `.65` with null inference metadata is rejected; an unbound temporal head against an unknown non-v2 spatial SHA is rejected. The trainer's actual `platt_calibration` function on synthetic one-class labels returns `scale=0.0`, and both gate kinds now load that exact constant-prevalence fit successfully. This retains the intended one-class behavior while rejecting negative/nonfinite fits.

**Actual frozen first-study compatibility observed.**

- `preflight_spatial_checkpoint` accepts actual spatial-v2 and identifies the known SHA with no external backbone.
- Actual temporal-v2 loads all three temporal kinds, and actual event-gate-v1 loads both learned gates through `preflight_heads`.
- Actual calibration-v2 passes the new operational guard for all twelve methods and matches the actual selected spatial/temporal/gate artifact hashes.
- All twelve methods execute on the same small synthetic feature sequence using those actual old heads and calibration, both with historical null inference metadata and the explicit reviewed tile 1024 / halo 64 / alignment 8 / scale 2 / candidate .45/source allowlist. Gate revision counts are respectively 7, 8 and 7 for observation_gate/gate_no_quality/proposal_tracking in each metadata mode. The nine occupancy methods return an empty history on this synthetic sequence under their actual frozen thresholds. These are execution/compatibility checks, not real-AOI accuracy measurements.

Legacy replay is intentionally limited. This recheck does not claim current full canonical processing metadata equals the archived first-study engine or that its old freeze can be revalidated under changed source. Original first-study reproduction still uses its archived commit; unused legacy-v1 workflows remain archived. No metadata was erased to pass the explicit replay checks.

**Evidence and limits.** Executable `source-binding-review-v6-probes.py`, captured `.log` and `source-binding-review-v6-evidence.json` retain each result, zero-truth/compute counters and stable hashes. The parent supplied the full-suite result of 165 passed in 119.41 seconds; its durable log was read, not independently rerun here. This review deliberately avoided another broad test cycle. Earlier all-TRAIN/DEV preflight, candidate/head identity and exact implementation-set probes remain preserved in v4/v5; the bounded repair source changes were inspected and no broadened numerical/model change was identified.

**Actual immutable old artifact hashes.**

- `spatial-v2.pt`: `7ab95679a2009918dbab62fca64244476df519080cb8c0d376c00d1494983886`
- `temporal-v2.pt`: `11af3e5e4e105795f4c74d00887afc786b906e786cb2c2c5afb4c085a5115295`
- `event-gate-v1.pt`: `9bae4c22a5fc6dca64b270a0e37a8d899424477ed7b2c4f6ac5c2e6e811e62ba`
- `calibration-v2.json`: `6d448246aa6c0cc21fcd69d675c6f591bd7aef996edaa2da9ccb659f7b4b95cf`

**Reviewed implementation/config hashes.**

- `src/terradelta/contracts.py`: `a31b0b9c6267c95cbc1ba6f12a9de856c4563bf18fcffcf122286c5e0d059280`
- `src/terradelta/event_gate.py`: `0b6b38054ac84c813adcf0eb5ed7b775d6f0b2470c60247a949e1fef38c8df8b`
- `src/terradelta/temporal.py`: `534cc05ca713d9d3f66d8ee8a64f3efe0ff2f01be0a02a31a8abdce047fd2159`
- `src/terradelta/cli.py`: `bfedff5a3f40ef55eba095b9ac8acb9cc07bbbea34d91bef7c45737a0983ec91`
- `src/terradelta/protocol.py`: `b774bb0f209d296181c7ff7d4b50d6618159bcd22b0d0a8c1b4a3f70be7a8960`
- `src/terradelta/pipeline.py`: `2e3cac760259a4f13c2edc22ff2a7f633c8f9be635e0176df2a1a1e414044069`
- `src/terradelta/spatial.py`: `a801eef9f770057cb7e5e26f74b7edc027ea73cd3136da8d1e7541d20b03ebff`
- `src/terradelta/pretrained_spatial.py`: `f529a7bb201106964a974944fd096e5b0b60a90700254e7462cc16a3b28fe503`
- `scripts/train_event_gate.py`: `eded8c52ba7be4d36fa43cd96e94df83f844adf3c093a166c256adac4adb8b93`
- `scripts/train_temporal.py`: `2d20a781982e20c60999c9c35c9c75d74b52ab894133629749ce077a89b66068`
- `scripts/calibrate.py`: `2c8cb87a2c6fc77151d011058c5416ffdb130ab7c3ec665b8a54b28dd2819a2f`
- `scripts/predict_all.py`: `b9bcc9ccb80116e78ad39a047acaf64e4af457ecc921ac30caced2e2a8b0450f`
- `configs/experiment-v6.json`: `531999182720a75a09bf394d8bcb4a4a25983a0e345aad925b33bc0eaa73d159`

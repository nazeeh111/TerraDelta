# Actual V8 confirmation-boundary review

Reviewed 2026-11-01T11:17:04.999658+00:00. **No actual boundary defect was found. V8 remains unscored as directed.** This conclusion applies only to freeze `a61666000acd1737c914a8b508db7d9822f4bba008610aa4cbec0be258903e70` and its exact bound artifacts. It is not a generic approval or a positive scientific outcome.

No source, data, model or configuration was edited. No MPS, Git or public action ran. The reviewer did not call `ground_truth_events`, load confirmation annotation arrays, or compute confirmation performance. Three prepared confirmation instance files were byte-hashed as required by the boundary validator; annotation interpretation was actively blocked in the audit process.

## Exact artifacts

| Artifact | SHA256 |
|---|---|
| `artifacts/calibration-v3-base.json` | `318bdea59d5c85af38b31cd293532f52d826eb02d3a647f1c7800180e8875487` |
| `artifacts/calibration-v3.json` | `24c69214030c75ebfd849b3e42f95103f74b1808c11d82458f4cbc22754fa41a` |
| `artifacts/freeze-v3.json` | `a61666000acd1737c914a8b508db7d9822f4bba008610aa4cbec0be258903e70` |
| `artifacts/temporal-v3.pt` | `f9eb05e197436f53cae8d4af25f479264cd5ca3c414a09d52bf1efa9c15b827f` |
| `artifacts/event-gate-v2.pt` | `1ff0cc2e7d2cfc2555c14d86f08a448441c8f82ff0bf3b3038127088dd4ea266` |
| `artifacts/spatial-v3.pt` | `7c35a3f04ef9e7610e4618a003a056ce6512a930be67db2ce9999ad8dc24820d` |
| `configs/experiment-v8.json` | `ed4204738e8a0079453d7469e64ed869dbf089dc2b78c0500004697d01d87bd9` |
| `configs/experiment-v7.json` | `bb0e9f7c1e41ad21d2183b6a3de0962825ede5f82dcb38a26f83cb798e0ac973` |
| `confirmation-preparation-v3.json` | `6af00c0052ee9364a17af36eafb90a538491bffdb0a94ba76476b21c77710460` |
| `online-invariance-v3.json` | `6a6fa9aa455733fef1e35f55a52f587898db00198092c7da9489c68ed43a1f87` |

The preparation receipt reports source commit `cfe2d6985efc7cd26de4ea7e179941c19c7ebfc2`. This audit did not use Git; it independently matched every actual frozen protocol-source byte hash, including package-version bytes, against both the freeze and the preparation receipt. The models retain actual V7 training identity while the comparison/calibration/freeze retain V8 identity. Exact sibling-file hashing and the whole-config comparison establish the bounded comparison-only amendment.

## Verified actual boundary

- The split is exactly 9 TRAIN, 4 DEV and 3 fresh confirmation regions, all mutually disjoint. Old audit-only regions appear in none of them. The exact held-out IDs are listed below.
- All 13 methods have one scalar global operating point. Selected DEV reports contain exactly the same four DEV IDs and no held-out ID. Every chosen point equals the recorded candidate-grid winner under its source-declared tie rule. The neural method has all 28 predeclared distance/decoder combinations, with equal regional means independently checked. Adding it preserved every base 12 operational parameter, selection report and timing calibration exactly.
- All operational parameters pass the actual preflight. Occupancy and event-gate calibration values are finite, slopes nonnegative, and actual optimizer success flags are true. Pairwise alone uses one support/zero dwell. Other occupancy methods use two supports / 20 days; all gate methods retain the exact raw proposal decoder. Timing padding matches each method’s DEV timing receipt; revision window 100 days and maximum gap 190 days remain fixed.
- Actual temporal and gate state dictionaries load strictly through boundary validation, have finite parameters, correct feature contracts, matching spatial/site policy, and V7 training headers. Temporal submodels are observation/no_quality/WATCH; gate submodels are observation_gate/gate_no_quality. Recorded training/development memberships, seeds and budgets match. Actual TRAIN/DEV spatial, temporal and gate input hashes, numerical-source hashes, companion metadata hashes and gate example-file hash independently match current immutable inputs/receipts. No confirmation membership occurs in those training data protocols.
- The original spatial training config and explicit recovery receipt match their V8 references. The spatial checkpoint is unchanged 7c35a3..., with the official frozen backbone matching its complete f37072... hash. This checks the documented recovery, not a claim that its historical terminal guard cleanly passed.
- All seven actual neural sidecar manifests use CPU, the exact v2 representation, V8 config, and held module/extractor hashes. Actual four-DEV sidecar file hashes match those recorded by the selected calibration. All three confirmation sidecar files, actual probability files and image-only feature/site/source metadata match the frozen boundary. Confirmation features contain only x/usable/births, without supervision.
- `validate_experiment` returned the actual full-freeze-v5 object successfully. It took about1.662s. The reviewer replaced truth-event construction and annotation-array loading with failure sentinels; neither was invoked. Only the three expected confirmation instance paths reached the blind byte-hash hook. Every source/freeze fingerprint was checked again afterward.

Held-out regions:

- `L15-1172E-1306N_4688_2967_13`
- `L15-1204E-1204N_4819_3372_13`
- `L15-0358E-1220N_1433_3310_13`

## Actual selected global settings

The decoder threshold applies to calibrated occupancy, or to the shared raw spatial proposal for gate methods. The separate learned-gate retention threshold must not be confused with that raw proposal threshold.

| Method | Decoder threshold | Learned-gate retention | Neural cosine-distance threshold | Supports / dwell days | Timing padding days |
|---|---:|---:|---:|---:|---:|
| `pairwise` | 0.55 | n/a | n/a | 1 / 0 | 61 |
| `spatial_tracking` | 0.55 | n/a | n/a | 2 / 20 | 61 |
| `smoothing` | 0.65 | n/a | n/a | 2 / 20 | 61 |
| `hmm` | 0.65 | n/a | n/a | 2 / 20 | 31 |
| `sn7_retrospective` | 0.65 | n/a | n/a | 2 / 20 | 31 |
| `ted` | 0.55 | n/a | n/a | 2 / 20 | 62 |
| `observation` | 0.55 | n/a | n/a | 2 / 20 | 61 |
| `no_quality` | 0.55 | n/a | n/a | 2 / 20 | 61 |
| `watch_lstm` | 0.65 | n/a | n/a | 2 / 20 | 31 |
| `observation_gate` | 0.55 | 0.2 | n/a | 2 / 20 | 31 |
| `gate_no_quality` | 0.55 | 0.2 | n/a | 2 / 20 | 31 |
| `proposal_tracking` | 0.55 | n/a | n/a | 2 / 20 | 61 |
| `embedding_distance` | 0.55 | n/a | 0.1 | 2 / 20 | 61 |

## Real online evidence and scope

The actual `online-invariance-v3.json` receipt has 36 passing region×method checks: all 12 online methods on all 3 held-out regions. Each comparison has nonempty early revisions, checks the complete earlier EventRevision records including transaction clock/model identity, and checks future feature/availability edits. The neural method additionally checks vector/mask edits and independently re-extracts actual first/last-prefix-frame vectors on reconstructed prefix sites. The retrospective SN7 comparator is correctly excluded from online claims.

The receipt’s output SHA matches the successful preparation stage. Source hashes were held around the actual subprocess, its command paths match the current artifacts, the exact model/calibration/config files predate that stage, and the freeze follows its completion. Its direct feature and gate hashes also match the frozen artifacts. This is an audited existing run, not a second execution of the 555-second check. The standalone invariance JSON does not itself contain separate temporal/calibration hashes, so retain the preparation receipt and immutable artifacts with it.

All actual source timelines validate as ordered acquisition/availability evidence. Publication times remain unknown and explicitly use retrospective month-end availability. The decoder supports conservative observation intervals, clamps padded intervals at available evidence, and does not establish exact construction/demolition dates.

The separate task-only DEV geometry investigation was not read or changed. No scoring command was launched. Preserve this unscored V8 preparation regardless of that prototype’s outcome; any future study revision needs its own declared source/data/model/calibration boundary. A boundary pass does not establish accuracy, a quality-gate benefit, timing coverage or analyst readiness.

Reproduce this audit with task `.venv/bin/python confirmation-boundary-v3-review-evidence/audit.py`. The task-local JSON retains exact headers, selected global parameters, input/source verification results and access limits.

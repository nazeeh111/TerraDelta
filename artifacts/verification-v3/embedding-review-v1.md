Public copy; local path prefixes replaced. Original report SHA256: 8469b9ea6af261757676868a5102030b56f593321c8720bc06607aa2b029d0b4

# Independent neural-embedding integration review

2026-11-01. **No open consequential correctness issue remains in the reviewed current source.** The v2 extractor and prospective V8 integration are clear to proceed with CPU image-only extraction and DEV-only calibration. This is not final experiment-freeze approval, an accuracy result or analyst readiness.

This worker made no repository edits, read no AOI truth arrays or performance outcomes, used no MPS, and performed no Git/public actions. Probes use invented data and the existing official local ResNet18 backbone on random image pixels. Actual production sidecars, fitted heads and operating points remain separate execution evidence.

## Current identities

| File | SHA256 |
|---|---|
| `src/terradelta/embeddings.py` | `3a05e90a3bab8c80cd4d831376226222de4ce494c50e986868c6ab26c4170e6f` |
| `src/terradelta/pipeline.py` | `57ef8e5a73f50e85fc068c4bb2a04dcda87a0cbc36330864857c534c26cbfaf0` |
| `src/terradelta/contracts.py` | `7da8a05ed2f7cb8fdeb3e4b13d90ee272dd7b2b96d624842754f465ed1b50308` |
| `src/terradelta/protocol.py` | `bffae3afe773dadf854e9f72ee52d92bd94fb93cafea33875bb5e632bc177df7` |
| `scripts/extract_embeddings.py` | `35bbf61197d25709fefbd519e6a6c12300e0155a22e387e5dcb24f38f09210ba` |
| `scripts/calibrate_embedding.py` | `55412d39cdf4e327adc1c13cb366ee988dcf503dd8f448b11a7574d1659d1fb5` |
| `scripts/evaluate.py` | `69112772e7b7620f63bfb8be07eebd9ad28bae8aa43711c60f062966ccc1ed55` |
| `scripts/calibrate.py` | `9c13c87e4e4edb052e82f038eff4e1b5b592eed9c27ddc94ff722f956d383f19` |
| `scripts/check_online_invariance.py` | `2f3a8112407e5dd70ee86e97e4e394676bb9e01cb2d878eb32ec351a76479f18` |
| `scripts/stress_and_calibration.py` | `1911944b1fb98877f6e720c75ed7e442a1868aa157d93d78370564bb22871189` |
| `scripts/evaluate_gate_calibration.py` | `3e269c8c5fa3ed2a1b03a155e76ca7ae0180c2c2101a3ce24b78b2689c8b5394` |
| `configs/experiment-v7.json` | `bb0e9f7c1e41ad21d2183b6a3de0962825ede5f82dcb38a26f83cb798e0ac973` |
| `configs/experiment-v8.json` | `ed4204738e8a0079453d7469e64ed869dbf089dc2b78c0500004697d01d87bd9` |

The original prospective V7 config is preserved. V8 changes the embedding-history semantics before any AOI descriptor outcomes, extraction/calibration or fresh confirmation scoring. V8 explicitly references the actual V7 head-training file; numerical head artifacts and their original metadata are not rewritten. The published research tag/data at b2ed remain outside these new-study edits.

## Consequential findings repaired during this review

| Finding | Severity | Reproduction and final resolution |
|---|---|
| Newborn additions disappeared | P1 | With all observations usable, birth2 and spatial probabilities `.1,.1,.9,.9,...`, the first implementation returned zero events at every distance threshold, while the shared decoder proposed and confirmed. Current v2 recovers both records at all seven declared points and emits nothing before birth. Missing archived absence remains inventory, not an invented addition. |
| Incomplete extractor/calibration source binding | P2 | Stale/missing producer identities could be accepted. Current exact two-file extractor map and calibration selection/dependency map reject malformed sources before the synthetic truth boundary. |
| Floating lookback passed equality check | P2 | `reference_lookback=3.0` passed preflight and raised a slice-index TypeError inside decoding. Current exact-int preflight rejects both float and boolean forms. |
| Unverified diagnostic probabilities | P2 | Stress diagnostics could read a stale probability folder despite a valid feature freeze. Current all-region helper checks actual probability-file SHA against each site manifest before full-freeze/truth access. |
| Type-insensitive head-config comparison | P2 | `seed=2901.0`, float resources and numeric booleans passed Python dict equality. Current canonical JSON equality rejects these, as well as every unapproved config field change. |
| Late frozen identity failures | P2 | A self-consistent replacement sidecar, deleted embedding header or wrong frozen training header could reach truth hashing before rejection. Current full-freeze-v5 performs these source/image/sidecar/config/format checks first. |

The evidence folder preserves earlier observations and repaired-state probes separately. All findings above are closed in the final hashes; none asserts a scientific performance result.

## What was independently verified

`embeddings.py:137` hashes the exact official backbone before a complete strict load, with downloads disabled. Parameters and BatchNorm remain frozen. Its native-stride4 layer1 produces genuine 64-dimensional neural vectors. The reviewer loaded the official SHA-verified file on CPU with random 37×43 pixels: altered pixels changed descriptors, every encoder state tensor remained unchanged, and no gradients accumulated.

`embeddings.py:167` pools source-valid site pixels at the actual convolution lattice, with fixed native-index chunk coordinates and reduction boundaries. New sites can therefore be included without shifting older sites’ arithmetic. The independent CPU probe includes future-site pixels and still gets bit-identical old-site vectors. `:226` uses only prior usable vectors for cosine references, and `:262` gates ordinary spatial state updates. Backfilled pre-discovery vectors describe already available archived images once the image-derived geometry is discovered; the unchanged shared decoder alone prevents pre-birth publication. This matches the existing 17-dimensional site-history convention without changing older sites or adding future image evidence.

`extract_embeddings.py:46` opens only RGB, valid masks, image-derived sites and image-only feature caches. It does not read instances or supervision, does not modify common sites/features, and refuses injected encoders for research outputs. `:93` rejects source/input mutation before writing a complete manifest. `embeddings.py:63` validates config, exact descriptor, source frames, pixel/mask/site/feature hashes, implementation map, vector-file/hash/dtypes/shapes and hard usability before use.

`pipeline.py:328` keeps full future-input hashes in the experiment boundary rather than earlier processing identities. Every synthetic prefix and future-vector edit retained complete earlier EventRevision records, including record time and model revision. `:381` rejects wrong source context/config/births. `:248` retains the shared two-observation/20-day persistence and rejects an undeclared decoder point. Missing observations do not become absence.

`calibrate_embedding.py:31` checks the declared representation, source/config/site identity and all DEV sidecars before supervision/truth. `:72` evaluates exactly seven distance points × four decoder thresholds, selects one global macro 62-day F1 point with the declared tie order, and retains all region reports. Timing padding uses DEV only. `:133` binds the selection source/dependency map and `:149` checks it remains unchanged. No region-specific operating point or fresh-confirmation selection was found.

`contracts.py:14` allows only an explicitly SHA-verified sibling training JSON and rejects self/chained aliases. Its whole-config canonical comparison removes only `version`, `embedding_baseline`, `embedding_baseline_declared_at`, `embedding_amendment`, and `head_training_config`; every other field, including unknown future fields and numeric types, must match. Eighteen independent cases gave two allowed controls and 16 unauthorized rejections. Both head trainers currently ignore the excluded comparison block. Config equivalence does not itself establish input-byte identity; actual fit input-hash receipts and held data lineage remain required evidence.

`protocol.py:101` checks the calibration’s actual head-training identity before any truth; `:125` checks every sidecar before the truth loop. `:195` revalidates format, source, config/model/calibration, image/site files and exact frozen sidecar hashes before calling the truth-hashing boundary. Twelve malformed initial-boundary cases and eight malformed revalidation cases caused zero synthetic truth-hash calls. Valid controls and a host-local path change reached the sentinel, showing the rejection tests did not simply fail every input.

`check_online_invariance.py:126` reconstructs prefix sites and recomputes real boundary-frame vectors before comparing earlier records. `stress_and_calibration.py:171` uses label-stable annual sites only as an explicitly conditional post-freeze diagnostic; masks, full-AOI denominator limitations and absence of tuning are stated. Probability files are now checked before either diagnostic uses them. The raw brightness controls remain sensitivity tests, not newly observed seasons. Exact real-frame descriptor comparison currently uses CPU; apply it to CPU-produced sidecars, or explicitly match backend if another device is introduced.

## Checks and limits

- Repository targeted tests: **45 passed, 6 upstream warnings, 21.20s** across embeddings, optional dispatch, protocol, label-boundary guards and the old twelve-method pipeline.
- New study-amendment tests: **20 passed, 4.99s**. The repository-pinned Ruff 0.14.6 check passes. A prior global Ruff binary enabled additional style rules and reported import/typing-style issues; no edits were made, and the actual pinned project check is clean.
- Reproduce task-local checks using `.venv/bin/python embedding-integration-review-v1-evidence/probe.py`, `freeze_revalidation_probes.py`, `amendment_probes.py`, and `cpu_neural_probe.py`. Logs/JSON retain exact source hashes and the original failed newborn control. The temporary datasets are entirely synthetic.

No AOI accuracy, superiority, quality benefit or timing coverage was assessed. The new comparator is an unsigned neural-distance gate on shared spatial occupancy, not a full reproduction of a published end-to-end system. Low-level dispatch accepts prevalidated dictionaries; production wrappers must continue using the strict sidecar loader and full freeze. The changed representation requires its new V8 descriptor/config and new sidecars, while actual heads retain their V7 training identity. Fresh confirmation outcomes must remain unopened until all real models, sidecars, calibration, config and consequential source bytes are frozen.

## Version-source addendum, 2026-11-01

**Clear; the completed review remains applicable.** This narrow follow-up verifies only the unreleased `0.2.0.dev1` version delta. Removing the single new `src/terradelta/__init__.py` entry from current `protocol.py` reconstructs the exact previously reviewed protocol hash. Every other previously reviewed method/config hash is unchanged. Reversing only the declared version substitutions reconstructs the prior exact bytes of `service.py`, `pyproject.toml` and `uv.lock`; the lock still contains the same 75 packages. Health/OpenAPI now obtain their version from package `__version__`; no model, decoder or guard semantics changed.

An independent synthetic valid freeze/revalidation succeeds and contains the package-initializer hash. Changing only the copied version source rejects with `protocol_sources` mismatch before any synthetic truth hashing (zero calls); restoring those bytes revalidates. No full review was repeated, no real fits/calibration were interpreted, and no AOI truth/performance, MPS or public/Git mutation occurred. Separate package-version consistency tests remain parent-owned.

| Updated file | SHA256 |
|---|---|
| `src/terradelta/protocol.py` | `59b8aac7d3a124e14f01e05cf6012983654abe4f94e1ad3428b7c28ad477b36b` |
| `src/terradelta/__init__.py` | `122cfc2362b7097e61c368cc22fbea64d0249791a871300660071629616ef725` |
| `src/terradelta/service.py` | `f64197cc2e453d2878fb7e155fb34998ae50551064c113dd015c7bc0938f9ea9` |
| `pyproject.toml` | `dc960720f35c1ffebf75ba1b484318f167f08fc5940207aa4d3fc4b42de6eabe` |
| `uv.lock` | `12473c0ceafbe0b51b3dc65d6696ef2b0f4ac51e3cbd4fcce5c3b3d1e57f0084` |

Evidence: `embedding-integration-review-v1-evidence/version-source-addendum.json` and `version_source_probe.py`. Original completed report bytes are preserved separately in that evidence folder.

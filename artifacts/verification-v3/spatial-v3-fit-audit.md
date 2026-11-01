# Retained spatial-v3 fit audit

**Clear for an explicit recovery receipt.** The unchanged retained checkpoint legitimately represents the best selected decoder from the completed 4000-step TRAIN/DEV selection loop. The command failed its final conservative disk-source guard, so it must not be described as a clean successful trainer termination. No refit or checkpoint rewrite is needed to recover this specific selection result. No confirmation result is established by this audit.

The checkpoint contains the selected step-3600 decoder. Its own `steps_completed` field remains 3600 and its trace contains 38 evaluations through that point. The separate final trace and log agree exactly on all 42 scheduled evaluations: steps 0, 1, and every 100 steps through 4000. Step 3600 is the unique maximum under the predeclared mean footprint/boundary DEV Dice criterion, at 0.4201145382457867. Later scores at steps 3700, 3800, 3900 and 4000 are respectively 0.3905568648378077, 0.386904363158182, 0.3843500321209051 and 0.3851530706788845. Thus the retained weights are the same best weights that normal finalization would have kept; the missed finalization updates provenance, not the selected decoder tensors.

Completed elapsed time is 2668.0256974159856 seconds, within the frozen 10800-second limit. This is recorded fit evidence, not a new performance evaluation.

| Preserved artifact | SHA256 |
|---|---|
| `artifacts/spatial-v3.pt` | `7c35a3f04ef9e7610e4618a003a056ce6512a930be67db2ce9999ad8dc24820d` |
| `artifacts/spatial-v3.training.json` | `a6f157762c5c35a1cbe6c305f9d46333de8884c71f078504283aa238b5d63ef8` |
| Task `spatial-pretrained-v1.log` | `41f573dcfa12ee7dbf490c5195fd48598a7b4bddc8163af9e894c9f1dba8fdf2` |
| `configs/experiment-v5.json` | `59433571ca28a99b635fb699362e0e1a91d958c2dd403a3f69e9156fdd6ed881` |

All four hashes were checked before and after the audit and remained identical. No artifact, source or model state file was modified.

All checkpoint-recorded training source hashes exactly match commit `17e6a9fde572ea9751002ac23c048c100471d68a`:

| Source | Recorded and committed SHA256 | Current relationship |
|---|---|---|
| `scripts/train_spatial_pretrained.py` | `734224bf530f78cdde70c51e18bc4f07508d693b6d5dfddcc1ec9c9ed1a65479` | Byte-identical |
| `scripts/train_spatial.py` | `787158065edc05b89b0affa7a56d50b3ee5428fa6f52a6955661c14b58189b28` | Byte-identical |
| `src/terradelta/pretrained_spatial.py` | `2bec25499465baa64fc9606f283c1debcef3d1f20ff88ecf0847b1d68c0d3021` | Formatting plus one unused plain attribute |
| `src/terradelta/spatial.py` | `9a855209ac02a6eda14cf51ade292b21b7fed11ffe8be7963ed9eeb3a1fa0f2e` | Only `predict_scene` differs |

After removing only `self.native_alignment = 32 // self.input_scale`, the entire current pretrained model module has the same Python syntax tree as the committed training version. This attribute is an ordinary integer, absent from parameters, buffers and the state dictionary. `_block`, normalization, frozen encoder behavior, decoder construction, model-grid forward, native-grid forward and checkpoint serialization are otherwise identical. In `spatial.py`, all definitions except `predict_scene` are unchanged, including the shared masked loss.

The training callgraph never calls `predict_scene` or reads `native_alignment`. It constructs the model once, samples TRAIN patches through unchanged `train_batch`/`extract`, computes unchanged direct model forwards and masked loss, updates only decoder parameters, and evaluates unchanged fixed DEV crops through `validate`. There is no module reload or dynamically executed source in the trainer/sampler. Editing their files on disk does not replace already imported function objects. The changed inference tiling therefore cannot change this fit's loaded training computations or selected crop metrics.

The final traceback points to the source/config mismatch at trainer line 228. In the unchanged control flow, the final `assert_frozen(model, encoder_before)` at line 223 precedes that guard. Its success is therefore inferred from the exact termination location. The final provenance update and checkpoint rewrite at lines 229-239 did not run. This distinction must remain explicit; do not fabricate a terminal encoder-audit field inside the retained checkpoint.

Config bytes match the embedded checkpoint config, recorded config SHA and commit 17e6a9f. Every recorded RGB, target, edge, validity and metadata hash was independently matched against the actual 13 TRAIN/DEV cache regions. The nine TRAIN and four DEV regions exactly match the frozen prospective selection manifest. Fresh confirmation and old audit-only regions are disjoint from these inputs. The training code opens only TRAIN and DEV lists. This audit streamed TRAIN/DEV file bytes for hashing, computed no label metrics, and opened no confirmation or audit-only arrays.

As a bounded runtime cross-check, the committed and current model definitions loaded the same local official encoder and retained decoder on CPU with two threads. Their registered state keys and values were identical, all retained decoder tensors were finite, and one synthetic odd-size input produced bitwise-identical native outputs and masked loss. No optimizer, update, refit or real-data metric was run. This checks source equivalence; it is not a repeated MPS training run or a claim of bitwise MPS reproducibility.

Recovery constraints:

- Keep checkpoint, trace, log and config bytes unchanged and bind their full hashes in the separate receipt.
- State selected decoder step 3600, observed completed optimizer loop 4000, and failed source-guard termination separately. Do not rewrite the checkpoint's `steps_completed=3600` field to 4000.
- Bind fit provenance to commit 17e6a9f. Bind later inference tiling to its own current implementation identity rather than attributing it to the fit.
- Do not claim final-step-4000 decoder weights, a saved optimizer state, a clean terminal trainer receipt, or a completed confirmation study.
- Preserve the fresh holdouts and separate prospective inference-policy selection. These DEV scores do not establish confirmation performance.
- This narrow recovery does not authorize disabling the source guard or accepting future source/config mismatches without their own evidence.

The loaded process's code was not separately captured from memory. The matching recorded/committed sources and standard single-process import behavior support the interpretation above. Frozen-encoder final success is a control-flow inference, while exact artifact/config/source/input hashes, trace agreement and synthetic runtime comparisons are directly checked facts.

Machine-readable result: `spatial-v3-fit-audit.json`, with `status: clear`, exact artifact hashes, `completed_steps: 4000`, `selected_step: 3600`, source comparisons, input hash checks, verification details and constraints. This audit is complete before any subsequent inference-selection action.

# Reproduce the bounded study

The original frozen study uses `configs/experiment-v4.json`, spatial-v2,
temporal-v2, event-gate-v1 and calibration-v2. **Its confirmation result is
negative:** [geographic transfer failed](results-v2.md). The released histories
are immutable outputs of that study. Restoring them requires no model execution.
The [editable demo](media-v2/README.md) is a replay of those same outputs.

Exact historical source reproduction requires checkout **`bd33082`** (full
commit `bd330827c76c29e20fcf3735357ec05ed35d5f3e`). Preserve your active checkout;
use a separate clone or worktree. The original `artifacts/freeze-v2.json` binds
27 source files plus model, configuration and prepared-input hashes. Current
source is intentionally different and must reject that original scientific
freeze. Do not rewrite the manifest to make a changed implementation pass.

```sh
git worktree add --detach ../terradelta-frozen-v2 bd33082
cd ../terradelta-frozen-v2
```

The commands below are a separately identified reconstruction using those
historical interfaces. Newly ingested arrays, inference numerics and training
can differ across environments. A newly generated freeze records a new run;
it does not establish byte-for-byte replay of the original frozen inputs.

## Environment and fixed inputs

Use Python 3.12 and `uv sync --frozen --extra ml --extra dev`. The lock selects CPU
PyTorch on Linux/Windows and the native Mac build, which can use a verified MPS
accelerator. CPU inference is supported; local CPU spatial training was much
slower than MPS. No accounts, billed APIs or paid imagery are required.

`configs/data-manifest-v1.json` pins 714 public source objects with hashes,
covering 238 monthly mosaics plus matched labels and UDM masks in ten complete
regions. The bounded download is about 1.36 GB. Imagery, labels and distributed
derived data/weights are CC BY-SA 4.0 with attribution. Code is MIT.

The five training, two development and three confirmation regions were fixed
before outcome inspection. Adjacent training crops are not independent held-out
regions. Original image publication times are unavailable; S3 LastModified is
object-distribution metadata, not original publication time. Month-end availability
is an explicit retrospective assumption.

## Reconstruct with the original frozen source and weights

Use a new working directory for every result version. These commands use the
distributed checkpoint identities rather than assume a new training run will be
bit-identical across hardware and library numerics.

```sh
uv run terradelta fetch-sn7 --output data/sn7 \
  --config configs/experiment-v4.json --manifest configs/data-manifest-v1.json
uv run python - <<'PYTHON'
import json
from pathlib import Path
import subprocess
settings = json.loads(Path("configs/experiment-v4.json").read_text())
for split in ("train", "development", "confirmation"):
    for region in settings[split]:
        subprocess.run([
            "terradelta", "ingest", str(Path("data/sn7") / region),
            "--output", str(Path("work/replay/cache") / region), "--labels",
        ], check=True)
PYTHON
uv run python scripts/predict_all.py \
  --cache-root work/replay/cache --output work/replay/study \
  --weights artifacts/spatial-v2.pt --config configs/experiment-v4.json --device cpu
```

The `--labels` imports above create offline research caches required for later
training/evaluation. Ordinary user inference omits that flag. `predict_all.py`
uses images and masks for candidate creation; it loads label supervision only for
training/development features, never for confirmation candidates. It requires
current anchored-registration metadata. The complete source grid is retained.

Feature metadata records calendar-aware quality context and the physical
80/20 supervision protocol. A reference site is present at at least 80% any-building
coverage, absent at at most 20%, and unknown in between, with at least 80% source
validity. Two usable observations establish persistence. ID reassignment cannot
alone establish demolition. Bounded duplicate grouping and exclusions retain
an audit. These labels do not annotate exact physical event dates.

Calibration fits global parameters on the same two development regions for all
twelve methods. Creating this new local freeze hashes prepared confirmation
bytes without interpreting their event outcomes. It is a new replay artifact;
do not substitute regenerated files under the original study's manifest.

```sh
uv run python scripts/calibrate.py \
  --features-root work/replay/study/features --cache-root work/replay/cache \
  --weights artifacts/temporal-v2.pt --spatial-weights artifacts/spatial-v2.pt \
  --event-gate-weights artifacts/event-gate-v1.pt --config configs/experiment-v4.json \
  --output work/replay/calibration.json --freeze-output work/replay/freeze.json
uv run python scripts/evaluate.py \
  --features-root work/replay/study/features --cache-root work/replay/cache \
  --weights artifacts/temporal-v2.pt --spatial-weights artifacts/spatial-v2.pt \
  --event-gate-weights artifacts/event-gate-v1.pt \
  --calibration work/replay/calibration.json --config configs/experiment-v4.json \
  --split confirmation --freeze-manifest work/replay/freeze.json \
  --output work/replay/confirmation
```

The evaluator validates models, calibration, configuration, prepared inputs and
consequential source hashes before interpreting confirmation truth. A changed
model, method, threshold, mask, feature, truth byte or metric invalidates that
freeze. Preserve the failure and start a declared new version. Do not tune from
confirmation results. The comparison is written to
`work/replay/confirmation/comparison.json`, with one report per method and region.

## Consume the original outputs with current source

Use the history archives' unchanged `revisions.jsonl`, GeoJSON and GeoParquet for
original-output replay. The bundled `tools/restore_history.py` rebuilds a new,
compact database and verifies every canonical revision, record time and stored
hash before publishing the output directory. This new database is a separate
physical artifact; its file hash is not the original database hash. See the
release `README.txt` for extraction and service commands. Source-image hashes,
URLs, monthly timing assumptions and provenance remain attached to the records.

Current fresh reconstruction with the original spatial-v2 checkpoint is allowed
only by an explicit reviewed compatibility identity: spatial implementation
SHA256 `a801eef9f770057cb7e5e26f74b7edc027ea73cd3136da8d1e7541d20b03ebff`,
1024-native-pixel tiles, a 64-pixel context halo, alignment 8, internal scale 2
and candidate threshold 0.45. The spatial checkpoint hash is
`7ab95679a2009918dbab62fca64244476df519080cb8c0d376c00d1494983886`.
A different backbone, source implementation or inference setting is not covered.
Current numeric padding and processing identity are recorded explicitly. Such a
fresh reconstruction is a **new processing identity**, not original bit replay.
Do not substitute its new products under the first study's frozen manifest.

## Separate fresh-region study, still underway

`configs/experiment-v6.json` declares nine training regions, four development
regions and three fresh confirmation regions. The three former confirmation
regions are audit-only. Model/feature production and the new source-bound
protocol are still underway; this document claims no new event result. The
frozen first-study scores and demo do not describe the new model. New processing,
models, fitting and a prospective freeze must remain distinct from `freeze-v2`.
The separately downloaded pretrained backbone is not included in the public
research archives.

## Stress, online invariance and native raster products

Run these commands in the historical checkout against the separately
reconstructed artifacts. Missing observations, registration shifts and seasonal context are evaluated separately from ordinary source cases.
The online check verifies both future extension and future edits; the named
retrospective SN7 method is intentionally excluded from an online claim.

```sh
uv run python scripts/check_online_invariance.py \
  --cache-root work/replay/cache --features-root work/replay/study/features \
  --predictions-root work/replay/study/predictions --weights artifacts/temporal-v2.pt \
  --event-gate-weights artifacts/event-gate-v1.pt \
  --calibration work/replay/calibration.json --config configs/experiment-v4.json \
  --output work/replay/online-invariance.json
uv run python scripts/stress_and_calibration.py \
  --cache-root work/replay/cache --features-root work/replay/study/features \
  --predictions-root work/replay/study/predictions --weights artifacts/temporal-v2.pt \
  --spatial-weights artifacts/spatial-v2.pt --event-gate-weights artifacts/event-gate-v1.pt \
  --calibration work/replay/calibration.json --config configs/experiment-v4.json \
  --freeze-manifest work/replay/freeze.json --output work/replay/stress.json --device cpu
uv run python scripts/export_rasters.py \
  --cache-root work/replay/cache --predictions-root work/replay/study/predictions \
  --weights artifacts/spatial-v2.pt --config configs/experiment-v4.json \
  --output work/replay/native-rasters
```

Native GeoTIFFs retain the original transform/CRS, source dates/identities,
footprint/boundary probabilities and validity with an internal dataset mask.
Masking invalid pixels as ordinary background would change the experiment.

## Reproduce training as a separately identified run

The initial and revised model traces remain in `artifacts/`. Spatial-v1 trained
3,000 steps; spatial-v2 warm-started from that exact checkpoint, changed internal
scale and completed 5,000 additional steps. Its 64 native-pixel training crops
and fixed 128 native-pixel development crops are recorded in experiment-v2.
This is not an isolated upsampling ablation.

```sh
uv run python scripts/train_spatial.py --cache work/replay/cache \
  --config configs/experiment-v2.json --initial artifacts/spatial-v1.pt \
  --output work/retrained/spatial-v2.pt --steps 5000 --input-scale 2 --device auto
uv run python scripts/train_temporal.py --features work/replay/study/features \
  --cache-root work/replay/cache --config configs/experiment-v3.json \
  --output work/retrained/temporal-v2.pt --steps 2000
uv run python scripts/train_event_gate.py --features-root work/replay/study/features \
  --cache-root work/replay/cache --config configs/experiment-v4.json \
  --output work/retrained/event-gate-v1.pt --steps 2000
```

The latter two commands train from the replay features produced by the distributed
spatial-v2 checkpoint. Event-gate provenance explicitly pins that backbone hash
and the reviewed truth/supervision source hashes. A newly retrained spatial
checkpoint with different bytes is not silently interchangeable. Extending the
experiment to it requires new image products, declared model/protocol identities,
new train/development fitting and a new freeze before any confirmation claim.
Existing output checkpoints are rejected rather than overwritten.

The gate learns physical-event compatibility for real raw proposals, with actual
positive/negative class-balanced sampling and proposal-weighted development
prevalence calibration. Its paired ablation uses the same examples/budget and
retains the same hard usability masks. WATCH's 128-unit, two-layer unidirectional
LSTM is an architecture/task adaptation. The SN7 method is a retrospective
common-backbone temporal adaptation, not the original ten-model EfficientNet-B3
ensemble. No published full-system score is claimed.

## Correctness and independent use

```sh
uv run ruff check src scripts tests
uv run pytest -q
uv run python -m build
```

The tests cover cumulative shifts, cloud gaps, publication ordering, native grid
failures, available-prefix invariance, gate proposal retention/correction,
append-only historical replay, independent geospatial readback and mutation
rejection at the frozen boundary. Passing them does not establish scientific
performance. The README's query/export commands and
`examples/consume_history.py` demonstrate consumption without the map; final
release evidence must additionally show an independent installation and a changed
input/query executed from it.

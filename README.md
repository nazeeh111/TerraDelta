# TerraDelta

**A source-linked, revisable history of supported building-footprint changes.**

**Research status: failed held-out geographic transfer. This detector is not analyst-ready.**

Import a permitted monthly image sequence, compute persistent additions and removals,
inspect the supporting observations, then query or export the history. Each event
carries geometry, a timing interval, detection availability, source hashes, quality
diagnostics and an explicit revision state.

This is a bounded geospatial ML research project. A cloud gap is an interval, and
unknown publication times are an explicit retrospective assumption. The model
does not establish a physical cause or legal evidence.

## What runs

- Native-grid RGBA SpaceNet 7 ingestion with source UDM cloud masks, CRS/grid checks
  and image-only registration against an immutable anchor and accepted references.
- A locally trained 482,050-parameter U-Net and a compact event transition gate.
  Image predictions propose sites; ordinary persistence creates change hypotheses;
  the learned gate can confirm compatible events using only available observations.
  Labels never propose inference sites or geometry.
- Persistent event identity, proposed/model-confirmed/reviewed/retracted revisions,
  and immutable DuckDB query snapshots.
- GeoJSON, GeoParquet and source-linked probability/validity GeoTIFFs, a typed local
  API/CLI, and an analyst workbench showing the complete study area.
- Controlled temporal comparisons with common imagery and spatial predictions.
  [Protocol and metrics](docs/metrics.md) distinguish online and retrospective methods.

The current spatial revision warm-started from the first model and completed
5,000 additional steps in **15.2 minutes** on Apple MPS. Selected step 4,500 had
footprint Dice **0.4395** and boundary Dice **0.3567** on fixed development crops.
Internal 2× interpolation preserves native output grids and adds no observations.
These are development diagnostics, not held-out event accuracy.

**The frozen three-region confirmation failed.** The quality gate matched only
27 alerts, with **671 false alerts and 1,479 misses**: precision **0.03868**, recall
**0.01793**, F1 **0.02450**. At the common 62-day horizon, only 17 matches remain
and F1 is **0.01543**. The paired no-quality gate has higher ordinary/timely F1
(**0.02603 / 0.01673**), so no overall quality-input benefit is established.
Only **391 of 1,506** truth events overlap an image-derived candidate at IoU >=0.25.

This version is a research artifact with known scientific failure. Its typed
API, revision semantics and independently readable exports do not make its
alerts reliable. The [full result](docs/results-v2.md) includes every method,
complete-region results, masks/exclusions, timing, calibration and uncertainty.
Earlier negative development evidence remains in the
[development record](docs/development-log.md). The source and
[Linux CI](https://github.com/nazeeh111/TerraDelta/actions/runs/41291280215)
are public. The [research release](https://github.com/nazeeh111/TerraDelta/releases/tag/v0.1.0-research)
contains the complete exported histories, source imagery for playback and editable
demo. All 17 downloaded assets match their sizes and SHA-256 digests; each ZIP
passes its integrity check. A [cold public replay](artifacts/verification-v3/public-release-v1.json)
also verifies all revisions, snapshots, source frames and changed-area exports.
These checks verify delivery and software behavior, not detector accuracy.
That released snapshot is `v0.1.0-research`; the current development source is
`0.2.0.dev1` and includes the unfinished scientific revision below.

A separate revision uses a frozen ImageNet ResNet18 encoder with a trained
decoder. Its nine training, four development and three fresh confirmation regions
are declared in [experiment-v8](configs/experiment-v8.json). Development selected
128-pixel tiles and a 0.55 candidate threshold, with object F1 **0.2826** on the
predeclared eight development scenes. Native sequence predictions and new temporal
fits, global development calibration and the
[independently audited freeze](artifacts/verification-v3/confirmation-boundary-v3.md)
are complete. **Fresh confirmation remains deliberately unscored** while
development-only checks address weak proposal coverage and false-alert burden.
The new study also includes a
[frozen neural embedding-distance comparator](docs/embedding-distance.md), with
its protocol fixed before real descriptor extraction and performance selection.
Its [independent review](artifacts/verification-v3/embedding-review-v1.md) records
the newborn-site correction and strict source/input guards.
No fresh confirmation event result is claimed.
All 73 fresh-region native rasters passed
[grid, mask, value and provenance checks](artifacts/verification-v3/native-products-v3.md).
All 12 online methods passed exact prefix/future-edit checks across those regions.
These label-free checks establish data and temporal contracts, not accuracy.
The retained spatial checkpoint has an explicit
[fit recovery audit](artifacts/verification-v3/spatial-v3-fit-audit.md): the
4,000-step loop completed and selected step 3,600, but a final source-integrity
check failed after inference-only files changed. Original checkpoint bytes and
the failure trace are preserved.

## Install

Use Python 3.12 and [uv](https://docs.astral.sh/uv/). Required install targets are
Linux x86-64, Windows x86-64 and Apple Silicon macOS. The lock selects explicit
CPU builds on Linux/Windows and native Mac builds with optional MPS. Windows
wheel resolution is checked; Windows runtime execution has not been tested here.

```sh
git clone https://github.com/nazeeh111/TerraDelta.git
cd TerraDelta
uv sync --frozen --extra ml --extra dev
uv run terradelta --help
```

No imagery account, API key or paid service is required for the bounded public
SpaceNet workflow. [Data and artifact terms](DATA_LICENSE.md) apply.
The released snapshot's [independent install](artifacts/verification-v3/public-install-v1.json) and
[real API/consumer check](artifacts/verification-v3/public-api-v1.json) used an
anonymous public clone and fresh public image/mask bytes, with no private cache.
The current development package also completed
[independent public-only raw computation](artifacts/verification-v3/public-proposals-v020.json)
on two versus three native observations, using newly downloaded official backbone
weights. Those runs verify changed-input execution and readable raster/vector
outputs; they do not evaluate the learned gate or establish event correctness.
The subsequent [installed learned-gate/API replay](artifacts/verification-v3/public-learned-v020.json)
preserved all 1,093 earlier records after a real append and independently verified
280 changed-area/date export records. On that three-observation prefix the main
gate confirmed **zero** events; its no-quality ablation confirmed one. Proposed
and retracted records remain explicitly labeled. This is integration evidence,
not a geographic accuracy result.

## Replay the complete research history

The release data packages preserve the complete Mexico area and all 24 monthly
observations. Playback restores recorded revisions and requires no model or
separate pretrained backbone. Download the Mexico history and UI-input ZIPs plus
`release-assets.json` from the [research release](https://github.com/nazeeh111/TerraDelta/releases/tag/v0.1.0-research).
From the download folder, verify those two selected files:

```sh
uv run --project /path/to/TerraDelta python - <<'PY'
from pathlib import Path
import hashlib, json
assets = {x["name"]: x for x in json.loads(Path("release-assets.json").read_text())["assets"]}
for name in ("terradelta-research-v2-history-mexico.zip", "terradelta-research-v2-mexico-ui-inputs.zip"):
    path, expected = Path(name), assets[name]
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if path.stat().st_size != expected["bytes"] or digest != expected["sha256"]:
        raise SystemExit(f"Verification failed: {name}")
    print(f"Verified: {name}")
PY
```

Extract both archives into the same **new** folder. From its `terradelta-v2` directory:

```sh
uv run --project /path/to/TerraDelta python tools/restore_history.py \
  histories/L15-0457E-1135N_1831_3648_13/revisions.jsonl \
  --output demo/run --batch-size 500 \
  --study-status "Frozen first study: failed geographic transfer. Not analyst-ready."
uv run --project /path/to/TerraDelta terradelta serve \
  --cache-root cache --run-root demo/run --port 8766
```

Replace `/path/to/TerraDelta` with the cloned checkout. Restoration validates every
revision and historical snapshots before publishing a new local ledger. The
public cold full-region restore took **709 seconds**, used **837 MB** peak process
memory and produced a **287 MB** ledger on the shared test host. Existing outputs are preserved. This
replay demonstrates the history contract and source inspection, with the weak
detector results visible.

[Watch the actual-data replay](https://github.com/nazeeh111/TerraDelta/releases/download/v0.1.0-research/terradelta-data-replay-v2.mp4)
or inspect its [sources and limits](docs/media-v2/README.md). The release includes
the editable offline demo bundle and its checksums.

## Import and compute

Fetch one complete frozen region with checksum verification, then import its
native RGBA images and source masks. The example uses a held-out region's imagery;
inference does not load its building labels.

```sh
uv run terradelta fetch-sn7 --output data/sn7 \
  --config configs/experiment-v4.json --manifest configs/data-manifest-v1.json \
  --region L15-0457E-1135N_1831_3648_13
uv run terradelta ingest data/sn7/L15-0457E-1135N_1831_3648_13 \
  --output work/cache/L15-0457E-1135N_1831_3648_13
uv run terradelta run work/cache/L15-0457E-1135N_1831_3648_13 \
  --spatial-weights artifacts/spatial-v2.pt \
  --method observation_gate --event-gate-weights artifacts/event-gate-v1.pt \
  --calibration artifacts/calibration-v2.json --output work/new-run
```

CPU is the portable default. Add `--device mps` on a verified Apple accelerator.
The event-gate route does not require `--temporal-weights`. The earlier learned
occupancy methods remain available for the controlled comparison.
The current source accepts the historical weights only with the explicitly
reviewed spatial-v2 inference grid. Fresh runs receive a new processing identity;
the original frozen study is reproduced from commit `bd33082`.
`--end 2019-06-30` restricts input by availability, not acquisition start. Sequences
with out-of-order publication require an availability replay implementation and
are explicitly rejected. Original SN7 publication timestamps are unknown.

Outputs go to a new directory. Existing histories are preserved. `history/` holds
the ledger, complete revision log and exports; `rasters/` holds native GeoTIFFs.

## Query without the map

```sh
uv run terradelta query work/new-run/history/events.duckdb \
  --bbox=-99.54,19.26,-99.49,19.32 \
  --start 2018-01-01T00:00:00Z --end 2019-12-31T23:59:59Z
uv run terradelta export work/new-run/history/events.duckdb \
  --output work/events.geojson --include-retracted
python examples/consume_history.py work/events.geojson
uv run terradelta serve --cache-root work/cache --run-root work/new-run/history
```

Open `http://127.0.0.1:8766` for the analyst workbench or `/docs` for typed OpenAPI
schemas. Queries use **change-interval overlap**. Replay the returned `snapshot_time`
with `--as-of` or the API's `as_of` parameter to retrieve the same recorded answer.
The independent standard-library consumer above requires neither TerraDelta nor
its map interface.

## Method, evidence and limits

[Architecture](docs/architecture.md) · [Event contract](docs/event-contract.md) ·
[Reproduction](docs/reproduce.md) · [Upstream source review](docs/foundations.md)

The experiment fixes five training, two development and three confirmation AOIs,
with complete acquisition sequences and no adjacent tiles across splits. Training
and held-out region centres are at least 1,489 km apart. Thresholds and timing
padding are selected on development regions globally, then frozen with model,
data, feature and evaluation-code hashes before confirmation scoring.

Monthly matched labels provide interval-censored visibility supervision, sometimes
more precise than the approximately 4 m imagery. A reference footprint is present
at at least 80% building coverage, absent at at most 20%, and otherwise unknown;
at least 80% must be source-valid. Two usable supports establish persistence.
An ID reassignment alone cannot establish removal. Bounded duplicate grouping
retains an audit of the reference sites. Timing diagnostics measure containment
of these label intervals, not coverage of known construction dates.
Two development regions and three confirmation regions cannot establish calibration
under arbitrary geographic shift. Source-valid area/time and all registration
exclusions remain visible; an empty event result is not proof of no physical change.

SpaceNet already tracks buildings, and WATCH already studies event localization.
This project tests an observation-conditioned event gate and useful revision
semantics. The paired no-quality-input gate has the same geometry, raw proposals,
hard usability mask and training budget. The frozen held-out comparison gives that ablation higher pooled ordinary and
timely event F1; learned quality inputs do not establish an overall benefit.
Persistence alone is not a novelty claim. Strong published systems are
identified as task/compute adaptations where appropriate, not reported as exact
full-system reproductions. See the [baseline distinctions](docs/foundations.md).

No land-cover adapter, SAR fusion, global monitoring, individual inference or
synthetic replacement observations are included. The name TerraDelta is also used
by unrelated projects; this repository is specifically `nazeeh111/TerraDelta`.

Code: MIT. SpaceNet imagery/labels and distributed derived data/weights:
CC BY-SA 4.0 with attribution. WATCH architecture attribution is retained in
[licenses/WATCH-MIT.txt](licenses/WATCH-MIT.txt).

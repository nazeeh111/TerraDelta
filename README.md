# TerraDelta

**A source-linked, revisable history of supported building-footprint changes.**

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
- A locally trained 482,050-parameter U-Net and compact observation-conditioned
  temporal model. Image predictions propose sites; labels never propose inference sites.
- Persistent event identity, proposed/model-confirmed/reviewed/retracted revisions,
  and immutable DuckDB query snapshots.
- GeoJSON, GeoParquet and source-linked probability/validity GeoTIFFs, a typed local
  API/CLI, and an analyst workbench showing the complete study area.
- Controlled temporal comparisons with common imagery and spatial predictions.
  [Protocol and metrics](docs/metrics.md) distinguish online and retrospective methods.

The spatial model completed 3,000 training steps in 26.8 minutes on Apple MPS.
Selected step 1,900 had footprint Dice **0.4220** on fixed development crops. This
is a development diagnostic, not a held-out performance claim. The final temporal
comparison is recorded separately in versioned evaluation artifacts.

## Install

Use Python 3.12 and [uv](https://docs.astral.sh/uv/). The lock uses CPU PyTorch on
Linux/Windows and the native Mac build, which can use MPS when available.

```sh
git clone https://github.com/nazeeh111/TerraDelta.git
cd TerraDelta
uv sync --frozen --extra ml --extra dev
uv run terradelta --help
```

No imagery account, API key or paid service is required for the bounded public
SpaceNet workflow. [Data and artifact terms](DATA_LICENSE.md) apply.

## Import and compute

Fetch one complete frozen region with checksum verification, then import its
native RGBA images and source masks. The example uses a held-out region's imagery;
inference does not load its building labels.

```sh
uv run terradelta fetch-sn7 --output data/sn7 \
  --config configs/experiment-v1.json --manifest configs/data-manifest-v1.json \
  --region L15-0457E-1135N_1831_3648_13
uv run terradelta ingest data/sn7/L15-0457E-1135N_1831_3648_13 \
  --output work/cache/L15-0457E-1135N_1831_3648_13
uv run terradelta run work/cache/L15-0457E-1135N_1831_3648_13 \
  --spatial-weights artifacts/spatial-v1.pt \
  --temporal-weights artifacts/temporal-v1.pt \
  --calibration artifacts/calibration-v1.json --output work/new-run
```

CPU is the portable default. Add `--device mps` on a verified Apple accelerator.
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

Monthly matched labels are interval-censored visibility supervision, sometimes
more precise than the approximately 4 m imagery. Timing diagnostics measure
containment of these label intervals, not coverage of known construction dates.
Two development regions and three confirmation regions cannot establish calibration
under arbitrary geographic shift. Source-valid area/time and all registration
exclusions remain visible; an empty event result is not proof of no physical change.

SpaceNet already tracks buildings, and WATCH already studies event localization.
This project tests an observation-conditioned event decoder and useful revision
semantics. Persistence alone is not a novelty claim. Strong published systems are
identified as task/compute adaptations where appropriate, not reported as exact
full-system reproductions. See the [baseline distinctions](docs/foundations.md).

No land-cover adapter, SAR fusion, global monitoring, individual inference or
synthetic replacement observations are included. The name TerraDelta is also used
by unrelated projects; this repository is specifically `nazeeh111/TerraDelta`.

Code: MIT. SpaceNet imagery/labels and distributed derived data/weights:
CC BY-SA 4.0 with attribution. WATCH architecture attribution is retained in
[licenses/WATCH-MIT.txt](licenses/WATCH-MIT.txt).

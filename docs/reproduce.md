# Reproduce the bounded study

Use `uv sync --frozen --extra ml --extra dev`. All dataset source keys and hashes are
in `configs/data-manifest-v1.json`; source imagery/labels are CC BY-SA 4.0. The full
ten-region download is about 1.36 GB. No accounts or paid APIs are required.

The complete sequences were fixed before label inspection. Geographic regions,
not overlapping patches, determine splits. Pixel training crops never cross this
boundary. The original spatial run used a 482,050-parameter U-Net, seed 2901,
batch 16, native 128-pixel crops and 3,000 Apple MPS steps. The selected checkpoint
uses maximum mean footprint/boundary Dice over fixed development crops.

```sh
uv run terradelta fetch-sn7 --output data/sn7 --config configs/experiment-v1.json \
  --manifest configs/data-manifest-v1.json
# Import each configured region with --labels for OFFLINE research caches.
uv run terradelta ingest data/sn7/REGION --output work/cache/REGION --labels
uv run python scripts/train_spatial.py --cache work/cache \
  --config configs/experiment-v1.json --output artifacts/spatial-reproduction.pt \
  --steps 3000 --device auto
uv run python scripts/predict_all.py --cache-root work/cache --output work/study \
  --weights artifacts/spatial-reproduction.pt --config configs/experiment-v1.json \
  --device cpu
uv run python scripts/train_temporal.py --features work/study/features \
  --config configs/experiment-v1.json --output artifacts/temporal-reproduction.pt
uv run python scripts/calibrate.py --features-root work/study/features \
  --cache-root work/cache --weights artifacts/temporal-reproduction.pt \
  --spatial-weights artifacts/spatial-reproduction.pt --config configs/experiment-v1.json \
  --output artifacts/calibration-reproduction.json --freeze-output artifacts/freeze-reproduction.json
uv run python scripts/evaluate.py --features-root work/study/features \
  --cache-root work/cache --weights artifacts/temporal-reproduction.pt \
  --spatial-weights artifacts/spatial-reproduction.pt \
  --calibration artifacts/calibration-reproduction.json --config configs/experiment-v1.json \
  --split confirmation --freeze-manifest artifacts/freeze-reproduction.json \
  --output work/confirmation-reproduction
```

Each region import must finish before prediction. `predict_all.py` requires the
anchored registration protocol produced by current ingestion. Labels are used for
training/dev supervision and later evaluation, never for inference candidates.
Confirmation feature production does not load building labels. Freeze creation
hashes prepared truth bytes without interpreting outcomes; the evaluator checks
all hashes before interpreting confirmation labels.

The controlled comparison uses a shared spatial backbone/sites. WATCH's 128-unit,
two-layer unidirectional LSTM architecture is adapted to these building features
and labels. The SN7 retrospective tracker is a temporal adaptation over common
predictions, not the original ten-model EfficientNet-B3 ensemble. No original
published full-system score is claimed.

Hardware and library numerics can affect training. Native prepared products and
frozen weights identify the reported experiment; a new training run has a new
artifact identity. CPU is supported but training is much slower on this Mac.

Correctness checks:

```sh
uv run ruff check src scripts tests
uv run pytest -q
uv run python -m build
```

These tests include cumulative shifts, cloud gaps, publication ordering, native
grid failures, temporal future-edit invariance, append-only historical replay,
independent geospatial readback and full experiment mutation rejection. Scientific
performance is measured separately; passing unit tests does not establish it.

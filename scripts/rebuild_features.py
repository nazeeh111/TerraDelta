"""Versioned feature regeneration from immutable spatial products, without retraining."""

import argparse
import json
from pathlib import Path
from terradelta.features import prepare_features

parser = argparse.ArgumentParser()
for arg in ["cache-root", "predictions-root", "output", "config"]:
    parser.add_argument("--" + arg, type=Path, required=True)
args = parser.parse_args()
config = json.loads(args.config.read_text())
if args.output.exists() and any(args.output.iterdir()):
    raise ValueError("New feature version requires a fresh output directory")
for split in ["train", "development", "confirmation"]:
    for region in config[split]:
        result = prepare_features(
            args.cache_root / region,
            args.predictions_root / region / "probabilities.npy",
            args.output / region,
            labels=split != "confirmation",
        )
        print(split, region, result, flush=True)

"""Common native spatial predictions and causal sites for frozen region splits."""

import argparse
import json
from pathlib import Path

from terradelta.data import sha256
from terradelta.pipeline import predict_sequence
from terradelta.features import prepare_features

parser = argparse.ArgumentParser()
for arg in ["cache-root", "output", "weights", "config"]:
    parser.add_argument("--" + arg, type=Path, required=True)
parser.add_argument("--device", default="cpu")
parser.add_argument("--tile-size", type=int, default=1024)
args = parser.parse_args()
config = json.loads(args.config.read_text())
for split in ["train", "development", "confirmation"]:
    for aoi in config[split]:
        cache = args.cache_root / aoi
        metadata = json.loads((cache / "metadata.json").read_text())
        if metadata.get("registration_protocol") != "anchor_and_all_accepted_offsets_v2":
            raise ValueError(
                f"Anchor registration QA must complete before feature production: {aoi}"
            )
        prediction = args.output / "predictions" / aoi
        if (prediction / "prediction.json").exists():
            recorded = json.loads((prediction / "prediction.json").read_text())
            if recorded["spatial_weights_sha256"] != sha256(args.weights):
                raise ValueError("Existing spatial product belongs to a different checkpoint")
        else:
            predict_sequence(
                cache, args.weights, prediction, device=args.device, tile_size=args.tile_size
            )
        features = args.output / "features" / aoi
        if not (features / "sites.json").exists():
            report = prepare_features(
                cache, prediction / "probabilities.npy", features, labels=split != "confirmation"
            )
            print("FEATURES", aoi, report, flush=True)

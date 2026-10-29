"""Versioned native raster products from immutable spatial probabilities."""

import argparse
import json
from pathlib import Path
import numpy as np

from terradelta.data import load_region, sha256
from terradelta.pipeline import export_probability_raster

parser = argparse.ArgumentParser()
for arg in ["cache-root", "predictions-root", "weights", "config", "output"]:
    parser.add_argument("--" + arg, type=Path, required=True)
args = parser.parse_args()
config = json.loads(args.config.read_text())
if args.output.exists():
    raise ValueError("New raster product version requires a new output directory")
manifest = []
modelsha = sha256(args.weights)
for split in ["train", "development", "confirmation"]:
    for aoi in config[split]:
        region = load_region(args.cache_root / aoi)
        prediction = args.predictions_root / aoi
        provenance = json.loads((prediction / "prediction.json").read_text())
        if provenance["spatial_weights_sha256"] != modelsha:
            raise ValueError("Wrong spatial model provenance")
        p = np.load(prediction / "probabilities.npy", mmap_mode="r")
        for t, frame in enumerate(region["metadata"]["frames"]):
            path = args.output / aoi / (frame["filename"][:-4] + "_probability.tif")
            export_probability_raster(
                path, p[t], np.asarray(region["valid"][t]), region["metadata"], frame, modelsha
            )
            manifest.append(
                {
                    "path": str(path.relative_to(args.output)),
                    "sha256": sha256(path),
                    "source_id": frame["source_id"],
                    "source_sha256": frame["sha256"],
                }
            )
        print(aoi, "exported", len(p), flush=True)
(args.output / "manifest.json").write_text(
    json.dumps(
        {
            "processing_revision": "native_probability_internal_mask_v2",
            "model_sha256": modelsha,
            "files": manifest,
        },
        indent=2,
    )
    + "\n"
)

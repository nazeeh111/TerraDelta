"""Common native spatial predictions and causal sites for frozen region splits."""

import argparse
import json
from pathlib import Path

from terradelta.data import sha256
from terradelta.pipeline import predict_sequence
from terradelta.features import prepare_features, validate_feature_protocol
from terradelta.event_gate import assert_spatial_provenance
from terradelta.contracts import preflight_implementation, study_binding

parser = argparse.ArgumentParser()
for arg in ["cache-root", "output", "weights", "config"]:
    parser.add_argument("--" + arg, type=Path, required=True)
parser.add_argument("--device", default="cpu")
parser.add_argument("--backbone-weights", type=Path)
parser.add_argument("--tile-size", type=int)
parser.add_argument("--candidate-threshold", type=float)
args = parser.parse_args()
config = json.loads(args.config.read_text())
binding = study_binding(config)
configured_tile = binding["spatial_inference"]["tile_size"] if binding else 1024
tile_size = args.tile_size if args.tile_size is not None else configured_tile
configured_threshold = config.get("candidate_threshold", 0.45)
threshold = args.candidate_threshold if args.candidate_threshold is not None else configured_threshold
if not 0 < threshold < 1:
    raise ValueError("Candidate threshold must be between zero and one")
if binding and (tile_size != configured_tile or threshold != configured_threshold):
    raise ValueError("Prediction settings differ from the versioned development selection")
if binding and threshold != binding["candidate_threshold"]:
    raise ValueError("Candidate threshold differs from the frozen spatial provenance")
if binding and sha256(args.weights) != binding["spatial_weights_sha256"]:
    raise ValueError("Spatial checkpoint differs from the selected model")
actual_backbone = sha256(args.backbone_weights) if args.backbone_weights else None
if binding and actual_backbone != binding["spatial_backbone_sha256"]:
    raise ValueError("Separate backbone differs from the selected model")
if binding:
    preflight_implementation(binding)
regions = [region for split in ("train", "development", "confirmation") for region in config[split]]
if len(regions) != len(set(regions)):
    raise ValueError("Geographic split overlap or duplicate region")
for aoi in regions:
    meta = json.loads((args.cache_root / aoi / "metadata.json").read_text())
    if meta.get("region_id") != aoi:
        raise ValueError("Source cache region differs from declared split")
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
            if recorded.get("region_id") != aoi:
                raise ValueError("Existing spatial product belongs to a different region")
            if recorded["spatial_weights_sha256"] != sha256(args.weights):
                raise ValueError("Existing spatial product belongs to a different checkpoint")
        else:
            recorded = predict_sequence(
                cache,
                args.weights,
                prediction,
                device=args.device,
                tile_size=tile_size,
                backbone_weights=args.backbone_weights,
            )
        if recorded.get("tile_size") != tile_size:
            raise ValueError("Existing predictions use a different tile size")
        if binding:
            assert_spatial_provenance(recorded | {"candidate_threshold": threshold}, binding)
        if sha256(prediction / "probabilities.npy") != recorded["probability_sha256"]:
            raise ValueError("Existing prediction bytes differ from their provenance")
        features = args.output / "features" / aoi
        if not (features / "sites.json").exists():
            report = prepare_features(
                cache, prediction / "probabilities.npy", features,
                labels=split != "confirmation", threshold=threshold
            )
            print("FEATURES", aoi, report, flush=True)
        else:
            info = validate_feature_protocol(features, supervision=split != "confirmation")
            if info.get("region_id") != aoi:
                raise ValueError("Existing feature product belongs to a different region")
            if (info.get("candidate_threshold") != threshold
                    or info.get("probability_sha256") != recorded["probability_sha256"]):
                raise ValueError("Existing features differ from the selected prediction settings")
            if binding:
                assert_spatial_provenance(info, binding)

#!/usr/bin/env python3
"""Extract fit-free ResNet18 descriptors after selected spatial products complete.

New sidecars only: never modifies the existing sites,17Dfeatures or supervision.
CPU extraction can overlap other-region production when authorized. MPS must
wait for the current MPS job to finish. No weight download occurs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

try:
    import resource
except ImportError:
    resource = None

import numpy as np
import torch

from terradelta.contracts import preflight_manifests, study_binding
from terradelta.data import sha256
import terradelta.embeddings as implementation
from terradelta.embeddings import (
    DIMENSIONS, FORMAT, FrozenResNet18, array_sha256, descriptor_plan, encode_scene,
)
from terradelta.event_gate import SPATIAL_PROVENANCE_KEYS
from terradelta.features import FEATURE_NAMES
from terradelta.protocol import semantic_metadata_hash


def source_hashes():
    return {"src/terradelta/embeddings.py": sha256(Path(implementation.__file__)),
            "scripts/extract_embeddings.py": sha256(Path(__file__))}


def peak_resident_bytes():
    if resource is None:
        return None
    import sys
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)


def extract_region(model, cache_dir, feature_dir, output, info, config_hash, device):
    started = time.monotonic()
    if model.test_encoder:
        raise ValueError("random/injected unit encoder cannot produce research descriptors")
    if output.exists():
        raise FileExistsError("embedding output exists; preserve it and use a new version")
    if info.get("feature_names") != FEATURE_NAMES:
        raise ValueError("existing17D feature order differs")
    source_before = source_hashes()
    paths = {"rgb.npy": cache_dir / "rgb.npy", "valid.npy": cache_dir / "valid.npy",
             "sites.npy": feature_dir / "sites.npy", "sites.json": feature_dir / "sites.json",
             "features.npz": feature_dir / "features.npz"}
    inputs = {name: sha256(path) for name, path in paths.items()}
    metadata_hash = semantic_metadata_hash(cache_dir / "metadata.json")
    metadata = json.loads((cache_dir / "metadata.json").read_text())
    rgb, valid, sites = [np.load(paths[name], mmap_mode="r")
                         for name in ("rgb.npy", "valid.npy", "sites.npy")]
    with np.load(paths["features.npz"]) as stored:
        original_usable, births = stored["usable"], stored["births"]
    n, times = original_usable.shape
    if (rgb.dtype != np.uint8 or rgb.shape != (times, 3, *sites.shape)
            or valid.dtype != np.bool_ or valid.shape != (times, *sites.shape)
            or original_usable.dtype != np.bool_ or births.dtype != np.int32
            or births.shape != (n,) or np.any((births < 0) | (births >= times))
            or int(sites.max(initial=0)) != n or len(metadata["frames"]) != times):
        raise ValueError("RGB/valid/site/birth/time cache contracts disagree")
    output.mkdir(parents=True)
    vectors = np.zeros((n, times, DIMENSIONS), np.float32)
    coverage = np.zeros((n, times), np.float32)
    usable = np.zeros((n, times), bool)
    trace = []
    for t in range(times):
        tick = time.monotonic()
        # At discovery the same geometry may query earlier available archive
        # images. Do not erase that observed history using the birth event gate.
        active = original_usable[:, t]
        if active.any():
            values, fraction = encode_scene(model, rgb[t], sites, valid[t], active, device=device)
            vectors[:, t], coverage[:, t] = values, fraction
            usable[:, t] = active & (fraction >= .8) & (np.linalg.norm(values, axis=1) > 1e-12)
        row = {"frame": t, "usable_sites": int(usable[:, t].sum()),
               "seconds": time.monotonic()-tick}
        trace.append(row)
        print(metadata["region_id"], json.dumps(row), flush=True)
    arrays = {"vectors": vectors, "usable": usable, "coverage": coverage, "births": births}
    with (output / "embeddings.npz").open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    if (source_hashes() != source_before
            or any(sha256(path) != inputs[name] for name, path in paths.items())
            or semantic_metadata_hash(cache_dir / "metadata.json") != metadata_hash):
        raise ValueError("source or input changed; partial arrays retained without complete manifest")
    manifest = {
        "format": FORMAT, "complete": True, "region_id": metadata["region_id"],
        "descriptor": descriptor_plan(), "config_sha256": config_hash,
        "spatial_provenance": {key: info[key] for key in SPATIAL_PROVENANCE_KEYS},
        "input_sha256": inputs, "metadata_semantic_sha256": metadata_hash,
        "implementation_sha256": source_before,
        "embeddings_npz_sha256": sha256(output / "embeddings.npz"),
        "array_sha256": {key: array_sha256(value) for key, value in arrays.items()},
        "shape": list(vectors.shape), "device": device, "cpu_threads": 2,
        "elapsed_seconds": time.monotonic()-started, "trace": trace,
        "encoder_and_pool_seconds": sum(row["seconds"] for row in trace),
        "array_bytes": sum(value.nbytes for value in arrays.values()),
        "npz_bytes": (output / "embeddings.npz").stat().st_size,
        "process_peak_resident_bytes": peak_resident_bytes(),
        "peak_scope": "process lifetime through region end; not reset per region; includes encoder startup",
        "source_frames": [{key: frame.get(key) for key in (
            "source_id", "sha256", "uri", "acquisition_start", "acquisition_end",
            "published_at", "availability_assumption") } for frame in metadata["frames"]],
        "rights": {"images": "SpaceNet Partners/Planet; CC-BY-SA-4.0",
                   "backbone": "Separate official ImageNet research input; not redistributed",
                   "notice": "https://github.com/pytorch/vision/blob/v0.24.1/README.md",
                   "dataset_terms": "https://image-net.org/accessagreement"},
    }
    with (output / "embedding.json").open("x") as stream:
        stream.write(json.dumps(manifest, indent=2)+"\n")
    return manifest


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("cache-root", "features-root", "config", "backbone", "output"):
        parser.add_argument("--"+key, type=Path, required=True)
    parser.add_argument("--split", choices=("development", "confirmation", "all"),
                        default="development")
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    args = parser.parse_args()
    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)
    if config.get("embedding_baseline") != descriptor_plan():
        raise ValueError("requires exact predeclared embedding_baseline descriptor in new config")
    binding = study_binding(config)
    if binding is None:
        raise ValueError("embedding baseline requires explicit immutable spatial-site binding")
    if any(set(config.get(a, [])) & set(config.get(b, [])) for a, b in (
        ("train", "development"), ("train", "confirmation"), ("development", "confirmation"))):
        raise ValueError("geographic split overlap")
    regions = config["development"] + config["confirmation"] if args.split == "all" else config[args.split]
    if set(regions) & set(config.get("audit_only", [])):
        raise ValueError("audit-only old confirmation regions are excluded")
    config_hash = sha256(args.config)
    run_receipt = args.output / f"extraction-{args.split}.json"
    if args.output.exists():
        if not args.output.is_dir() or run_receipt.exists():
            raise FileExistsError("existing output/receipt must be preserved")
        # Permit disjoint DEV then CONF batches, never overwrite a region.
        for path in args.output.iterdir():
            if path.is_dir():
                prior = json.loads((path / "embedding.json").read_text())
                if (prior.get("complete") is not True or prior.get("config_sha256") != config_hash
                        or prior.get("descriptor") != descriptor_plan()
                        or prior.get("implementation_sha256") != source_hashes()
                        or path.name not in config["development"] + config["confirmation"]):
                    raise ValueError("existing embedding root is incomplete or from a different study")
            elif path.name not in {"extraction-development.json", "extraction-confirmation.json",
                                  "extraction-all.json"}:
                raise ValueError("output root has an unrecognized existing file")
    if any((args.output / region).exists() for region in regions):
        raise FileExistsError("selected region already exists; no embedding outputs are overwritten")
    # All selected metadata/model bindings settle before any pixel arrays open.
    manifests = preflight_manifests(args.features_root, args.cache_root, regions,
                                    binding, supervision=None)
    torch.set_num_threads(2)
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("requested MPS is unavailable; no silent fallback")
    encoder_started = time.monotonic()
    model = FrozenResNet18(args.backbone).to(args.device).eval()
    encoder_load_seconds = time.monotonic()-encoder_started
    args.output.mkdir(parents=True, exist_ok=True)
    outputs = []
    for region in regions:
        if args.config.read_bytes() != config_bytes:
            raise ValueError("config bytes changed during extraction")
        outputs.append(extract_region(model, args.cache_root / region, args.features_root / region,
                                      args.output / region, manifests[region], config_hash, args.device))
    if args.config.read_bytes() != config_bytes:
        raise ValueError("config bytes changed during extraction")
    with run_receipt.open("x") as stream:
        stream.write(json.dumps({"complete": True, "regions": regions, "config_sha256": config_hash,
                                 "device": args.device, "encoder_load_seconds": encoder_load_seconds,
                                 "total_elapsed_seconds": time.monotonic()-started,
                                 "process_peak_resident_bytes": peak_resident_bytes(),
                                 "array_bytes": sum(m["array_bytes"] for m in outputs),
                                 "npz_bytes": sum(m["npz_bytes"] for m in outputs),
                                 "implementation_sha256": source_hashes()}, indent=2)+"\n")


if __name__ == "__main__":
    main()

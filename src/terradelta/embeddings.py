"""Fit-free frozen neural image descriptors on existing image-derived sites.

No labels, candidate generation, spatial-model decoder or network download.
Layer1 has a native stride4 lattice and theoretical43-pixel receptive field;
bilinear sampling expresses that representation on source pixels, adding no
observations or spatial resolution.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .data import sha256

FORMAT = "terradelta-frozen-embedding-v2"
BACKBONE_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
BACKBONE_URI = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
BACKBONE_ENUM = "ResNet18_Weights.IMAGENET1K_V1"
DIMENSIONS, STRIDE, RECEPTIVE_FIELD = 64, 4, 43
DISTANCE_GRID = (0., .01, .025, .05, .1, .2, .4)


def descriptor_plan() -> dict:
    """Fixed before outcome inspection; no layer/representation search is implied."""
    return {
        "format": FORMAT, "backbone_sha256": BACKBONE_SHA256,
        "backbone_uri": BACKBONE_URI, "backbone_enum": BACKBONE_ENUM,
        "layer": "layer1", "dimensions": DIMENSIONS, "native_input_scale": 1,
        "stride_native_pixels": STRIDE, "receptive_field_native_pixels": RECEPTIVE_FIELD,
        "normalization": "RGB/255; ImageNet mean .485,.456,.406 / std .229,.224,.225",
        "grid": "feature[j,i] centered at native pixel indices [4*j,4*i]",
        "pooling": "bilinear_at_native_centers_border_clamp; fixed_native_index_chunks32768; source-valid site means",
        "mask": "birth-independent original feature usability AND >=.8 valid site coverage AND nonzero vector",
        "site_discovery_history": "At causal discovery, pool that image-derived site over archived prior available images; no future image evidence; shared decoder suppresses all events before birth",
        "distance": "cosine to renormalized coordinate median of last<=3 usable unit descriptors",
        "reference_lookback": 3, "distance_threshold_grid": list(DISTANCE_GRID),
        "state_update": "current ordinary spatial probability when distance>=threshold; otherwise hold previous state",
        "initial_state": "first usable observation initializes from its ordinary spatial probability",
        "decoder": "existing shared two supporting usable observations /20day dwell /quality abstention",
        "decoder_threshold_grid": [.55, .65, .75, .85],
        "selection": "one global DEVELOPMENT distance/decoder point; macro event F1 at62day horizon; tie:ordinary macro F1 then lower distance threshold then lower decoder threshold",
        "confirmation": "no confirmation outcomes until threshold and all source/data/model identities frozen",
    }


def array_sha256(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(json.dumps([value.dtype.str, list(value.shape)]).encode())
    if not value.size:
        return digest.hexdigest()
    raw = memoryview(value).cast("B")
    for start in range(0, len(raw), 1024*1024):
        digest.update(raw[start:start+1024*1024])
    return digest.hexdigest()


def load_embedding_data(directory, *, feature_dir, cache_dir, expected_plan=None,
                        expected_config_sha256=None, expected_extractor_sha256=None):
    """Strict image-only sidecar preflight; no supervision/truth files are opened."""
    from .event_gate import SPATIAL_PROVENANCE_KEYS, assert_spatial_provenance
    from .features import validate_feature_protocol
    from .protocol import semantic_metadata_hash

    directory, feature_dir, cache_dir = map(Path, (directory, feature_dir, cache_dir))
    manifest = json.loads((directory / "embedding.json").read_text())
    if (manifest.get("format") != FORMAT or manifest.get("complete") is not True
            or manifest.get("descriptor") != (expected_plan or descriptor_plan())):
        raise ValueError("embedding sidecar is incomplete or has a different descriptor protocol")
    if expected_config_sha256 and manifest.get("config_sha256") != expected_config_sha256:
        raise ValueError("embedding sidecar config differs from experiment")
    info = validate_feature_protocol(feature_dir)
    metadata = json.loads((cache_dir / "metadata.json").read_text())
    if (manifest.get("region_id") != info.get("region_id")
            or metadata.get("region_id") != info.get("region_id")):
        raise ValueError("embedding sidecar region differs from spatial sites/cache")
    assert_spatial_provenance(info, manifest["spatial_provenance"])
    if manifest["spatial_provenance"] != {key: info[key] for key in SPATIAL_PROVENANCE_KEYS}:
        raise ValueError("embedding/spatial provenance mismatch")
    if (manifest.get("metadata_semantic_sha256") != semantic_metadata_hash(cache_dir / "metadata.json")
            or manifest["metadata_semantic_sha256"] != info.get("metadata_semantic_sha256")):
        raise ValueError("embedding observation metadata differs from source cache")
    paths = {"rgb.npy": cache_dir / "rgb.npy", "valid.npy": cache_dir / "valid.npy",
             "sites.npy": feature_dir / "sites.npy", "sites.json": feature_dir / "sites.json",
             "features.npz": feature_dir / "features.npz"}
    if set(manifest.get("input_sha256", {})) != set(paths):
        raise ValueError("embedding sidecar is missing source/site identities")
    if any(sha256(path) != manifest["input_sha256"][name] for name, path in paths.items()):
        raise ValueError("embedding input pixels/masks/sites/features changed")
    source_fields = ("source_id", "sha256", "uri", "acquisition_start", "acquisition_end",
                     "published_at", "availability_assumption")
    if manifest.get("source_frames") != [{key: frame.get(key) for key in source_fields}
                                        for frame in metadata["frames"]]:
        raise ValueError("embedding frame/source identities differ from source metadata")
    extractor = Path(__file__).resolve().parents[2] / "scripts/extract_embeddings.py"
    if expected_extractor_sha256 is None:
        if not extractor.is_file():
            raise ValueError("installed loading requires caller-pinned expected_extractor_sha256")
        expected_extractor_sha256 = sha256(extractor)
    if (not isinstance(expected_extractor_sha256, str) or len(expected_extractor_sha256) != 64
            or any(c not in "0123456789abcdef" for c in expected_extractor_sha256)):
        raise ValueError("expected extractor identity must be a full lowercase SHA256")
    expected_sources = {"src/terradelta/embeddings.py": sha256(Path(__file__)),
                        "scripts/extract_embeddings.py": expected_extractor_sha256}
    if manifest.get("implementation_sha256") != expected_sources:
        raise ValueError("embedding implementation differs from extracted sidecar")
    if sha256(directory / "embeddings.npz") != manifest.get("embeddings_npz_sha256"):
        raise ValueError("embedding sidecar array file changed")
    with np.load(directory / "embeddings.npz") as stored:
        data = {key: stored[key] for key in ("vectors", "usable", "coverage", "births")}
    with np.load(feature_dir / "features.npz") as stored:
        original_usable, original_births = stored["usable"], stored["births"]
    vectors, usable, coverage, births = [data[k] for k in ("vectors", "usable", "coverage", "births")]
    if (vectors.shape != (*original_usable.shape, DIMENSIONS) or vectors.dtype != np.float32
            or usable.shape != original_usable.shape or usable.dtype != np.bool_
            or coverage.shape != usable.shape or coverage.dtype != np.float32
            or not np.array_equal(births, original_births) or births.dtype != np.int32
            or not np.isfinite(vectors).all() or not np.isfinite(coverage).all()
            or np.any((coverage < 0) | (coverage > 1))
            or np.any(usable & ~(original_usable & (coverage >= .8)))):
        raise ValueError("embedding shapes, birth identities or hard validity differ from spatial sites")
    if any(array_sha256(value) != manifest.get("array_sha256", {}).get(key)
           for key, value in data.items()):
        raise ValueError("embedding vector/mask content hash differs")
    data["manifest"] = manifest
    return data


class FrozenResNet18(nn.Module):
    """Native RGB[0,1] -> frozen64D layer1 map; encoder injection is unit-only."""

    def __init__(self, backbone_path: str | Path | None = None, *, encoder: nn.Module | None = None):
        super().__init__()
        self.test_encoder = encoder is not None
        if encoder is None:
            if backbone_path is None or sha256(Path(backbone_path)) != BACKBONE_SHA256:
                raise ValueError("local official ResNet18 full SHA256 mismatch")
            from torchvision.models import resnet18

            encoder = resnet18(weights=None)
            encoder.load_state_dict(torch.load(backbone_path, map_location="cpu", weights_only=True),
                                    strict=True)  # Includes fc; no missing-key fallback.
        self.encoder = encoder.requires_grad_(False).eval()
        self.register_buffer("mean", torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([.229, .224, .225])[None, :, None, None])

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()  # Freezes BatchNorm statistics as well as parameters.
        return self

    @torch.inference_mode()
    def forward(self, rgb):
        if rgb.ndim != 4 or rgb.shape[1] != 3 or min(rgb.shape[-2:]) < 1:
            raise ValueError("expected nonempty native RGB[batch,3,height,width]")
        value = (rgb - self.mean) / self.std
        value = self.encoder.relu(self.encoder.bn1(self.encoder.conv1(value)))
        return self.encoder.layer1(self.encoder.maxpool(value))


@torch.inference_mode()
def pool_sites(feature_map, sites, valid, active, *, chunk_pixels=32768):
    """Pool source-valid site pixels, including archived pre-discovery observations.

    Sampling uses the actual convolution lattice centers, not resize-to-shape.
    Sampling batches cover fixed native-index intervals independent of candidate
    inventory. Within each batch, each site's native pixels reduce in raster order.
    Adding future-born sites cannot shift old sites' interpolation/reduction order.
    """
    sites, valid, active = np.asarray(sites), np.asarray(valid), np.asarray(active)
    if (sites.ndim != 2 or sites.dtype.kind not in "iu" or valid.shape != sites.shape
            or valid.dtype != np.bool_ or active.ndim != 1 or active.dtype != np.bool_
            or np.any(sites < 0) or int(sites.max(initial=0)) > len(active)):
        raise ValueError("sites/valid/active grid or immutable site IDs are invalid")
    if chunk_pixels < 1 or feature_map.ndim != 4 or feature_map.shape[0] != 1:
        raise ValueError("feature map must have batch1 and a positive pooling chunk")
    height, width = sites.shape
    if tuple(feature_map.shape[-2:]) != ((height + STRIDE - 1)//STRIDE,
                                        (width + STRIDE - 1)//STRIDE):
        raise ValueError("feature map does not match the native stride4 grid")
    n, channels = len(active), int(feature_map.shape[1])
    area = np.bincount(sites.ravel(), minlength=n+1)[1:]
    active_lookup = np.r_[False, active]
    selected = valid.ravel() & active_lookup[sites.ravel()]
    sums = np.zeros((n+1, channels), np.float64)
    counts = np.bincount(sites.ravel()[selected], minlength=n+1)
    fh, fw = feature_map.shape[-2:]
    for start in range(0, sites.size, chunk_pixels):
        # Include background coordinates so sample shape/order do not depend on
        # whether a future-born candidate occupies another pixel in this chunk.
        indices = np.arange(start, min(start+chunk_pixels, sites.size))
        local_valid = selected[indices]
        if not local_valid.any():
            continue
        y, x = np.divmod(indices, width)
        # Conv/pool padding fixes lattice center0 at source pixel center0.
        gx = np.zeros_like(x, dtype=np.float32) if fw == 1 else 2*(x/STRIDE)/(fw-1)-1
        gy = np.zeros_like(y, dtype=np.float32) if fh == 1 else 2*(y/STRIDE)/(fh-1)-1
        grid = torch.from_numpy(np.stack((gx, gy), axis=-1).astype(np.float32))[None, None]
        sample = F.grid_sample(feature_map, grid.to(feature_map.device), mode="bilinear",
                               padding_mode="border", align_corners=True)[0, :, 0].cpu().numpy()
        ids = sites.ravel()[indices[local_valid]]
        for channel in range(channels):
            sums[:, channel] += np.bincount(ids, weights=sample[channel, local_valid], minlength=n+1)
    result = (sums[1:] / np.maximum(counts[1:, None], 1)).astype(np.float32)
    coverage = (counts[1:] / np.maximum(area, 1)).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("nonfinite frozen embedding")
    return result, coverage


@torch.inference_mode()
def encode_scene(model, rgb, sites, valid, active, *, device="cpu", chunk_pixels=32768):
    if rgb.dtype != np.uint8 or rgb.shape != (3, *sites.shape):
        raise ValueError("native RGB must be uint8[3,height,width] on the existing site grid")
    model.eval()
    value = torch.from_numpy(np.ascontiguousarray(rgb, dtype=np.float32)/255)[None].to(device)
    return pool_sites(model(value), sites, valid, active, chunk_pixels=chunk_pixels)


def causal_distance(embeddings, usable, births, *, lookback=3):
    """Cosine distance uses prior usable observations only; no geographic fit."""
    z, usable, births = np.asarray(embeddings), np.asarray(usable), np.asarray(births)
    if (z.ndim != 3 or usable.shape != z.shape[:2] or usable.dtype != np.bool_
            or births.shape != (len(z),) or births.dtype.kind not in "iu" or lookback < 1
            or np.any(births < 0) or not np.isfinite(z[usable]).all()):
        raise ValueError("expected finite[site,time,embedding] plus usable mask and births")
    distances = np.zeros(usable.shape, np.float32)
    # Discovery authorizes looking back at available archived images at this
    # fixed geometry, as in the common17D feature cube. Birth is a publication
    # gate in the shared decoder, not a reason to erase observed prior absence.
    available = usable.copy()
    for site in range(len(z)):
        history = []
        for t in range(z.shape[1]):
            if not available[site, t]:
                continue
            vector = np.asarray(z[site, t], np.float64)
            norm = np.linalg.norm(vector)
            if norm <= 1e-12:
                available[site, t] = False
                continue
            vector = vector/norm
            if history:
                reference = np.median(history[-lookback:], axis=0)
                length = np.linalg.norm(reference)
                # A zero reference has no direction; abstain without inventing distance.
                if length <= 1e-12:
                    available[site, t] = False
                    history.append(vector)  # Keep the usable image; future references may recover.
                    continue
                distances[site, t] = np.clip(1-np.dot(vector, reference/length), 0, 2)
            history.append(vector)
    return distances, available


def distance_gated_scores(probability, embeddings, usable, births, threshold, *, lookback=3):
    """Fit-free baseline; unsigned distance gates updates, spatial probability gives state.

    A held value is a temporal filter state, not new neural evidence. The caller
    must keep the returned hard usability mask and ordinary decoder semantics.
    """
    probability = np.asarray(probability)
    if (probability.shape != np.shape(usable) or not np.isfinite(probability).all()
            or np.any((probability < 0) | (probability > 1))):
        raise ValueError("ordinary spatial probability must match[site,time] usability")
    if not np.isfinite(threshold) or not 0 <= threshold <= 2:
        raise ValueError("cosine distance threshold must be in[0,2]")
    distance, available = causal_distance(embeddings, usable, births, lookback=lookback)
    score = np.full(probability.shape, .5, np.float32)
    for site in range(len(score)):
        initialized, previous = False, .5
        for t in range(score.shape[1]):
            if available[site, t] and (not initialized or distance[site, t] >= threshold):
                previous, initialized = float(probability[site, t]), True
            score[site, t] = previous
    return score, available, distance

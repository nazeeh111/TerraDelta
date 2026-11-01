"""Optional neural comparison: real decoder causality and strict identity guards."""

from copy import deepcopy

import numpy as np
import pytest

from test_gate_pipeline import ORIGIN, fixture_inputs
from terradelta import pipeline
from terradelta.embeddings import descriptor_plan


def inputs():
    x, usable, births, info = fixture_inputs()
    binding = {
        "spatial_weights_sha256": "a" * 64, "spatial_backbone_sha256": None,
        "candidate_threshold": .55,
        "spatial_inference": {
            "tile_size": 128, "context_halo_native_pixels": 64,
            "native_alignment": 8, "internal_input_scale": 2,
            "implementation_sha256": {"spatial.py": "b" * 64},
        },
    }
    info.update(binding)
    info["metadata_semantic_sha256"] = "c" * 64
    vectors = np.zeros((1, x.shape[1], 64), np.float32)
    for t in range(x.shape[1]):
        vectors[0, t, int(x[0, t, 0] > .5)] = 1
    sidecar = {"vectors": vectors, "usable": usable.copy(), "births": births.copy(),
               "manifest": {"descriptor": descriptor_plan(), "config_sha256": "d" * 64,
                            "region_id": info["region_id"], "spatial_provenance": binding,
                            "metadata_semantic_sha256": "c" * 64,
                            "implementation_sha256": {"src/terradelta/embeddings.py": "e" * 64},
                            "embeddings_npz_sha256": "f" * 64}}
    calibration = {
        "embedding_distance": {"descriptor": descriptor_plan(), "distance_threshold": .05,
                               "reference_lookback": 3},
        "config_sha256": "d" * 64, "spatial_provenance": binding,
        "occupancy": {"embedding_distance": {"scale": 1., "bias": 0.}},
        "decoders": {"embedding_distance": {"threshold": .65}},
    }
    return x, usable, births, info, sidecar, calibration


def run(x, usable, births, info, sidecar, calibration):
    return pipeline.method_events(x, usable, births, info, "embedding_distance",
                                  embedding_data=sidecar, calibration=calibration,
                                  record_origin=ORIGIN)


def test_embedding_dispatch_exact_prefix_and_future_edit_invariance():
    x, usable, births, info, sidecar, calibration = inputs()
    full = run(x, usable, births, info, sidecar, calibration)
    assert any(e.state == "confirmed_by_model" for e in full)
    n = 4
    prefix_info = deepcopy(info)
    prefix_info["metadata"]["frames"] = info["metadata"]["frames"][:n]
    prefix = deepcopy(sidecar)
    prefix["vectors"] = prefix["vectors"][:, :n]
    prefix["usable"] = prefix["usable"][:, :n]
    prefix["manifest"]["embeddings_npz_sha256"] = "1" * 64
    early = run(x[:, :n], usable[:, :n], births, prefix_info, prefix, calibration)
    assert early
    assert [e.model_dump(mode="json") for e in early] == [
        e.model_dump(mode="json") for e in full[:len(early)]
    ]
    changed = deepcopy(sidecar)
    changed["vectors"][:, n:] = -changed["vectors"][:, n:]
    changed["manifest"]["embeddings_npz_sha256"] = "2" * 64
    edited = run(x, usable, births, info, changed, calibration)
    assert [e.model_dump(mode="json") for e in early] == [
        e.model_dump(mode="json") for e in edited[:len(early)]
    ]


@pytest.mark.parametrize("damage", ["source", "config", "births", "distance", "dwell", "lookback"])
def test_embedding_dispatch_rejects_mixed_inputs_or_undeclared_operating_point(damage):
    x, usable, births, info, sidecar, calibration = inputs()
    if damage == "source":
        sidecar["manifest"]["metadata_semantic_sha256"] = "0" * 64
    elif damage == "config":
        sidecar["manifest"]["config_sha256"] = "0" * 64
    elif damage == "births":
        sidecar["births"] = np.array([1])
    elif damage == "distance":
        calibration["embedding_distance"]["distance_threshold"] = .123
    elif damage == "lookback":
        calibration["embedding_distance"]["reference_lookback"] = 3.0
    else:
        calibration["decoders"]["embedding_distance"]["confirmations"] = 1
    with pytest.raises(ValueError):
        run(x, usable, births, info, sidecar, calibration)


def test_embedding_all_missing_never_becomes_observed_absence():
    x, usable, births, info, sidecar, calibration = inputs()
    sidecar["usable"][:] = False
    assert run(x, usable, births, info, sidecar, calibration) == []

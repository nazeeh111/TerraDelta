"""Preflight model and observation identities before labels, compute or scoring."""

from pathlib import Path
import json
import math
from dataclasses import asdict

from .data import sha256

TEMPORAL_METHODS = {"observation", "no_quality", "watch_lstm"}
LEARNED_GATE_METHODS = {"observation_gate", "gate_no_quality"}


def head_training_config_sha256(config_path, settings=None):
    """Resolve a bounded comparison-only amendment without relabeling fitted heads.

    Both head trainers ignore the embedding block. Every other configuration
    field, including unknown future fields, must match the actual training file.
    Data-byte identity remains recorded separately in the actual fit receipts.
    """
    config_path = Path(config_path)
    settings = json.loads(config_path.read_text()) if settings is None else settings
    reference = settings.get("head_training_config")
    if reference is None:
        return sha256(config_path)
    if (not isinstance(reference, dict) or set(reference) != {"path", "sha256"}
            or not isinstance(reference["path"], str)
            or Path(reference["path"]).name != reference["path"]
            or Path(reference["path"]).suffix != ".json"):
        raise ValueError("Head training config must be an explicitly hashed sibling JSON file")
    training_path = config_path.with_name(reference["path"])
    if (training_path.resolve() == config_path.resolve()
            or sha256(training_path) != reference["sha256"]):
        raise ValueError("Head training config reference changed or refers to itself")
    training = json.loads(training_path.read_text())
    if "head_training_config" in training:
        raise ValueError("Chained head-training configuration references are unsupported")
    comparison_only = {
        "version", "embedding_baseline", "embedding_baseline_declared_at",
        "embedding_amendment", "head_training_config",
    }
    def canonical(value):
        return json.dumps({k: v for k, v in value.items() if k not in comparison_only},
                          sort_keys=True, separators=(",", ":"), allow_nan=False)

    if canonical(settings) != canonical(training):
        raise ValueError("Comparison amendment changed the actual head training configuration")
    return reference["sha256"]


def preflight_probability_files(features_root, predictions_root, regions):
    """Bind actual diagnostic raster arrays before any truth is opened or hashed."""
    from .features import validate_feature_protocol

    for region in regions:
        info = validate_feature_protocol(Path(features_root) / region, supervision=False)
        path = Path(predictions_root) / region / "probabilities.npy"
        if sha256(path) != info.get("probability_sha256"):
            raise ValueError(f"Diagnostic probability bytes differ from frozen features: {region}")


def study_binding(config):
    from .event_gate import validate_spatial_provenance

    binding = config.get("event_gate", {}).get("spatial_provenance")
    if binding is None:
        if config.get("spatial_backbone") is not None:
            raise ValueError("pretrained study requires explicit spatial_provenance")
        return None
    binding = validate_spatial_provenance(binding)
    if config.get("spatial_backbone") is not None and (
        binding["spatial_backbone_sha256"] != config["spatial_backbone"].get("sha256")
    ):
        raise ValueError("backbone binding differs from study")
    if config.get("candidate_threshold", binding["candidate_threshold"]) != binding["candidate_threshold"]:
        raise ValueError("candidate policy differs from study binding")
    return binding


def preflight_manifests(features_root, cache_root, regions, binding, *, supervision):
    from .event_gate import assert_spatial_provenance
    from .features import validate_feature_protocol
    from .protocol import semantic_metadata_hash

    if len(regions) != len(set(regions)):
        raise ValueError("duplicate geographic region")
    manifests = {}
    for region in regions:
        info = validate_feature_protocol(Path(features_root) / region, supervision=supervision)
        metadata_path = Path(cache_root) / region / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        if info.get("region_id") != region or metadata.get("region_id") != region:
            raise ValueError("region identity differs from declared split")
        if binding is not None:
            assert_spatial_provenance(info, binding)
        if info.get("metadata_semantic_sha256") != semantic_metadata_hash(metadata_path):
            raise ValueError("feature observation metadata differs from source cache")
        manifests[region] = info
    return manifests


def preflight_implementation(binding):
    from .event_gate import validate_spatial_provenance

    binding = validate_spatial_provenance(binding)
    for name, expected in binding["spatial_inference"]["implementation_sha256"].items():
        if sha256(Path(__file__).with_name(name)) != expected:
            raise ValueError("spatial inference implementation differs from frozen binding")


def preflight_spatial_checkpoint(path, binding=None):
    import torch
    from .spatial import load_model

    state = torch.load(path, map_location="cpu", weights_only=True)
    backbone = None
    if state.get("format") == "terradelta-pretrained-spatial-v1":
        from .pretrained_spatial import BACKBONE_ENUM, BACKBONE_URI, PretrainedBuildingUNet

        architecture = state["architecture"]
        backbone = state["backbone"]["sha256"]
        if (not isinstance(backbone, str) or len(backbone) != 64
                or any(character not in "0123456789abcdef" for character in backbone)):
            raise ValueError("pretrained checkpoint requires a full backbone SHA256")
        if (architecture.get("name") != "FrozenResNet18GroupNormDecoder"
                or architecture.get("group_norm_groups") != 8
                or architecture.get("outputs") != ["footprint", "boundary"]
                or state["backbone"].get("enum") != BACKBONE_ENUM
                or state["backbone"].get("uri") != BACKBONE_URI):
            raise ValueError("unsupported pretrained spatial checkpoint identity")
        # Decoder structure can be validated without loading or downloading the
        # separate frozen backbone. Actual inference SHA-checks that file too.
        model = PretrainedBuildingUNet(encoder=torch.nn.Identity(),
                                      input_scale=architecture["input_scale"])
        model.decoder.load_state_dict(state["decoder_state_dict"], strict=True)
    else:
        model = load_model(path)
    if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise ValueError("nonfinite spatial parameters")
    identity = {"spatial_weights_sha256": sha256(path), "spatial_backbone_sha256": backbone}
    if binding is not None:
        if any(identity[key] != binding[key] for key in identity):
            raise ValueError("spatial checkpoint identity differs from study binding")
        if (binding["spatial_inference"]["native_alignment"] != getattr(model, "native_alignment", 8)
                or binding["spatial_inference"]["internal_input_scale"] != model.input_scale):
            raise ValueError("spatial checkpoint architecture differs from inference binding")
    return identity


def preflight_heads(info, *, temporal_weights=None, event_gate_weights=None,
                    temporal_kinds=(), require_bound=False, config_sha256=None):
    """Load real state dictionaries, never trust a checkpoint header alone."""
    import torch
    from .event_gate import assert_spatial_provenance, load_gate
    from .features import FEATURE_NAMES
    from .temporal import load_temporal

    if temporal_kinds:
        if temporal_weights is None:
            raise ValueError("selected temporal methods require trained weights")
        state = torch.load(temporal_weights, map_location="cpu", weights_only=True)
        if require_bound:
            if state.get("spatial_provenance") is None:
                raise ValueError("temporal checkpoint is missing its spatial binding")
            assert_spatial_provenance(info, state["spatial_provenance"])
            if state.get("feature_names") != FEATURE_NAMES:
                raise ValueError("temporal checkpoint feature order differs")
            if config_sha256 and state.get("config_sha256") != config_sha256:
                raise ValueError("temporal training config differs from study")
        for kind in temporal_kinds:
            if kind not in TEMPORAL_METHODS:
                raise ValueError("unsupported learned temporal kind")
            model = load_temporal(temporal_weights, kind, spatial_info=info)
            if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
                raise ValueError("nonfinite temporal parameters")
    if event_gate_weights is not None:
        gate = load_gate(event_gate_weights)
        if gate.spatial_weights_sha256 != info.get("spatial_weights_sha256"):
            raise ValueError("gate spatial checkpoint differs from feature source")
        if require_bound:
            if gate.spatial_provenance is None:
                raise ValueError("gate checkpoint is missing its spatial binding")
            assert_spatial_provenance(info, gate.spatial_provenance)
        elif gate.spatial_provenance is not None:
            assert_spatial_provenance(info, gate.spatial_provenance)
        if config_sha256:
            state = torch.load(event_gate_weights, map_location="cpu", weights_only=True)
            if state.get("training", {}).get("config_sha256") != config_sha256:
                raise ValueError("gate training config differs from study")
        if any(not torch.isfinite(parameter).all()
               for model in gate.models.values() for parameter in model.parameters()):
            raise ValueError("nonfinite gate parameters")


def assert_calibration(info, calibration, *, method=None, temporal_weights=None,
                       event_gate_weights=None, require_bound=False, config_sha256=None):
    """Validate every supplied identity; strict study/CLI paths require bindings."""
    from .event_gate import assert_spatial_provenance

    binding = calibration.get("spatial_provenance")
    versioned = any(key in calibration for key in (
        "version", "config_sha256", "weights_sha256", "event_gate_weights_sha256",
        "spatial_weights_sha256", "spatial_provenance",
    ))
    if binding is not None:
        assert_spatial_provenance(info, binding)
    elif require_bound or (versioned and info.get("spatial_backbone_sha256") is not None):
        raise ValueError("calibration is missing its spatial binding")
    if "spatial_weights_sha256" in calibration or require_bound:
        if calibration.get("spatial_weights_sha256") != info.get("spatial_weights_sha256"):
            raise ValueError("calibration spatial checkpoint differs from input")
    if config_sha256 and calibration.get("config_sha256") != config_sha256:
        raise ValueError("calibration config differs from study")
    for key, path, relevant in (
        ("weights_sha256", temporal_weights, method is None or method in TEMPORAL_METHODS),
        ("event_gate_weights_sha256", event_gate_weights,
         method is None or method in LEARNED_GATE_METHODS),
    ):
        if relevant and path is not None and (key in calibration or require_bound or versioned):
            if calibration.get(key) != sha256(path):
                raise ValueError(f"calibration {key} differs from selected head")
    if versioned or require_bound or method == "embedding_distance":
        preflight_calibration_operations(calibration, method)


def preflight_calibration_operations(calibration, method=None):
    """Validate parameters the decoder will actually use before truth is opened."""
    from .pipeline import METHODS, _method_decoder

    def finite_number(value):
        return type(value) in (int, float) and math.isfinite(value)

    selected_methods = METHODS if method is None else (method,)
    if method is None and "embedding_distance" in calibration:
        selected_methods = (*selected_methods, "embedding_distance")
    for selected in selected_methods:
        if selected == "embedding_distance":
            from .embeddings import descriptor_plan

            params = calibration.get("embedding_distance", {})
            plan = descriptor_plan()
            if (params.get("descriptor") != plan
                    or not finite_number(params.get("distance_threshold"))
                    or params["distance_threshold"] not in plan["distance_threshold_grid"]
                    or type(params.get("reference_lookback")) is not int
                    or params.get("reference_lookback") != plan["reference_lookback"]):
                raise ValueError("invalid neural embedding operating point or descriptor")
        if selected in {"proposal_tracking", *LEARNED_GATE_METHODS}:
            params = calibration.get("event_gates", {}).get(selected)
            if not isinstance(params, dict):
                raise ValueError(f"missing operational event calibration: {selected}")
            if selected in LEARNED_GATE_METHODS and (
                any(not finite_number(params.get(key)) for key in ("scale", "bias", "threshold"))
                or params["scale"] < 0 or not 0 <= params["threshold"] <= 1
            ):
                raise ValueError(f"invalid event calibration parameters: {selected}")
            padding = params.get("timing_padding_days")
            if type(padding) is not int or padding < 0:
                raise ValueError(f"invalid event timing padding: {selected}")
        else:
            fit = calibration.get("occupancy", {}).get(selected)
            if (not isinstance(fit, dict) or any(not finite_number(fit.get(key))
                                                for key in ("scale", "bias")) or fit["scale"] < 0):
                raise ValueError(f"missing or invalid occupancy calibration: {selected}")
            if not isinstance(calibration.get("decoders", {}).get(selected,
                              calibration.get("decoder")), dict):
                raise ValueError(f"missing decoder operating point: {selected}")
        try:
            decoder = _method_decoder(selected, calibration)
            values = asdict(decoder)
            if not finite_number(values.pop("threshold")) or any(type(v) is not int for v in values.values()):
                raise ValueError("decoder counts/durations must be integers")
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid decoder operating point: {selected}") from error

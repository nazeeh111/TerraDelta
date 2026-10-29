"""Geographically scoped event evaluation with interval-censored visibility labels.

Ground-truth footprints are used only here, never to create model candidates.
Timing metrics compare observation support, not unknown physical event dates.
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from pyproj import Geod
from rasterio.features import shapes
from rasterio.transform import Affine
from rasterio.warp import transform_geom
from scipy import sparse
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from shapely.geometry import Polygon, shape
from shapely.geometry.polygon import orient
from shapely.strtree import STRtree

from .schema import EventRevision, utc_datetime

_GEOD = Geod(ellps="WGS84")
_DAY = 86400.0


def _date(value: datetime | str) -> datetime:
    return utc_datetime(value)


def _area(geometry: Any) -> float:
    """Geodesic area for polygonal intersections, including collections."""
    if geometry.is_empty:
        return 0.0
    if isinstance(geometry, Polygon):
        return abs(float(_GEOD.geometry_area_perimeter(orient(geometry, sign=1.0))[0]))
    if hasattr(geometry, "geoms"):
        return sum(_area(part) for part in geometry.geoms)
    return 0.0


def geometry_iou(first: Any, second: Any) -> float:
    intersection = _area(first.intersection(second))
    union = _area(first) + _area(second) - intersection
    return intersection / union if union > 0 else 0.0


def _join_parts(parts: list[dict]) -> dict:
    if len(parts) == 1:
        return parts[0]
    polygons = []
    for part in parts:
        if part["type"] == "Polygon":
            polygons.append(part["coordinates"])
        elif part["type"] == "MultiPolygon":
            polygons.extend(part["coordinates"])
        else:
            raise ValueError("rasterized truth must produce polygonal geometry")
    return {"type": "MultiPolygon", "coordinates": polygons}


def ground_truth_events(
    cache: str | Path, *, protocol: str = "physical_coverage_v2", diagnostics: dict | None = None
) -> list[dict[str, Any]]:
    """Find persistent geometric presence changes on source-ID reference footprints.

    Two usable observations establish a new state; masked months do not count as
    absence or reset pending support. Initial inventory is not an addition.
    Physical-coverage v2 tests ANY building label, requiring >=80% coverage for
    presence and <=20% for absence. The middle range is unknown. The preserved
    matched_id_v1 audit instead tests membership of a particular source ID.
    Observability requires >=80% source-valid coverage of the first label
    footprint. Model registration rejection does not remove a truth event.
    Reference pixel pooling is sparse/vectorized, not one full raster scan per ID.
    """
    if protocol not in {"physical_coverage_v2", "matched_id_v1"}:
        raise ValueError("Unknown truth protocol")
    cache = Path(cache)
    metadata = json.loads((cache / "metadata.json").read_text())
    instances = np.load(cache / "instances.npy", mmap_mode="r")
    valid = np.load(cache / "valid.npy", mmap_mode="r")
    frames = metadata["frames"]
    if instances.shape != valid.shape or len(instances) != len(frames):
        raise ValueError("truth instances, source masks and metadata grids must agree")
    if instances.ndim != 3 or not np.issubdtype(instances.dtype, np.integer):
        raise ValueError("truth requires time x row x column integer source IDs")
    if int(instances.min()) < 0:
        raise ValueError("source IDs must be nonnegative; zero is background")
    max_id = int(instances.max())
    if max_id == 0:
        if diagnostics is not None:
            diagnostics.update(
                {
                    "reference_count": 0,
                    "total_reference_frames": 0,
                    "source_valid_reference_frames": 0,
                    "source_unusable_reference_frames": 0,
                    "ambiguous_coverage_reference_frames": 0,
                    "initial_inventory_references": 0,
                    "initial_absent_references": 0,
                    "no_usable_state_references": 0,
                    "right_censored_pending_references": 0,
                    "zero_event_references": 0,
                    "references": [],
                }
            )
        return []
    height, width = instances.shape[1:]
    n_pixels = height * width
    seen = np.zeros(max_id + 1, dtype=bool)
    seen[0] = True
    reference_rows: list[np.ndarray] = []
    reference_columns: list[np.ndarray] = []
    geometry_parts: dict[int, list[dict]] = {}
    transform = Affine(*metadata["transform"])
    # Each raster is scanned once; first footprints may overlap other IDs' later
    # footprints, so a sparse object/pixel matrix preserves independent supports.
    for labels in instances:
        flattened = np.asarray(labels).ravel()
        new_pixels = ~seen[flattened]
        if not new_pixels.any():
            continue
        columns = np.flatnonzero(new_pixels)
        rows = flattened[columns]
        reference_rows.append(rows)
        reference_columns.append(columns)
        new_mask = new_pixels.reshape(height, width)
        for geometry, value in shapes(np.asarray(labels), mask=new_mask, transform=transform):
            source_id = int(value)
            converted = transform_geom(metadata["crs"], "EPSG:4326", geometry)
            geometry_parts.setdefault(source_id, []).append(converted)
        seen[np.unique(rows)] = True
    rows = np.concatenate(reference_rows)
    columns = np.concatenate(reference_columns)
    reference = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, columns)),
        shape=(max_id + 1, n_pixels),
    )
    reference_area = np.asarray(reference.sum(axis=1)).ravel()
    object_ids = np.flatnonzero(reference_area > 0)
    observed = np.zeros((len(frames), max_id + 1), dtype=bool)
    source_observed = np.zeros_like(observed)
    present = np.zeros_like(observed)
    building_fractions = np.zeros((len(frames), max_id + 1), dtype="float32")
    for t, (labels, good) in enumerate(zip(instances, valid)):
        coverage = np.asarray(reference @ np.asarray(good, dtype=np.float32).ravel()).ravel()
        observed[t] = coverage >= 0.8 * np.maximum(reference_area, 1)
        source_observed[t] = observed[t]
        covered = np.asarray(
            reference @ ((np.asarray(labels) > 0) & good).astype("float32").ravel()
        ).ravel()
        fraction = covered / np.maximum(coverage, 1)
        building_fractions[t] = fraction
        if protocol == "matched_id_v1":
            counts = np.bincount(
                np.asarray(labels)[np.asarray(good, dtype=bool)], minlength=max_id + 1
            )
            present[t] = counts > 0
        else:
            present[t] = fraction >= 0.8
            observed[t] &= (fraction >= 0.8) | (fraction <= 0.2)
    events: list[dict[str, Any]] = []
    reference_diagnostics = []
    for object_id in object_ids:
        stable = None
        stable_at = None
        pending_at = None
        count = 0
        geometry = _join_parts(geometry_parts[int(object_id)])
        initial_state = None
        before_count = len(events)
        for t in np.flatnonzero(observed[:, object_id]):
            state = bool(present[t, object_id])
            if stable is None:
                stable, stable_at = state, int(t)
                initial_state = state
                continue
            if state == stable:
                stable_at = int(t)
                pending_at, count = None, 0
                continue
            if pending_at is None:
                pending_at, count = int(t), 1
            else:
                count += 1
            if count < 2:
                continue
            change_class = "building_addition" if state else "building_removal"
            events.append(
                {
                    "truth_id": f"gt:{metadata['region_id']}:{int(object_id) - 1}:{pending_at}:{change_class}",
                    "object_id": int(object_id) - 1,
                    "region_id": metadata["region_id"],
                    "geometry": geometry,
                    "change_class": change_class,
                    "lower": frames[stable_at]["acquisition_start"],
                    "upper": frames[pending_at]["acquisition_end"],
                    "first_support": frames[pending_at]["acquisition_end"],
                    "confirmed_at": frames[int(t)]["acquisition_end"],
                    "first_support_index": pending_at,
                    "confirmation_index": int(t),
                    "reference_geometry": "first rasterized footprint of matched source ID; evaluation only",
                    "timing_target": "interval-censored monthly visibility; physical date unknown",
                    "truth_protocol": protocol,
                    "any_building_coverage_before": float(building_fractions[stable_at, object_id]),
                    "any_building_coverage_first_support": float(
                        building_fractions[pending_at, object_id]
                    ),
                    "any_building_coverage_confirmation": float(
                        building_fractions[int(t), object_id]
                    ),
                }
            )
            stable, stable_at = state, int(t)
            pending_at, count = None, 0
        source_count = int(source_observed[:, object_id].sum())
        ambiguous = int(
            (
                source_observed[:, object_id]
                & (building_fractions[:, object_id] > 0.2)
                & (building_fractions[:, object_id] < 0.8)
            ).sum()
        )
        reference_diagnostics.append(
            {
                "object_id": int(object_id) - 1,
                "source_valid_frames": source_count,
                "source_unusable_frames": len(frames) - source_count,
                "ambiguous_coverage_frames": ambiguous,
                "initial_state": initial_state,
                "usable_state_frames": int(observed[:, object_id].sum()),
                "right_censored_pending_support": int(count) if pending_at is not None else 0,
                "events_before_reference_dedup": len(events) - before_count,
            }
        )
    if diagnostics is not None:
        diagnostics.update(
            {
                "reference_count": len(object_ids),
                "total_reference_frames": len(frames) * len(object_ids),
                "source_valid_reference_frames": sum(
                    d["source_valid_frames"] for d in reference_diagnostics
                ),
                "source_unusable_reference_frames": sum(
                    d["source_unusable_frames"] for d in reference_diagnostics
                ),
                "ambiguous_coverage_reference_frames": sum(
                    d["ambiguous_coverage_frames"] for d in reference_diagnostics
                ),
                "initial_inventory_references": sum(
                    d["initial_state"] is True for d in reference_diagnostics
                ),
                "initial_absent_references": sum(
                    d["initial_state"] is False for d in reference_diagnostics
                ),
                "no_usable_state_references": sum(
                    d["initial_state"] is None for d in reference_diagnostics
                ),
                "right_censored_pending_references": sum(
                    d["right_censored_pending_support"] > 0 for d in reference_diagnostics
                ),
                "zero_event_references": sum(
                    d["events_before_reference_dedup"] == 0 for d in reference_diagnostics
                ),
                "references": reference_diagnostics,
            }
        )
    ordered = sorted(events, key=lambda event: (event["first_support"], event["object_id"]))
    if protocol == "matched_id_v1" or not ordered:
        return ordered
    # Multiple postprocessed IDs can describe the same physical footprint history.
    # Merge only strongly overlapping, same-class, near-simultaneous references;
    # retain every reference identity in the audit, never silently double-count it.
    return deduplicate_physical_references(ordered)


def deduplicate_physical_references(events: list[dict]) -> list[dict]:
    ordered = sorted(events, key=lambda event: (event["first_support"], event["object_id"]))
    groups = []
    for event in ordered:
        geometry = shape(event["geometry"])
        placed = False
        for group in groups:
            if all(
                event["change_class"] == member["change_class"]
                and abs(
                    (_date(event["first_support"]) - _date(member["first_support"])).total_seconds()
                )
                <= 62 * _DAY
                and geometry.intersects(shape(member["geometry"]))
                and geometry_iou(geometry, shape(member["geometry"])) >= 0.5
                for member in group
            ):
                group.append(event)
                placed = True
                break
        if not placed:
            groups.append([event])
    result = []
    for group in groups:
        representative = dict(group[0])
        representative["reference_truth_ids"] = [event["truth_id"] for event in group]
        representative["reference_members"] = [dict(event) for event in group]
        result.append(representative)
    return result


def ground_truth_audit(cache: str | Path) -> dict:
    """Keep ID transitions visible alongside the geometric-occupancy target.

    Identity-only changes are not verified physical footprint changes. This audit
    records all legacy targets, including coverage conflicts and shifted support.
    """
    legacy = ground_truth_events(cache, protocol="matched_id_v1")
    diagnostics = {}
    physical = ground_truth_events(cache, protocol="physical_coverage_v2", diagnostics=diagnostics)

    def public(event):
        return {k: v for k, v in event.items() if k != "geometry"}

    return {
        "primary_protocol": "physical_coverage_v2",
        "observation_and_exclusion_counts": diagnostics,
        "deduplication": "deterministic all-member IoU>=.5 and first-support time span<=62days; all member geometry/timing retained",
        "presence_min_fraction": 0.8,
        "absence_max_fraction": 0.2,
        "minimum_source_valid_fraction": 0.8,
        "ambiguous_middle_coverage": "unknown, cannot update label state",
        "legacy_id_event_count": len(legacy),
        "physical_coverage_event_count": len(physical),
        "merged_duplicate_reference_events": sum(
            len(e.get("reference_truth_ids", [e["truth_id"]])) - 1 for e in physical
        ),
        "identity_presence_conflicts": sum(
            (
                e["change_class"] == "building_removal"
                and min(
                    e["any_building_coverage_first_support"],
                    e["any_building_coverage_confirmation"],
                )
                >= 0.8
            )
            or (
                e["change_class"] == "building_addition"
                and e["any_building_coverage_before"] >= 0.8
            )
            for e in legacy
        ),
        "legacy_id_events": [public(e) for e in legacy],
        "physical_coverage_events": [public(e) for e in physical],
        "scope": "Label-derived geometric presence; not independent physical event dates or causal/legal evidence",
    }


def first_confirmed_indices(events: list[EventRevision]) -> list[int]:
    """Original list indices of each event ID's first model/human confirmation."""
    by_id: dict[str, int] = {}
    for index in sorted(
        range(len(events)), key=lambda i: (events[i].revision, events[i].record_time)
    ):
        event = events[index]
        if event.state in {"confirmed_by_model", "reviewed"} and event.event_id not in by_id:
            by_id[event.event_id] = index
    return sorted(by_id.values())


def _raw_interval(event: EventRevision) -> tuple[datetime | None, datetime]:
    raw = event.quality.get("unexpanded_observation_interval")
    if raw is not None:
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            raise ValueError("unexpanded observation interval must contain lower and upper")
        lower, upper = None if raw[0] is None else _date(raw[0]), _date(raw[1])
        if lower is not None and lower >= upper:
            raise ValueError("raw observation interval must have positive duration")
        return lower, upper
    return event.observation_interval.lower, event.observation_interval.upper


def _candidate_edges(
    events: list[EventRevision],
    truth: list[dict],
    *,
    iou_threshold: float,
    temporal_tolerance_days: float,
    max_detection_delay_days: float | None,
) -> tuple[list[int], dict[tuple[int, int], float]]:
    if not 0 < iou_threshold <= 1 or temporal_tolerance_days < 0:
        raise ValueError("matching requires IoU in (0,1] and nonnegative timing tolerance")
    predictions = first_confirmed_indices(events)
    if not predictions or not truth:
        return predictions, {}
    geometries = [shape(target["geometry"]) for target in truth]
    tree = STRtree(geometries)
    candidates: dict[tuple[int, int], float] = {}
    for row, original_index in enumerate(predictions):
        event = events[original_index]
        geometry = shape(event.geometry)
        _, first_support = _raw_interval(event)
        for truth_index in tree.query(geometry):
            truth_index = int(truth_index)
            target = truth[truth_index]
            if event.change_class != target["change_class"]:
                continue
            time_error = (
                abs((first_support - _date(target["first_support"])).total_seconds()) / _DAY
            )
            if time_error > temporal_tolerance_days:
                continue
            delay = (event.detected_at - _date(target["first_support"])).total_seconds() / _DAY
            if max_detection_delay_days is not None and delay > max_detection_delay_days:
                continue
            iou = geometry_iou(geometry, geometries[truth_index])
            if iou >= iou_threshold:
                candidates[row, truth_index] = iou
    return predictions, candidates


def match_events(
    events: list[EventRevision],
    truth: list[dict],
    *,
    iou_threshold: float = 0.25,
    temporal_tolerance_days: float = 93,
    max_detection_delay_days: float | None = None,
) -> list[tuple[int, int, float]]:
    """Maximum-cardinality, then maximum-IoU one-to-one event assignment.

    Results contain ORIGINAL event-revision indices, truth indices and geodesic
    IoU. Expanded/calibrated interval padding cannot improve match eligibility.
    Each candidate row gets its own unmatched dummy in a sparse assignment.
    """
    predictions, candidates = _candidate_edges(
        events,
        truth,
        iou_threshold=iou_threshold,
        temporal_tolerance_days=temporal_tolerance_days,
        max_detection_delay_days=max_detection_delay_days,
    )
    if not candidates:
        return []
    count = len(predictions)
    row_indices, columns, costs = [], [], []
    for (row, column), iou in sorted(candidates.items()):
        row_indices.append(row)
        columns.append(column)
        costs.append(1.0 - iou + 1e-6)
    unmatched_cost = min(count, len(truth)) + 2.0
    row_indices.extend(range(count))
    columns.extend(len(truth) + np.arange(count))
    costs.extend([unmatched_cost] * count)
    matrix = sparse.csr_matrix((costs, (row_indices, columns)), shape=(count, len(truth) + count))
    matched_rows, matched_columns = min_weight_full_bipartite_matching(matrix)
    return [
        (predictions[int(row)], int(column), candidates[int(row), int(column)])
        for row, column in zip(matched_rows, matched_columns)
        if column < len(truth)
    ]


def _ratio(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator else None


def _prf(tp: int, predictions: int, targets: int) -> dict:
    precision, recall = _ratio(tp, predictions), _ratio(tp, targets)
    f1 = _ratio(2 * tp, predictions + targets)
    return {
        "tp": int(tp),
        "fp": int(predictions - tp),
        "fn": int(targets - tp),
        "predictions": int(predictions),
        "targets": int(targets),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _statistics(values: Iterable[float]) -> dict:
    data = np.asarray(list(values), dtype=float)
    return {
        "n": len(data),
        "mean": float(np.mean(data)) if len(data) else None,
        "median": float(np.median(data)) if len(data) else None,
        "p90": float(np.quantile(data, 0.9)) if len(data) else None,
    }


def region_area_time(metadata: dict) -> dict:
    """Geodesic AOI area times source validity fraction times monthly duration.

    Pixel-count validity fractions approximate within-AOI valid-area fractions;
    area itself is measured ellipsoidally, never in EPSG:3857 square metres.
    """
    width, height = metadata["width"], metadata["height"]
    affine = Affine(*metadata["transform"])
    corners = [
        (affine.a * x + affine.b * y + affine.c, affine.d * x + affine.e * y + affine.f)
        for x, y in [(0, 0), (width, 0), (width, height), (0, height), (0, 0)]
    ]
    geometry = transform_geom(
        metadata["crs"], "EPSG:4326", {"type": "Polygon", "coordinates": [corners]}
    )
    area_km2 = _area(shape(geometry)) / 1e6
    source_years, model_years, duration_years = 0.0, 0.0, 0.0
    registration_rejected = 0
    for frame in metadata["frames"]:
        years = (
            _date(frame["acquisition_end"]) - _date(frame["acquisition_start"])
        ).total_seconds()
        years = years / (_DAY * 365.25)
        fraction = float(frame["valid_fraction"])
        if not 0 <= fraction <= 1 or years <= 0:
            raise ValueError("invalid source-valid fraction or acquisition duration")
        duration_years += years
        source_years += fraction * years
        if frame.get("registration", {}).get("usable", True):
            model_years += fraction * years
        else:
            registration_rejected += 1
    return {
        "geodesic_area_km2": area_km2,
        "calendar_observed_years": duration_years,
        "source_valid_area_km2_years": area_km2 * source_years,
        "model_usable_area_km2_years": area_km2 * model_years,
        "registration_rejected_frames": registration_rejected,
        "valid_area_fraction_method": "source-valid pixel fraction on a compact native AOI",
    }


def evaluate_candidate_sites(geometries: dict[str, dict], truth: list[dict]) -> dict:
    candidates = [shape(geometry) for geometry in geometries.values()]
    covered = 0
    if candidates:
        tree = STRtree(candidates)
        for target in truth:
            geometry = shape(target["geometry"])
            if any(
                geometry_iou(geometry, candidates[int(i)]) >= 0.25 for i in tree.query(geometry)
            ):
                covered += 1
    return {
        "sites": len(candidates),
        "truth_events_covered_at_iou_025": covered,
        "truth_events": len(truth),
        "coverage": _ratio(covered, len(truth)),
        "site_area_m2": _statistics(_area(geometry) for geometry in candidates),
    }


def evaluate_events(events: list[EventRevision], truth: list[dict], metadata: dict) -> dict:
    """Evaluate all alert histories and final retained event hypotheses separately."""
    first_indices = first_confirmed_indices(events)
    matched = match_events(events, truth)
    spatial_matched = match_events(events, truth, temporal_tolerance_days=float("inf"))
    horizon_matched = match_events(events, truth, max_detection_delay_days=62)
    latest: dict[str, EventRevision] = {}
    for event in events:
        if event.event_id not in latest or event.revision > latest[event.event_id].revision:
            latest[event.event_id] = event
    retained = [
        event for event in latest.values() if event.state in {"confirmed_by_model", "reviewed"}
    ]
    retained_matched = match_events(retained, truth)
    _, candidates = _candidate_edges(
        events, truth, iou_threshold=0.25, temporal_tolerance_days=93, max_detection_delay_days=None
    )
    matched_prediction_indices = {i for i, _, _ in matched}
    matched_target_indices = {j for _, j, _ in matched}
    duplicate_count = sum(
        index not in matched_prediction_indices
        and any(target in matched_target_indices for (row, target) in candidates if row == position)
        for position, index in enumerate(first_indices)
    )
    confirmed_ids = {events[index].event_id for index in first_indices}
    retracted_ids = {event.event_id for event in events if event.state == "retracted"}
    matched_ids = {events[index].event_id for index, _, _ in matched}
    false_ids = confirmed_ids - matched_ids
    delays, midpoint_errors, ious, raw_widths, expanded_widths = [], [], [], [], []
    containment, raw_containment, overlap = [], [], []
    delay_lower, delay_upper = [], []
    for index, target_index, iou in matched:
        event, target = events[index], truth[target_index]
        gt_lower, gt_upper = _date(target["lower"]), _date(target["upper"])
        lower, upper = _raw_interval(event)
        expanded_lower = event.observation_interval.lower
        expanded_upper = event.observation_interval.upper
        delays.append((event.detected_at - _date(target["first_support"])).total_seconds() / _DAY)
        delay_lower.append((event.detected_at - gt_upper).total_seconds() / _DAY)
        delay_upper.append((event.detected_at - gt_lower).total_seconds() / _DAY)
        ious.append(iou)
        raw_containment.append((lower is None or lower <= gt_lower) and upper >= gt_upper)
        containment.append(
            (expanded_lower is None or expanded_lower <= gt_lower) and expanded_upper >= gt_upper
        )
        overlap.append((lower is None or lower <= gt_upper) and upper >= gt_lower)
        if lower is not None:
            midpoint = lower + (upper - lower) / 2
            gt_midpoint = gt_lower + (gt_upper - gt_lower) / 2
            midpoint_errors.append(abs((midpoint - gt_midpoint).total_seconds()) / _DAY)
            raw_widths.append((upper - lower).total_seconds() / _DAY)
        if expanded_lower is not None:
            expanded_widths.append((expanded_upper - expanded_lower).total_seconds() / _DAY)
    exposure = region_area_time(metadata)
    false_alerts = len(first_indices) - len(matched)
    report = {
        "region_id": metadata["region_id"],
        "truth_protocols": sorted({event.get("truth_protocol", "unspecified") for event in truth}),
        "matching": {
            "iou_threshold": 0.25,
            "first_support_tolerance_days": 93,
            "one_to_one": True,
            "geometry_area": "WGS84 geodesic",
            "first_support": "unexpanded interval upper; monthly observation proxy",
        },
        "event": _prf(len(matched), len(first_indices), len(truth)),
        "spatial_object": _prf(len(spatial_matched), len(first_indices), len(truth)),
        "within_62_days": _prf(len(horizon_matched), len(first_indices), len(truth)),
        "final_retained": _prf(len(retained_matched), len(retained), len(truth)),
        "ever_confirmed_events": len(first_indices),
        "final_retained_events": len(retained),
        "candidate_event_ids": len({event.event_id for event in events}),
        "proposed_event_ids": len(
            {event.event_id for event in events if event.state == "proposed"}
        ),
        "revision_rows": len(events),
        "duplicate_confirmed_events": int(duplicate_count),
        "duplicate_rate": _ratio(duplicate_count, len(first_indices)),
        "retracted_event_ids": len(retracted_ids),
        "confirmed_then_retracted_event_ids": len(confirmed_ids & retracted_ids),
        "confirmed_retraction_rate": _ratio(len(confirmed_ids & retracted_ids), len(confirmed_ids)),
        "false_confirmed_then_retracted_events": len(false_ids & retracted_ids),
        "matched_confirmed_then_retracted_events": len(matched_ids & retracted_ids),
        "positive_detection_delay_building_months": sum(
            max(0, delay) / 30.4375 for delay in delays
        ),
        "analyst_cost_sensitivity": [
            {
                "units": "illustrative dimensionless work units, not measured currency or user preferences",
                "false_alert_weight": weight,
                "miss_weight": 10,
                "delay_building_month_weight": 1,
                "correction_weight": 5,
                "total": weight * false_alerts
                + 10 * (len(truth) - len(matched))
                + sum(max(0, delay) / 30.4375 for delay in delays)
                + 5 * len(confirmed_ids & retracted_ids),
            }
            for weight in [1, 10, 100]
        ],
        "matched_footprint_iou": _statistics(ious),
        "detection_delay_from_label_first_support_end_days": _statistics(delays),
        "matched_early_alerts": int(sum(delay < 0 for delay in delays)),
        "matched_late_over_62_days": int(sum(delay > 62 for delay in delays)),
        "delay_relative_to_label_interval_lower_bound_days": _statistics(delay_lower),
        "delay_relative_to_label_interval_upper_bound_days": _statistics(delay_upper),
        "timing": {
            "target": "monthly observation-supported visibility interval; physical date unknown",
            "matched_targets": len(matched),
            "expanded_interval_contains_label_interval": _ratio(sum(containment), len(containment)),
            "raw_interval_contains_label_interval": _ratio(
                sum(raw_containment), len(raw_containment)
            ),
            "raw_interval_overlaps_label_interval": _ratio(sum(overlap), len(overlap)),
            "raw_midpoint_absolute_error_days_interval_proxy": _statistics(midpoint_errors),
            "raw_interval_width_days": _statistics(raw_widths),
            "expanded_interval_width_days": _statistics(expanded_widths),
            "is_physical_date_coverage": False,
        },
        "exposure": exposure,
        "false_confirmed_alerts_per_source_valid_km2_year": _ratio(
            false_alerts, exposure["source_valid_area_km2_years"]
        ),
        "false_confirmed_alerts_per_model_usable_km2_year": _ratio(
            false_alerts, exposure["model_usable_area_km2_years"]
        ),
        "truth_events_by_class": {
            name: sum(target["change_class"] == name for target in truth)
            for name in ["building_addition", "building_removal"]
        },
        "limits": [
            "Native rasterized matched IDs omit labels without any rasterized pixel.",
            "First label geometry and future label identity are evaluation-only references.",
            "Source-valid reference coverage establishes observable absence; masked months are unknown.",
            "Registration-rejected frames remain in the source-observed truth denominator.",
            "Delay assumes retrospective month-end availability; original publication lag is unknown.",
            "Timing containment measures label-interval support, not physical event-date coverage.",
        ],
    }
    if "candidate_geometries" in metadata:
        report["candidate_sites"] = evaluate_candidate_sites(
            metadata["candidate_geometries"], truth
        )
    return report


def summarize_regions(
    per_region: dict[str, dict], *, bootstrap_samples: int = 2000, seed: int = 2901
) -> dict:
    """Aggregate counts, with region-resampled uncertainty rather than tile CIs."""
    values = list(per_region.values())
    if not values:
        raise ValueError("at least one evaluated geographic region is required")

    def aggregate(regions: list[dict]) -> dict:
        result = {}
        for name in ["event", "spatial_object", "within_62_days", "final_retained"]:
            result[name] = _prf(
                sum(region[name]["tp"] for region in regions),
                sum(region[name]["predictions"] for region in regions),
                sum(region[name]["targets"] for region in regions),
            )
        exposure = sum(region["exposure"]["source_valid_area_km2_years"] for region in regions)
        result["false_confirmed_alerts_per_source_valid_km2_year"] = _ratio(
            result["event"]["fp"], exposure
        )
        for key in [
            "duplicate_confirmed_events",
            "retracted_event_ids",
            "confirmed_then_retracted_event_ids",
            "ever_confirmed_events",
            "final_retained_events",
        ]:
            result[key] = sum(region[key] for region in regions)
        return result

    summary = aggregate(values)
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {}
    for _ in range(bootstrap_samples):
        sampled = aggregate([values[int(i)] for i in rng.integers(len(values), size=len(values))])
        for metric in ["event", "spatial_object", "within_62_days", "final_retained"]:
            for key in ["precision", "recall", "f1"]:
                value = sampled[metric][key]
                if value is not None:
                    samples.setdefault(f"{metric}.{key}", []).append(value)
        value = sampled["false_confirmed_alerts_per_source_valid_km2_year"]
        if value is not None:
            samples.setdefault("false_confirmed_alerts_per_source_valid_km2_year", []).append(value)
    summary["region_bootstrap_95_percentile_intervals"] = {
        key: {
            "lower": float(np.quantile(data, 0.025)),
            "upper": float(np.quantile(data, 0.975)),
            "defined_resamples": len(data),
        }
        for key, data in samples.items()
    }
    summary["regions"] = list(per_region)
    summary["uncertainty"] = {
        "resampling_unit": "complete geographic region",
        "regions": len(values),
        "samples": bootstrap_samples,
        "seed": seed,
        "limit": "Only a few regions; percentile intervals are unstable and do not establish geographic generalization.",
        "zero_denominators": "undefined precision/recall remain null; excluded from that bootstrap percentile",
    }
    return summary

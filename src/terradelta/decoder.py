"""Dwell, abstention, timing support and explicit correction semantics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
from typing import Any

import numpy as np

from .schema import EventRevision, ObservationInterval, SourceIdentity, validate_observation_order


@dataclass(frozen=True)
class DecoderConfig:
    threshold: float = 0.65
    confirmations: int = 2
    minimum_dwell_days: int = 20
    revision_window_days: int = 100
    maximum_gap_days: int = 190
    timing_padding_days: int = 0

    def __post_init__(self):
        if not 0.5 <= self.threshold < 1 or self.confirmations < 1:
            raise ValueError("threshold must be in [.5,1), confirmations positive")
        if (
            min(
                self.minimum_dwell_days,
                self.revision_window_days,
                self.maximum_gap_days,
                self.timing_padding_days,
            )
            < 0
        ):
            raise ValueError("decoder durations cannot be negative")


def _source(frame: dict) -> SourceIdentity:
    fields = [
        "source_id",
        "uri",
        "sha256",
        "acquisition_start",
        "acquisition_end",
        "published_at",
        "availability_assumption",
    ]
    license = frame.get("license")
    if license is None and frame["uri"].startswith("https://spacenet-dataset.s3.amazonaws.com/"):
        license = "CC-BY-SA-4.0"
    return SourceIdentity(**{k: frame[k] for k in fields}, license=license)


def decode_events(
    probabilities: np.ndarray,
    usable: np.ndarray,
    births: np.ndarray,
    info: dict,
    config: DecoderConfig = DecoderConfig(),
    *,
    model_revision: str = "unversioned",
    record_origin: datetime | None = None,
    retrospective: bool = False,
) -> list[EventRevision]:
    """Create causal revisions. Unknown evidence holds state rather than becoming absence.

    Sites present in the first usable observation are baseline inventory, not additions.
    A new opposite state proposes an event; independently usable repeat evidence confirms
    it. Fast reversal retracts model support. Later reversals outside the revision window
    may create another event. Monthly sources only bracket observation-supported change.
    """
    if probabilities.shape != usable.shape or probabilities.shape[0] != len(births):
        raise ValueError("site probability, availability and birth shapes disagree")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("occupancy probabilities must be finite in [0,1]")
    frames = info["metadata"]["frames"]
    validate_observation_order(frames)
    if len(frames) != probabilities.shape[1]:
        raise ValueError("source frame count differs from temporal evidence")
    sources = [_source(frame) for frame in frames]
    origin = record_origin or datetime.now(timezone.utc)
    records: list[tuple[int, int, dict[str, Any]]] = []
    for site in range(len(births)):
        stable: bool | None = None
        stable_at: int | None = None
        pending: dict | None = None
        confirmed: dict | None = None
        revision_counter: dict[str, int] = {}
        geometry = info["geometries"][str(site + 1)]

        def add(t: int, candidate: dict, state: str, reason: str):
            event_id = candidate["event_id"]
            revision_counter[event_id] = revision_counter.get(event_id, 0) + 1
            support = sorted(set([candidate["before"], candidate["first"], t]))
            available = sources[-1].available_at if retrospective else sources[t].available_at
            if retrospective:
                support = list(range(len(sources)))
            raw_lower = sources[candidate["before"]].acquisition_start
            raw_upper = sources[candidate["first"]].acquisition_end
            lower = max(
                sources[0].acquisition_start, raw_lower - timedelta(days=config.timing_padding_days)
            )
            upper = min(available, raw_upper + timedelta(days=config.timing_padding_days))
            records.append(
                (
                    t,
                    site,
                    dict(
                        event_id=event_id,
                        revision=revision_counter[event_id],
                        geometry=geometry,
                        change_class="building_addition"
                        if candidate["target"]
                        else "building_removal",
                        observation_interval=ObservationInterval(lower=lower, upper=upper),
                        detected_at=available,
                        sources=[sources[s] for s in support],
                        quality={
                            "site_id": site + 1,
                            "support_before_index": candidate["before"],
                            "support_first_index": candidate["first"],
                            "support_current_index": t,
                            "mode": "retrospective" if retrospective else "online_prefix",
                            "availability_assumption": "retrospective_month_end",
                            "registration": frames[t]["registration"],
                            "site_validity_required": 0.8,
                            "observed_support_count": candidate["count"],
                            "cloud_or_registration_gap": candidate.get("gap", False),
                            "unexpanded_observation_interval": [
                                raw_lower.isoformat(),
                                raw_upper.isoformat(),
                            ],
                            "development_timing_padding_days": config.timing_padding_days,
                            "score_meaning": "calibrated_occupancy_score_not_event_correctness_probability",
                        },
                        model_revision=model_revision,
                        state=state,
                        confidence=float(
                            probabilities[site, t]
                            if candidate["target"]
                            else 1 - probabilities[site, t]
                        ),
                        reason=reason,
                        region_id=info["region_id"],
                    ),
                )
            )

        for t, source in enumerate(sources):
            if not usable[site, t]:
                if pending:
                    pending["gap"] = True
                continue
            p = probabilities[site, t]
            observed = (
                True if p >= config.threshold else False if p <= 1 - config.threshold else None
            )
            if observed is None:
                continue  # ambiguous is neither absence nor support
            if stable is None:
                stable, stable_at = observed, t
                continue
            if observed == stable:
                if pending and t >= births[site]:
                    add(
                        t,
                        pending,
                        "retracted",
                        "Candidate lost support after another usable observation",
                    )
                    pending = None
                stable_at = t
                continue
            # Do not create/confirm public candidates before image-derived site discovery.
            if t < births[site]:
                continue
            assert stable_at is not None
            elapsed_gap = (source.acquisition_end - sources[stable_at].acquisition_end).days
            if elapsed_gap > config.maximum_gap_days:
                # Too long to support an event from this baseline. Re-establish inventory.
                if pending:
                    add(
                        t,
                        pending,
                        "retracted",
                        "Observation gap exceeded the frozen event support limit",
                    )
                stable, stable_at = observed, t
                pending = None
                continue
            if pending is None:
                key = f"{info['region_id']}:{site + 1}:{source.source_id}:{observed}"
                pending = {
                    "event_id": "td-" + hashlib.sha256(key.encode()).hexdigest()[:20],
                    "target": observed,
                    "before": stable_at,
                    "first": t,
                    "count": 1,
                    "gap": elapsed_gap > 62,
                }
                add(
                    t,
                    pending,
                    "proposed",
                    "First usable opposite-state support; persistence pending",
                )
            else:
                pending["count"] += 1
            duration = (source.acquisition_end - sources[pending["first"]].acquisition_end).days
            if pending["count"] < config.confirmations or duration < config.minimum_dwell_days:
                continue
            if (
                confirmed
                and (
                    source.acquisition_end - sources[confirmed["confirmed_at"]].acquisition_end
                ).days
                <= config.revision_window_days
            ):
                add(
                    t,
                    confirmed,
                    "retracted",
                    "Model support reversed within the revision window; physical cause unknown",
                )
                add(
                    t,
                    pending,
                    "retracted",
                    "Reversal treated as a correction rather than a second physical event",
                )
                confirmed = None
            else:
                add(
                    t,
                    pending,
                    "confirmed_by_model",
                    "Repeated usable evidence met the frozen persistence rule",
                )
                confirmed = dict(pending, confirmed_at=t)
            stable, stable_at = pending["target"], t
            pending = None
    # Temporal order is essential for globally monotone ledger transaction timestamps.
    records.sort(key=lambda row: (row[0], row[1]))
    result = []
    for i, (_, _, record) in enumerate(records):
        # Imported retrospective dates precede real reconstruction record time.
        record["record_time"] = origin + timedelta(microseconds=i)
        result.append(EventRevision(**record))
    return result

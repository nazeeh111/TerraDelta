"""Validated public contracts for observation-aware change histories.

Dates describe evidence availability and observation support, never a known
construction/demolition date. Geometry is always two-dimensional WGS84.
"""

from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pyproj import Geod
from shapely.geometry import MultiPolygon, Polygon, mapping, shape
from shapely.geometry.polygon import orient

EventState = Literal["proposed", "confirmed_by_model", "reviewed", "retracted"]
ChangeClass = Literal["building_addition", "building_removal"]
_GEOD = Geod(ellps="WGS84")


def utc_datetime(value: datetime | str) -> datetime:
    """Parse an explicitly offset timestamp; reject ambiguous local times."""
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamps must include a timezone offset")
    return value.astimezone(timezone.utc)


def validate_observation_order(frames: list[dict[str, Any]]) -> None:
    """Reject out-of-order publication until availability-order replay is supported.

    A monthly acquisition stream is only an online prefix when availability is
    monotone too. Recording unknown publication as assumed month end is explicit;
    silently indexing a noncontiguous publication selection is never allowed.
    """
    acquisition = [utc_datetime(f["acquisition_start"]) for f in frames]
    availability = [utc_datetime(f.get("published_at") or f["acquisition_end"]) for f in frames]
    if any(a >= b for a, b in zip(acquisition, acquisition[1:])):
        raise ValueError("Acquisition intervals must be strictly ordered")
    if any(a > b for a, b in zip(availability, availability[1:])):
        raise ValueError(
            "Out-of-order publication requires availability-order replay, which is not supported"
        )
    if any(a < utc_datetime(f["acquisition_end"]) for a, f in zip(availability, frames)):
        raise ValueError("Publication cannot precede acquisition interval end")


def validated_geometry(value: dict[str, Any]) -> dict[str, Any]:
    """Validate, orient and normalize an RFC 7946 footprint."""
    if not isinstance(value, dict) or value.get("type") not in {"Polygon", "MultiPolygon"}:
        raise ValueError("geometry must be a GeoJSON Polygon or MultiPolygon")
    geom = shape(value)
    if geom.is_empty or not geom.is_valid or geom.has_z or geom.area <= 0:
        raise ValueError("geometry must be a nonempty, valid, two-dimensional footprint")
    west, south, east, north = geom.bounds
    if not all(math.isfinite(v) for v in geom.bounds):
        raise ValueError("geometry coordinates must be finite")
    if west < -180 or east > 180 or south < -90 or north > 90:
        raise ValueError("geometry coordinates must use EPSG:4326 longitude/latitude")
    parts = list(geom.geoms) if isinstance(geom, MultiPolygon) else [geom]
    for polygon in parts:
        for ring in [polygon.exterior, *polygon.interiors]:
            coords = list(ring.coords)
            if any(abs(a[0] - b[0]) > 180 for a, b in zip(coords, coords[1:])):
                raise ValueError("split antimeridian-crossing footprints before import")
    oriented = [orient(p, sign=1.0) for p in parts]
    normalized = oriented[0] if isinstance(geom, Polygon) else MultiPolygon(oriented)
    return mapping(normalized)


def geodesic_area_m2(geometry: dict[str, Any]) -> float:
    """Ellipsoidal footprint area, not square degrees or Web Mercator area."""
    geom = shape(validated_geometry(geometry))
    parts = list(geom.geoms) if isinstance(geom, MultiPolygon) else [geom]
    return float(sum(abs(_GEOD.geometry_area_perimeter(p)[0]) for p in parts))


class SourceIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str = Field(min_length=1)
    uri: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    acquisition_start: datetime
    acquisition_end: datetime
    published_at: datetime | None = None
    availability_assumption: Literal["retrospective_month_end", "known_publication"] = (
        "retrospective_month_end"
    )
    license: str | None = None
    date_basis: Literal["mosaic_filename_month", "reported_acquisition_interval"] = (
        "mosaic_filename_month"
    )

    @field_validator("acquisition_start", "acquisition_end", "published_at", mode="before")
    @classmethod
    def timestamps(cls, value: Any) -> datetime | None:
        return None if value is None else utc_datetime(value)

    @field_validator("uri")
    @classmethod
    def sanitized_uri(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("source URI must omit credentials, query strings and fragments")
        if parsed.scheme not in {"https", "http", "s3", "file"}:
            raise ValueError("source URI must use https, http, s3 or file")
        return value

    @model_validator(mode="after")
    def valid_time_bounds(self) -> SourceIdentity:
        if self.acquisition_end <= self.acquisition_start:
            raise ValueError("source acquisition interval must have positive duration")
        if self.published_at and self.published_at < self.acquisition_end:
            raise ValueError("publication cannot precede the acquisition interval end")
        if self.availability_assumption == "known_publication" and self.published_at is None:
            raise ValueError("known_publication requires a publication timestamp")
        return self

    @property
    def available_at(self) -> datetime:
        """Known publication time, otherwise the explicitly assumed month end."""
        return self.published_at or self.acquisition_end


class ObservationInterval(BaseModel):
    """Conservative, closed timing support; lower=None means left-censored."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    lower: datetime | None
    upper: datetime

    @field_validator("lower", "upper", mode="before")
    @classmethod
    def timestamps(cls, value: Any) -> datetime | None:
        return None if value is None else utc_datetime(value)

    @model_validator(mode="after")
    def ordered(self) -> ObservationInterval:
        if self.lower is not None and self.lower >= self.upper:
            raise ValueError("timing support must have positive duration")
        return self


class EventRevision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    event_id: str = Field(min_length=1)
    revision: int = Field(ge=1)
    geometry: dict[str, Any]
    change_class: ChangeClass
    observation_interval: ObservationInterval
    detected_at: datetime
    record_time: datetime
    sources: list[SourceIdentity] = Field(min_length=1)
    quality: dict[str, Any] = Field(default_factory=dict)
    model_revision: str = Field(min_length=1)
    state: EventState
    confidence: float | None = Field(default=None, ge=0, le=1)
    reason: str | None = None
    region_id: str | None = None

    @field_validator("geometry")
    @classmethod
    def geometry_contract(cls, value: dict[str, Any]) -> dict[str, Any]:
        return validated_geometry(value)

    @field_validator("detected_at", "record_time", mode="before")
    @classmethod
    def timestamps(cls, value: Any) -> datetime:
        return utc_datetime(value)

    @model_validator(mode="after")
    def evidence_precedes_detection(self) -> EventRevision:
        if self.observation_interval.upper > self.detected_at:
            raise ValueError("timing support cannot extend beyond detection availability")
        if self.record_time < self.detected_at:
            raise ValueError("record_time cannot precede detected_at")
        source_ids = [s.source_id for s in self.sources]
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("source identities must be unique within an event revision")
        for source in self.sources:
            if source.available_at > self.detected_at:
                raise ValueError("supporting source was unavailable at detected_at")
        return self

    @property
    def area_m2(self) -> float:
        return geodesic_area_m2(self.geometry)

    def as_feature(self) -> dict[str, Any]:
        properties = self.model_dump(mode="json", exclude={"geometry"})
        properties["area_m2"] = self.area_m2
        return {
            "type": "Feature",
            "id": self.event_id,
            "geometry": self.geometry,
            "properties": properties,
        }

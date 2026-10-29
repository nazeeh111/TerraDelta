"""Append-only event revisions with stable transaction-time snapshots.

DuckDB files have a single writer. Use one EventStore instance per service and
serialize mutations; immutable exported snapshots may have arbitrary readers.
"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import threading
import uuid
from typing import Any, Iterable

import duckdb
from shapely.geometry import box, shape

from .schema import EventRevision, EventState, utc_datetime


def _canonical(event: EventRevision) -> str:
    return json.dumps(
        event.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), allow_nan=False
    )


class EventStore:
    """An append-only ledger; ``as_of`` refers to record time, not imagery date."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = duckdb.connect(self.path)
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS event_revisions (
              event_id VARCHAR NOT NULL,
              revision INTEGER NOT NULL,
              record_time TIMESTAMPTZ NOT NULL,
              payload VARCHAR NOT NULL,
              sha256 VARCHAR NOT NULL,
              PRIMARY KEY (event_id, revision)
            )
        """)

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def __enter__(self) -> EventStore:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def append(self, event: EventRevision | dict[str, Any]) -> bool:
        """Append a revision, or return False for byte-identical canonical replay.

        Record times must be strictly later than the ledger's last transaction
        time. That prevents later writes from altering an earlier ``as_of``
        snapshot, including when the new write introduces a different event.
        """
        # Revalidate nested structures, which callers can mutate despite frozen models.
        event = EventRevision.model_validate(
            event.model_dump(mode="json") if isinstance(event, EventRevision) else event
        )
        payload = _canonical(event)
        digest = hashlib.sha256(payload.encode()).hexdigest()
        with self._lock:
            self._db.execute("BEGIN TRANSACTION")
            try:
                found = self._db.execute(
                    "SELECT payload FROM event_revisions WHERE event_id=? AND revision=?",
                    [event.event_id, event.revision],
                ).fetchone()
                if found:
                    if found[0] != payload:
                        raise ValueError("revision already exists with different content")
                    self._db.execute("COMMIT")
                    return False
                previous = self._db.execute(
                    "SELECT revision, payload FROM event_revisions WHERE event_id=? "
                    "ORDER BY revision DESC LIMIT 1",
                    [event.event_id],
                ).fetchone()
                expected = previous[0] + 1 if previous else 1
                if event.revision != expected:
                    raise ValueError(f"expected revision {expected}, received {event.revision}")
                if previous:
                    old = EventRevision.model_validate_json(previous[1])
                    if old.change_class != event.change_class or old.region_id != event.region_id:
                        raise ValueError(
                            "change class and region identity cannot change across revisions"
                        )
                    if event.detected_at < old.detected_at:
                        raise ValueError("detection availability cannot move backwards")
                not_later = self._db.execute(
                    "SELECT coalesce(max(record_time) >= ?, false) FROM event_revisions",
                    [event.record_time],
                ).fetchone()[0]
                if not_later:
                    raise ValueError(
                        "record_time must be later than the ledger's latest record time"
                    )
                self._db.execute(
                    "INSERT INTO event_revisions VALUES (?, ?, ?, ?, ?)",
                    [event.event_id, event.revision, event.record_time, payload, digest],
                )
                self._db.execute("COMMIT")
                return True
            except Exception:
                self._db.execute("ROLLBACK")
                raise

    def history(self, event_id: str) -> list[EventRevision]:
        with self._lock:
            rows = self._db.execute(
                "SELECT payload FROM event_revisions WHERE event_id=? ORDER BY revision", [event_id]
            ).fetchall()
        return [EventRevision.model_validate_json(row[0]) for row in rows]

    def count_revisions(self) -> int:
        with self._lock:
            return int(self._db.execute("SELECT count(*) FROM event_revisions").fetchone()[0])

    def snapshot_time(self, as_of: datetime | str | None = None) -> datetime | None:
        """Return an existing transaction timestamp suitable as an immutable token.

        Cap a requested time at the latest transaction that already exists by
        that time. Return None when no transaction exists. Clients should return
        this token with query answers instead of inventing a future timestamp.
        """
        as_of = utc_datetime(as_of) if as_of is not None else None
        predicate = " WHERE record_time <= ?" if as_of is not None else ""
        with self._lock:
            value = self._db.execute(
                "SELECT CAST(max(record_time) AS VARCHAR) FROM event_revisions" + predicate,
                [as_of] if as_of is not None else [],
            ).fetchone()[0]
        return utc_datetime(value) if value is not None else None

    def query(
        self,
        *,
        bbox: Iterable[float] | None = None,
        start: datetime | str | None = None,
        end: datetime | str | None = None,
        as_of: datetime | str | None = None,
        states: Iterable[EventState] | None = None,
        include_retracted: bool = False,
        region_id: str | None = None,
    ) -> list[EventRevision]:
        """Latest revisions at transaction time, then exact geometry/time filters.

        ``start/end`` overlap the conservative observation interval, inclusively.
        A left-censored interval has no finite lower bound. Dateline-crossing
        query boxes must be split into two ordinary longitude boxes by clients.
        """
        start = utc_datetime(start) if start is not None else None
        end = utc_datetime(end) if end is not None else None
        as_of = utc_datetime(as_of) if as_of is not None else None
        if start is not None and end is not None and end < start:
            raise ValueError("end must not precede start")
        spatial_filter = None
        if bbox is not None:
            coords = list(bbox)
            if len(coords) != 4:
                raise ValueError("bbox must be [west, south, east, north]")
            west, south, east, north = coords
            if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
                raise ValueError("bbox must be a nonempty EPSG:4326 box without dateline crossing")
            spatial_filter = box(west, south, east, north)
        state_filter = set(states) if states is not None else None
        allowed = {"proposed", "confirmed_by_model", "reviewed", "retracted"}
        if state_filter is not None and not state_filter <= allowed:
            raise ValueError("unknown event state")
        predicate = "WHERE record_time <= ?" if as_of is not None else ""
        sql = (
            "SELECT payload FROM event_revisions "
            + predicate
            + " QUALIFY row_number() OVER (PARTITION BY event_id ORDER BY revision DESC)=1 "
            "ORDER BY event_id"
        )
        with self._lock:
            rows = self._db.execute(sql, [as_of] if as_of is not None else []).fetchall()
        result = []
        for (payload,) in rows:
            event = EventRevision.model_validate_json(payload)
            if region_id is not None and event.region_id != region_id:
                continue
            if event.state == "retracted" and not include_retracted:
                continue
            if state_filter is not None and event.state not in state_filter:
                continue
            interval = event.observation_interval
            if start is not None and interval.upper < start:
                continue
            if end is not None and interval.lower is not None and interval.lower > end:
                continue
            if spatial_filter is not None and not shape(event.geometry).intersects(spatial_filter):
                continue
            result.append(event)
        return result

    def export_geojson(self, path: str | Path, **query: Any) -> Path:
        """Write an RFC 7946 FeatureCollection (coordinates are lon/lat)."""
        snapshot = self.snapshot_time(query.get("as_of"))
        events = self.query(**{**query, "as_of": snapshot}) if snapshot is not None else []
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema_version": "terradelta-events-v1",
            "coordinate_reference_system": "OGC:CRS84",
            "area_method": "WGS84 ellipsoidal geodesic",
            "timing": "observation support intervals, not known construction/demolition dates",
            "query": _json_safe(query),
            "snapshot_time": _json_safe(snapshot),
        }
        collection = {
            "type": "FeatureCollection",
            "terradelta": metadata,
            "features": [event.as_feature() for event in events],
        }
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(
                json.dumps(collection, sort_keys=True, indent=2, allow_nan=False) + "\n"
            )
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return path

    def export_parquet(self, path: str | Path, **query: Any) -> Path:
        """Write GeoParquet with WGS84 geometry and lossless nested event JSON."""
        import geopandas as gpd
        import pyarrow.parquet as pq

        snapshot = self.snapshot_time(query.get("as_of"))
        events = self.query(**{**query, "as_of": snapshot}) if snapshot is not None else []
        rows = []
        for event in events:
            rows.append(
                {
                    "event_id": event.event_id,
                    "revision": event.revision,
                    "change_class": event.change_class,
                    "state": event.state,
                    "lower": event.observation_interval.lower,
                    "upper": event.observation_interval.upper,
                    "detected_at": event.detected_at,
                    "record_time": event.record_time,
                    "area_m2": event.area_m2,
                    "model_revision": event.model_revision,
                    "event_json": _canonical(event),
                    "geometry": shape(event.geometry),
                }
            )
        columns = [
            "event_id",
            "revision",
            "change_class",
            "state",
            "lower",
            "upper",
            "detected_at",
            "record_time",
            "area_m2",
            "model_revision",
            "event_json",
            "geometry",
        ]
        frame = gpd.GeoDataFrame(rows, columns=columns, geometry="geometry", crs="EPSG:4326")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            frame.to_parquet(temporary, index=False)
            table = pq.read_table(temporary)
            metadata = dict(table.schema.metadata or {})
            metadata[b"terradelta"] = json.dumps(
                {
                    "schema_version": "terradelta-events-v1",
                    "query": _json_safe(query),
                    "snapshot_time": _json_safe(snapshot),
                    "area_method": "WGS84 ellipsoidal geodesic",
                    "event_json": "lossless EventRevision including sources and quality",
                },
                sort_keys=True,
            ).encode()
            pq.write_table(table.replace_schema_metadata(metadata), temporary)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return path


def _json_safe(value: Any) -> Any:
    if isinstance(value, datetime):
        return utc_datetime(value).isoformat()
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [_json_safe(item) for item in value]
    return value

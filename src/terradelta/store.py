"""Append-only event revisions with stable transaction-time snapshots.

DuckDB files have a single writer. Use one EventStore instance per service and
serialize mutations; immutable exported snapshots may have arbitrary readers.
"""

from __future__ import annotations

import hashlib
import json
import threading
import uuid
from collections.abc import Iterable
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Self

import duckdb
from shapely.geometry import box, shape

from .schema import EventRevision, EventState, SourceIdentity, utc_datetime

COMPACT_FORMAT = "terradelta-store-compact-v1"
_FORMAT_KEY = "_terradelta_store_format"
_ROW_COLUMNS = "event_id, revision, CAST(record_time AS VARCHAR) AS record_time, payload, sha256"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _canonical(event: EventRevision) -> str:
    return _canonical_json(event.model_dump(mode="json"))


def _digest(payload: str) -> str:
    return hashlib.sha256(payload.encode()).hexdigest()


class EventStore:
    """Append-only ledger; ``as_of`` is record time, not imagery date.

    ``compact`` controls new rows only. Existing full-payload rows are never
    rewritten; either format can coexist and is expanded through the same API.
    Merely opening an existing ledger creates no tables and performs no writes.
    """

    def __init__(self, path: str | Path = ":memory:", *, compact: bool = True) -> None:
        self.path = str(path)
        self.compact = compact
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._source_cache: dict[str, SourceIdentity] = {}
        self._db = duckdb.connect(self.path)
        exists = self._db.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_schema='main' "
            "AND table_name='event_revisions'"
        ).fetchone()[0]
        if not exists:
            self._db.execute("""
                CREATE TABLE event_revisions (
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
            self._source_cache.clear()
            self._db.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _ensure_compact_tables(self) -> None:
        # Called only after a nonempty append has passed all ledger checks,
        # inside its transaction. Never migrate legacy rows on open/read.
        for table in ("event_sources", "event_geometries"):
            self._db.execute(
                f"CREATE TABLE IF NOT EXISTS {table} "
                "(sha256 VARCHAR PRIMARY KEY, payload VARCHAR NOT NULL)"
            )
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS event_store_formats (format VARCHAR PRIMARY KEY)"
        )
        self._db.execute(
            "INSERT INTO event_store_formats VALUES (?) ON CONFLICT DO NOTHING", [COMPACT_FORMAT]
        )

    def _reference_payloads(self, table: str, references: set[str]) -> dict[str, str]:
        if not references:
            return {}
        try:
            rows = self._db.execute(
                f"SELECT sha256, payload FROM {table} WHERE sha256 IN (SELECT unnest(?))",
                [list(references)],
            ).fetchall()
        except duckdb.CatalogException as exc:
            raise ValueError(f"missing compact reference table: {table}") from exc
        result = dict(rows)
        if set(result) != references:
            raise ValueError(f"missing compact reference in {table}")
        if any(_digest(payload) != digest for digest, payload in rows):
            raise ValueError(f"corrupt compact reference in {table}")
        return result

    def _decode_rows(self, rows: list[tuple]) -> list[EventRevision]:
        """Verify stored references and full public canonical hashes on every read.

        Typed immutable sources avoid repeated validation. Their stored payloads
        are still fetched/hashed each call, so cache warmth never hides a missing
        or corrupt reference. Geometry and returned sources are independent copies.
        """
        documents = [json.loads(row[3]) for row in rows]
        geometries: set[str] = set()
        sources: set[str] = set()
        for document in documents:
            if _FORMAT_KEY not in document:
                continue
            if (
                document[_FORMAT_KEY] != COMPACT_FORMAT
                or set(document) != {_FORMAT_KEY, "event", "geometry_ref", "source_refs"}
                or not isinstance(document["event"], dict)
                or {"geometry", "sources"} & set(document["event"])
                or not isinstance(document["geometry_ref"], str)
                or not isinstance(document["source_refs"], list)
                or not document["source_refs"]
                or any(not isinstance(ref, str) for ref in document["source_refs"])
            ):
                raise ValueError("unsupported or malformed compact event envelope")
            geometries.add(document["geometry_ref"])
            sources.update(document["source_refs"])
        geometry_payloads = self._reference_payloads("event_geometries", geometries)
        source_payloads = self._reference_payloads("event_sources", sources)
        geometry_values = {key: json.loads(value) for key, value in geometry_payloads.items()}
        if any(
            _canonical_json(value) != geometry_payloads[key]
            for key, value in geometry_values.items()
        ):
            raise ValueError("noncanonical compact geometry reference")
        for key, payload in source_payloads.items():
            if key not in self._source_cache:
                source = SourceIdentity.model_validate_json(payload)
                if _canonical_json(source.model_dump(mode="json")) != payload:
                    raise ValueError("noncanonical compact source reference")
                self._source_cache[key] = source
        result = []
        for row, document in zip(rows, documents):
            if _FORMAT_KEY in document:
                document = dict(
                    document["event"],
                    geometry=deepcopy(geometry_values[document["geometry_ref"]]),
                    sources=[
                        self._source_cache[ref].model_copy() for ref in document["source_refs"]
                    ],
                )
            event = EventRevision.model_validate(document)
            if _digest(_canonical(event)) != row[4] or (
                event.event_id,
                event.revision,
                event.record_time,
            ) != (row[0], row[1], utc_datetime(row[2])):
                raise ValueError("stored EventRevision canonical hash or identity mismatch")
            result.append(event)
        return result

    def append(self, event: EventRevision | dict[str, Any]) -> bool:
        """Append a revision, or return False for byte-identical canonical replay.

        Record times must be strictly later than the ledger's last transaction
        time, preventing later writes from changing an earlier ``as_of`` snapshot.
        """
        return bool(self.append_many([event]))

    def append_many(self, events: Iterable[EventRevision | dict[str, Any]]) -> int:
        """Append in iterable order atomically; return the number of new revisions.

        Exact duplicates are idempotent, including repeats within this batch.
        Every new row obeys append's consecutive-revision, identity, availability
        and strictly increasing global-record-time rules. Any failure rolls back
        the entire batch; existing rows are never rewritten.
        """
        # Revalidate nested structures, which callers can mutate despite frozen
        # models, and detach their mutable dictionaries before any database write.
        candidates = []
        for value in events:
            event = EventRevision.model_validate(
                value.model_dump(mode="json") if isinstance(value, EventRevision) else value
            )
            candidates.append((event, _canonical(event)))
        if not candidates:
            return 0
        ids = list({event.event_id for event, _ in candidates})
        with self._lock:
            self._db.execute("BEGIN TRANSACTION")
            try:
                # Preload only exact replay keys and the latest revision for
                # each relevant ID, in one query, rather than every old payload.
                stored = self._db.execute(
                    f"WITH ranked AS (SELECT {_ROW_COLUMNS}, row_number() OVER "
                    "(PARTITION BY event_id ORDER BY revision DESC) AS rank "
                    "FROM event_revisions WHERE event_id IN (SELECT unnest(?))) "
                    f"SELECT {_ROW_COLUMNS} FROM ranked WHERE rank=1 OR "
                    "(event_id, revision) IN (SELECT unnest(?), unnest(?))",
                    [
                        ids,
                        [event.event_id for event, _ in candidates],
                        [event.revision for event, _ in candidates],
                    ],
                ).fetchall()
                latest_rows = {}
                for row in stored:
                    if row[0] not in latest_rows or row[1] > latest_rows[row[0]][1]:
                        latest_rows[row[0]] = row
                old_events = self._decode_rows(stored)
                known = {
                    (event.event_id, event.revision): _canonical(event) for event in old_events
                }
                latest = {
                    event.event_id: event
                    for event in old_events
                    if event.revision == latest_rows[event.event_id][1]
                }
                clock_value = self._db.execute(
                    "SELECT CAST(max(record_time) AS VARCHAR) FROM event_revisions"
                ).fetchone()[0]
                clock = utc_datetime(clock_value) if clock_value is not None else None
                pending = []
                for event, canonical in candidates:
                    key = (event.event_id, event.revision)
                    if key in known:
                        if known[key] != canonical:
                            raise ValueError("revision already exists with different content")
                        continue
                    old = latest.get(event.event_id)
                    expected = old.revision + 1 if old else 1
                    if event.revision != expected:
                        raise ValueError(f"expected revision {expected}, received {event.revision}")
                    if old:
                        if (
                            old.change_class != event.change_class
                            or old.region_id != event.region_id
                        ):
                            raise ValueError(
                                "change class and region identity cannot change across revisions"
                            )
                        if event.detected_at < old.detected_at:
                            raise ValueError("detection availability cannot move backwards")
                    if clock is not None and event.record_time <= clock:
                        raise ValueError(
                            "record_time must be later than the ledger's latest record time"
                        )
                    pending.append((event, canonical))
                    known[key] = canonical
                    latest[event.event_id] = event
                    clock = event.record_time
                if self.compact and pending:
                    self._ensure_compact_tables()
                source_rows, geometry_rows, rows = {}, {}, []
                for event, canonical in pending:
                    digest = _digest(canonical)  # Always the FULL public EventRevision.
                    payload = canonical
                    if self.compact:
                        public = json.loads(canonical)
                        geometry = _canonical_json(public.pop("geometry"))
                        geometry_ref = _digest(geometry)
                        geometry_rows[geometry_ref] = geometry
                        source_refs = []
                        for source in public.pop("sources"):
                            source_payload = _canonical_json(source)
                            source_ref = _digest(source_payload)
                            source_rows[source_ref] = source_payload
                            source_refs.append(source_ref)
                        payload = _canonical_json(
                            {
                                _FORMAT_KEY: COMPACT_FORMAT,
                                "event": public,
                                "geometry_ref": geometry_ref,
                                "source_refs": source_refs,
                            }
                        )
                    rows.append(
                        (event.event_id, event.revision, event.record_time, payload, digest)
                    )
                for table, references in (
                    ("event_sources", source_rows),
                    ("event_geometries", geometry_rows),
                ):
                    if references:
                        existing = dict(
                            self._db.execute(
                                f"SELECT sha256, payload FROM {table} WHERE sha256 IN (SELECT unnest(?))",
                                [list(references)],
                            ).fetchall()
                        )
                        if any(existing[key] != references[key] for key in existing):
                            raise ValueError(f"corrupt/conflicting compact reference in {table}")
                        additions = [
                            (key, value) for key, value in references.items() if key not in existing
                        ]
                        if additions:
                            self._db.executemany(f"INSERT INTO {table} VALUES (?, ?)", additions)
                if rows:
                    self._db.executemany("INSERT INTO event_revisions VALUES (?, ?, ?, ?, ?)", rows)
                self._db.execute("COMMIT")
                return len(rows)
            except Exception:
                self._db.execute("ROLLBACK")
                raise

    def history(self, event_id: str) -> list[EventRevision]:
        with self._lock:
            rows = self._db.execute(
                f"SELECT {_ROW_COLUMNS} FROM event_revisions WHERE event_id=? ORDER BY revision",
                [event_id],
            ).fetchall()
            return self._decode_rows(rows)

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
        effective_states = allowed if state_filter is None else state_filter
        if not include_retracted:
            effective_states = effective_states - {"retracted"}
        if not effective_states:
            return []
        predicate = "WHERE record_time <= ?" if as_of is not None else ""
        # State filtering MUST follow latest-at-snapshot selection. Filtering
        # history first would resurrect an older confirmed revision after a
        # retraction, and expanding all sources first wastes most UI query work.
        sql = (
            f"WITH latest AS (SELECT {_ROW_COLUMNS} FROM event_revisions "
            + predicate
            + " QUALIFY row_number() OVER (PARTITION BY event_id ORDER BY revision DESC)=1) "
            f"SELECT {_ROW_COLUMNS} FROM latest WHERE coalesce("
            "json_extract_string(payload, '$.event.state'), "
            "json_extract_string(payload, '$.state')) IN (SELECT unnest(?)) "
        )
        parameters = ([as_of] if as_of is not None else []) + [sorted(effective_states)]
        if region_id is not None:
            sql += (
                "AND coalesce(json_extract_string(payload, '$.event.region_id'), "
                "json_extract_string(payload, '$.region_id')) = ? "
            )
            parameters.append(region_id)
        sql += "ORDER BY event_id"
        with self._lock:
            rows = self._db.execute(sql, parameters).fetchall()
            events = self._decode_rows(rows)
        result = []
        for event in events:
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

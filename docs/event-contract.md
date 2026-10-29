# Event history contract

`EventRevision` represents a supported building addition or removal hypothesis.
It is not proof of construction, demolition, cause, ownership, or legal status.
`proposed`, `confirmed_by_model`, `reviewed`, and `retracted` remain distinct.
A model confirmation does not imply human review. A correction appends a new
revision with its reason and supporting sources; it does not replace old rows.

## Geometry and evidence

Geometry is a valid 2D GeoJSON Polygon or MultiPolygon in WGS84 longitude/latitude
(EPSG:4326). Dateline-crossing polygons must be split before import. Area is
computed on the WGS84 ellipsoid in square metres. Web Mercator label area is not
used as physical area. GeoJSON follows RFC 7946 and omits the deprecated `crs`
member. GeoParquet records CRS metadata and includes a lossless `event_json`
column containing all source and quality fields.

Each source identity records a stable ID, sanitized URI, SHA-256 content hash,
acquisition interval, license when supplied, and publication time when known.
Source URLs cannot contain credentials, query strings, or fragments. The
pipeline must verify hashes against the actual bytes; the ledger validates the
hash format but does not download or inspect files.

SpaceNet 7's monthly mosaics give acquisition months, not exact image capture
times or physical event dates. With unknown original publication timestamps,
`availability_assumption=retrospective_month_end` explicitly assumes an image
available at the acquisition interval end. S3 object modification timestamps
are distribution metadata, not measured publication times. `known_publication`
requires an actual publication timestamp. `SourceIdentity.available_at` returns
that timestamp when present and the assumed acquisition end otherwise.

`observation_interval.lower` is the start of the last confidently opposite-state
observation month and `upper` is the end of the first supporting-state month.
This deliberately retains monthly mosaic and cloud-gap uncertainty. A null
lower bound is left-censored, meaning the first occurrence may predate the
sequence; such records are not ordinary newly observed additions. Interval
coverage must be evaluated against interval-censored labels, separately from
unknown exact physical event timing. A confidence value is not a guaranteed
calibrated probability under geographic shift.

`detected_at` is evidence availability time, including later observations used
to establish persistence. All supporting sources and the interval upper bound
must be available by detection. `record_time` is the actual ledger transaction
timestamp. These timestamps are timezone-aware and stored in UTC. Retrospective
reconstruction and prefix-restricted online inference must be labeled by the
producing pipeline; the ledger does not infer processing mode from timestamps.

## Append and query behavior

```python
from terradelta.schema import EventRevision
from terradelta.store import EventStore

with EventStore("events.duckdb") as store:
    store.append(EventRevision.model_validate(event_dict))
    events = store.query(
        bbox=[-80.2, 37.0, -80.0, 37.2],
        start="2019-01-01T00:00:00Z",
        end="2020-01-01T00:00:00Z",
        as_of="2026-10-29T15:00:00Z",
    )
    store.export_geojson("events.geojson", include_retracted=True)
    store.export_parquet("events.parquet", include_retracted=True)
```

- Revision numbering starts at 1 and increases by exactly 1 per event ID.
- An exact canonical replay returns `False`; changed content at an existing
  revision is rejected. Gaps and out-of-order revisions are rejected.
- Change class and region identity remain constant within an event history.
  Geometry, timing support, quality, confidence, model and state may be revised.
- Record times must be strictly increasing across the entire ledger. This
  prevents future additions or edits from changing an already returned `as_of`
  answer. A batch producer should assign distinct UTC microseconds or use its
  ingestion clock. A repeated identical revision is still idempotent.
- `as_of` chooses the latest revision recorded at or before that transaction
  time, before applying geometry, state and observation-interval filters.
- Return `store.snapshot_time(requested_as_of)` with a query answer and use that
  existing transaction timestamp for its immutable replay. Times beyond the
  latest existing transaction are open time intervals until ingestion advances;
  they are not immutable snapshot tokens. An empty ledger has no transaction
  token and should be reported explicitly as empty rather than assigning one.
- Bbox filtering uses actual polygon intersection, including polygon holes.
  Boxes must be `[west, south, east, north]` in EPSG:4326. Split dateline-crossing
  queries into two boxes. `start/end` overlap timing support inclusively.
- Current retractions are omitted by default, but remain in `history(event_id)`
  and in queries using `include_retracted=True`. Historical queries before the
  retraction still return the earlier state.
- Missing imagery and observed absence are source-status catalog facts, not
  interchangeable event rows. The ingest pipeline must retain that catalog.
- DuckDB supports one process writing a database file. One `EventStore` instance
  serializes its reads and writes with a lock. This is a local analytical ledger,
  not a concurrent multi-writer service. The raw DuckDB file is not a tamperproof
  audit log; external direct SQL writes bypass the application contract.

Exports use a temporary sibling file and atomic replacement so a failed write
does not destroy a previous valid export. Code uses MIT terms; imagery, labels
and derived data keep their own licenses, including SpaceNet's CC BY-SA 4.0.

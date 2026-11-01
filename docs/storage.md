# Immutable event storage

The public `EventRevision` contract and exports are unchanged. New `EventStore`
instances write compact rows by default; use `EventStore(path, compact=False)`
when full JSON payloads are needed for compatibility. Both formats can coexist
in one ledger. This is a storage revision, not a new scientific model, source,
event interpretation or evaluation result.

## Formats and hashes

`event_revisions` keeps its original five columns: `event_id`, `revision`,
`record_time`, `payload`, `sha256`, with the same `(event_id, revision)` primary
key. Its `sha256` always hashes the **full public canonical EventRevision JSON**,
never an internal envelope. Canonical JSON uses sorted keys, compact separators,
UTF-8 and rejects nonfinite numbers.

A legacy row's payload remains its complete canonical event. A new compact row
uses an explicitly versioned internal envelope:

```json
{
  "_terradelta_store_format": "terradelta-store-compact-v1",
  "event": {"event_id": "...", "revision": 1, "state": "proposed"},
  "geometry_ref": "full canonical geometry SHA256",
  "source_refs": ["full canonical SourceIdentity SHA256 in original order"]
}
```

The example's `event` is abbreviated; real envelopes contain every public field
except `geometry` and `sources`. `event_geometries` and `event_sources` each map
a SHA256 primary key to canonical JSON. Source references hash **all metadata**,
including exact URI, acquisition/publication timing, availability assumption,
license and date basis, as well as the pixel hash. The same image bytes with a
changed publication time or license therefore have a different source reference.
`event_store_formats` records the supported compact format explicitly.

These reference tables are immutable through the public store API. Repeated
content is inserted only once. A conflicting payload under an existing hash is
an error. Reads fetch and verify referenced payload hashes, validate the expanded
public event and verify its full canonical hash and row identity. Missing,
corrupt, unknown-format or malformed references fail; no source is guessed or
substituted. Typed `SourceIdentity` values are cached, but stored content is still
fetched and hashed on each read operation. Returned source objects and geometry
are independent copies, so caller mutation does not change later results.

## Existing ledgers and preservation

Opening an existing legacy ledger for history, query, snapshot or export performs
no table creation or migration. Reference tables are created lazily, inside the
first successful compact append transaction. An empty batch, exact replay or
rejected batch creates no reference tables. Existing revision rows are never
rewritten. A compact append to a legacy ledger adds new reference tables and new
rows only; historical snapshot answers and old row bytes remain unchanged.

Tests compare the entire closed legacy database file before and after opening,
reading and closing it. This covers the tested DuckDB version, not a promise
that a future database engine never changes its native file format. Compact
ledgers require the current reader; historical software expects full payloads.
Keep original study databases and their frozen software intact. Any separately
rebuilt compact database is a new physical artifact with its own file hash, even
when every public canonical revision is identical. Do not overwrite or migrate
original ledgers, and do not open a database while its original writer is active.

## Atomic bulk append

`append(event)` retains its boolean result. `append_many(events)` consumes an
iterable in order, appends everything in one transaction, and returns the count
of new revisions. Exact canonical duplicates return no additional count, even
when repeated within a batch. New revisions must still be consecutive per event,
keep change class and region identity, never decrease detection availability,
and have record times strictly later than the entire ledger's previous record
time. Evidence availability and all nested schema checks remain enforced.

Relevant replay revisions and latest revisions are preloaded in one query; the
global record clock is read once. Validation covers the complete ordered batch
before inserts. Reference and revision inserts use `executemany` inside the same
transaction. Any validation, reference conflict or database failure rolls back
the entire batch. No earlier valid member of a mixed invalid batch survives.
Callers should use bounded chunks appropriate to their memory budget, because
canonical candidates and new reference dictionaries are retained until commit.

## Query order and public exports

Queries first select each event's latest revision at or before `as_of`. Only
then are state and optional region predicates applied in SQL, using the legacy
`$.state`/`$.region_id` or compact `$.event.state`/`$.event.region_id` fields.
Reference expansion happens after these predicates. Filtering history before
latest selection would incorrectly resurrect an older confirmed event after a
retraction; regression tests prohibit that. Default query behavior remains all
latest states except retracted, unless the existing options request otherwise.
Other exact geometry and observation-interval filters retain their behavior.

History, snapshots, GeoJSON and Parquet expose full public events. They do not
expose envelopes or internal reference keys. Tests compare every synthetic
canonical history and every transaction snapshot across legacy sequential,
compact sequential and compact bulk stores. JSON and Parquet export files are
byte-identical across these formats in the tested environment, including empty
results, retractions and historical snapshots.

## Bounded synthetic benchmark

A task-local benchmark uses 500 revisions, 25 sites, 20 revisions per site,
24 rich source identities per revision and 97-coordinate polygon rings. It
compares the preserved original per-row writer, new full-payload bulk writer and
new compact bulk writer, with closed-file sizes and every canonical history
checked. It opens only synthetic databases. The reproducible script and measured
JSON are retained under the task's `storage-v1-evidence` folder. The final source-snapshot run measured:

| Writer | Append seconds | Closed database bytes |
|---|---:|---:|
| Original full-payload per-row append | 17.73 | 13,643,776 |
| New full-payload atomic bulk append | 6.74 | 13,643,776 |
| New compact atomic bulk append | 4.19 | 2,895,872 |

Compact event payloads totaled 1,142,800 bytes plus 111,920 bytes of references
(24 source records and 25 geometry records), compared with 12,743,660 full event
payload bytes. Including close/checkpointing, elapsed write times were 17.88s,
6.99s and 4.38s respectively. The compact database was about 79% smaller in this
run. Native database page allocation and concurrent local activity affect these
numbers. This single synthetic comparison is not a prediction for real study
workloads or evidence of faster scientific inference. The exact measurement is
`storage-v1-evidence/run-final/benchmark.json`; its copied source hash matches
the reviewed store implementation.

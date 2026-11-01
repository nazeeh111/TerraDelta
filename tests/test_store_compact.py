"""Storage-only synthetic histories; no model inputs, study stores or labels."""

import hashlib
import json
from datetime import timedelta

import pytest
from test_store import APR, MAR, REC, event, source

from terradelta.schema import EventRevision
from terradelta.store import COMPACT_FORMAT, EventStore, _canonical


def canonical(events):
    return [_canonical(item) for item in events]


def history():
    rows = []
    for revision in range(1, 4):
        for index in range(3):
            rows.append(
                event(
                    event_id=f"site-{index}",
                    revision=revision,
                    record_time=REC + timedelta(seconds=len(rows)),
                    detected_at=APR,
                    state=(
                        "retracted"
                        if revision == 3 and index == 1
                        else "proposed"
                        if revision == 1
                        else "confirmed_by_model"
                    ),
                    sources=[
                        source(source_id=f"image-{i}", uri=f"https://example.org/{i}.tif")
                        for i in range(3)
                    ],
                    quality={"nested": {"indices": [0, 1]}, "score": 0.1 * revision},
                )
            )
    return rows


def table_names(store):
    return {row[0] for row in store._db.execute("SHOW TABLES").fetchall()}


def test_every_public_history_snapshot_and_export_identical(tmp_path):
    import pyarrow.parquet as pq

    rows = history()
    with (
        EventStore(tmp_path / "legacy.db", compact=False) as legacy,
        EventStore(tmp_path / "sequential.db") as sequential,
        EventStore(tmp_path / "bulk.db") as bulk,
    ):
        for row in rows:
            assert legacy.append(row) and sequential.append(row)
        assert bulk.append_many(iter(rows)) == len(rows)
        assert bulk.append_many(rows + rows) == 0
        for event_id in ("site-0", "site-1", "site-2", "missing"):
            expected = canonical(legacy.history(event_id))
            assert (
                canonical(sequential.history(event_id))
                == canonical(bulk.history(event_id))
                == expected
            )
        for timestamp in [REC - timedelta(seconds=1), *[row.record_time for row in rows]]:
            for include_retracted in (False, True):
                query = {"as_of": timestamp, "include_retracted": include_retracted}
                expected = canonical(legacy.query(**query))
                assert (
                    canonical(sequential.query(**query))
                    == canonical(bulk.query(**query))
                    == expected
                )
                assert legacy.snapshot_time(timestamp) == bulk.snapshot_time(timestamp)
        for query in (
            {},
            {"as_of": REC + timedelta(seconds=3)},
            {"include_retracted": True},
            {"states": ["reviewed"]},
            {"bbox": [-80.01, 36.99, -79.99, 37.01], "region_id": "region-a"},
        ):
            json_paths = [
                store.export_geojson(tmp_path / f"{name}.json", **query)
                for name, store in (("legacy", legacy), ("seq", sequential), ("bulk", bulk))
            ]
            assert len({path.read_bytes() for path in json_paths}) == 1
            parquet_paths = [
                store.export_parquet(tmp_path / f"{name}.parquet", **query)
                for name, store in (("legacy", legacy), ("seq", sequential), ("bulk", bulk))
            ]
            assert len({path.read_bytes() for path in parquet_paths}) == 1
            assert pq.read_table(parquet_paths[0]).equals(pq.read_table(parquet_paths[-1]))
        payload, digest = bulk._db.execute(
            "SELECT payload, sha256 FROM event_revisions LIMIT 1"
        ).fetchone()
        assert json.loads(payload)["_terradelta_store_format"] == COMPACT_FORMAT
        assert digest == hashlib.sha256(_canonical(rows[0]).encode()).hexdigest()
        assert bulk._db.execute("SELECT count(*) FROM event_sources").fetchone()[0] == 3
        assert bulk._db.execute("SELECT count(*) FROM event_geometries").fetchone()[0] == 1


def test_legacy_read_does_not_change_bytes_or_schema_then_append_is_mixed(tmp_path):
    path = tmp_path / "legacy.db"
    with EventStore(path, compact=False) as store:
        store.append(event())
        original = store._db.execute(
            "SELECT * EXCLUDE(record_time) FROM event_revisions"
        ).fetchall()
        assert table_names(store) == {"event_revisions"}
    before = path.read_bytes()
    with EventStore(path) as store:
        assert canonical(store.history("event-a")) == [_canonical(event())]
        assert canonical(store.query()) == [_canonical(event())]
        assert store.snapshot_time() == REC
        assert table_names(store) == {"event_revisions"}
    assert path.read_bytes() == before
    with EventStore(path) as store:
        assert not store.append(event())  # Idempotent append creates no reference tables either.
        assert table_names(store) == {"event_revisions"}
        later = event(revision=2, record_time=REC + timedelta(seconds=1))
        assert store.append(later)
        assert (
            store._db.execute(
                "SELECT * EXCLUDE(record_time) FROM event_revisions WHERE revision=1"
            ).fetchall()
            == original
        )
        assert canonical(store.query(as_of=REC)) == [_canonical(event())]
        assert canonical(store.history("event-a")) == [_canonical(event()), _canonical(later)]
    with EventStore(path, compact=False) as store:
        last = event(revision=3, record_time=REC + timedelta(seconds=2))
        assert store.append(last)
        assert canonical(store.history("event-a"))[-1] == _canonical(last)


@pytest.mark.parametrize(
    "bad,pattern",
    [
        ({"revision": 4}, "expected revision"),
        ({"event_id": "backfill", "revision": 1, "record_time": REC}, "later than"),
        ({"change_class": "building_removal"}, "identity"),
        ({"region_id": "other"}, "identity"),
        ({"detected_at": MAR}, "backwards"),
        ({"revision": 1, "confidence": 0.9}, "different content"),
    ],
)
def test_mixed_invalid_batch_is_atomic(bad, pattern):
    with EventStore() as store:
        first = event(detected_at=APR)
        store.append(first)
        original_tables = {
            table: store._db.execute(f"SELECT * EXCLUDE(record_time) FROM {table}").fetchall()
            if table == "event_revisions"
            else store._db.execute(f"SELECT * FROM {table}").fetchall()
            for table in table_names(store)
        }
        valid = event(event_id="new", record_time=REC + timedelta(seconds=1))
        changes = {
            "revision": 2,
            "detected_at": APR,
            "record_time": REC + timedelta(seconds=2),
        } | bad
        with pytest.raises(ValueError, match=pattern):
            store.append_many([valid, event(**changes)])
        assert store.count_revisions() == 1
        assert canonical(store.query()) == [_canonical(first)]
        for table, rows in original_tables.items():
            select = "* EXCLUDE(record_time)" if table == "event_revisions" else "*"
            assert store._db.execute(f"SELECT {select} FROM {table}").fetchall() == rows


def test_batch_internal_revision_order_duplicates_and_global_clock():
    first = event()
    second = event(revision=2, record_time=REC + timedelta(seconds=1))
    third = event(event_id="other", record_time=REC + timedelta(seconds=2))
    with EventStore() as store:
        assert store.append_many([]) == 0
        assert store.append_many([first, first, second, first, third, second]) == 3
        assert not store.append(first)
        with pytest.raises(ValueError, match="later than"):
            store.append_many([event(event_id="late", record_time=REC + timedelta(seconds=2))])
        with pytest.raises(ValueError, match="expected revision 3"):
            store.append_many([event(revision=4, record_time=REC + timedelta(seconds=3))])
        assert store.count_revisions() == 3


def test_source_reference_hashes_all_metadata_and_preserves_order():
    old = source(source_id="same-image")
    revised = source(
        source_id="same-image",
        published_at=APR,
        availability_assumption="known_publication",
        license="updated terms",
    )
    other = source(source_id="another-image")
    first = event(detected_at=APR, sources=[old, other])
    second = event(
        revision=2,
        record_time=REC + timedelta(seconds=1),
        detected_at=APR,
        sources=[other, revised],
    )
    with EventStore() as store:
        assert store.append_many([first, second]) == 2
        assert old.sha256 == revised.sha256
        assert store._db.execute("SELECT count(*) FROM event_sources").fetchone()[0] == 3
        assert canonical(store.history(first.event_id)) == [_canonical(first), _canonical(second)]


def test_mutable_inputs_and_return_values_do_not_change_persisted_copies():
    candidate = event(quality={"nested": {"values": [1, 2]}})
    expected = _canonical(candidate)
    with EventStore() as store:
        store.append(candidate)
        candidate.geometry["coordinates"] = []
        candidate.quality["nested"]["values"].append(3)
        candidate.sources.clear()
        a, b = store.query()[0], store.history("event-a")[0]
        a.geometry["coordinates"] = []
        a.quality["nested"]["values"].append(4)
        a.sources.clear()
        assert _canonical(b) == expected
        assert _canonical(store.query()[0]) == expected
        bad = event()
        bad.geometry["coordinates"] = []
        with pytest.raises(ValueError):
            store.append_many([event(event_id="safe", record_time=REC + timedelta(seconds=1)), bad])
        assert store.count_revisions() == 1


def test_generator_input_is_canonicalized_when_consumed():
    first = event(quality={"nested": [1]}).model_dump(mode="json")
    expected = EventRevision.model_validate(first)
    expected_payload = _canonical(expected)

    def generate():
        yield first
        first["quality"]["nested"].append(99)
        yield event(event_id="other", record_time=REC + timedelta(seconds=1))

    with EventStore() as store:
        assert store.append_many(generate()) == 2
        assert _canonical(store.history("event-a")[0]) == expected_payload


@pytest.mark.parametrize("table", ["event_sources", "event_geometries"])
@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_missing_or_corrupt_reference_fails_even_with_warm_source_cache(table, damage):
    with EventStore() as store:
        store.append(event())
        assert store.query()  # Warm the typed-source cache.
        if damage == "missing":
            store._db.execute(f"DELETE FROM {table}")
        else:
            store._db.execute(f"UPDATE {table} SET payload='{{}}'")
        with pytest.raises(ValueError, match="compact reference"):
            store.query()
        with pytest.raises(ValueError, match="compact reference"):
            store.history("event-a")
        with pytest.raises(ValueError, match="compact reference"):
            store.append(event())


def test_full_public_digest_still_detects_envelope_corruption():
    with EventStore() as store:
        store.append(event())
        payload = json.loads(store._db.execute("SELECT payload FROM event_revisions").fetchone()[0])
        payload["event"]["quality"] = {"corrupt": True}
        store._db.execute("UPDATE event_revisions SET payload=?", [json.dumps(payload)])
        with pytest.raises(ValueError, match="canonical hash"):
            store.query()


@pytest.mark.parametrize("compact", [False, True])
def test_state_filter_follows_latest_at_snapshot_never_resurrects_old_confirmation(compact):
    with EventStore(compact=compact) as store:
        confirmed = event(state="confirmed_by_model")
        retracted = event(revision=2, state="retracted", record_time=REC + timedelta(seconds=1))
        store.append_many([confirmed, retracted])
        assert store.query(states=["confirmed_by_model"]) == []
        assert store.query(states=["confirmed_by_model"], as_of=retracted.record_time) == []
        assert canonical(store.query(states=["confirmed_by_model"], as_of=REC)) == [
            _canonical(confirmed)
        ]
        assert canonical(store.query(states=["retracted"], include_retracted=True)) == [
            _canonical(retracted)
        ]
        assert store.query(states=["retracted"]) == []


def test_state_filter_does_not_materialize_nonmatching_sources_or_geometry():
    from shapely.geometry import box, mapping

    with EventStore() as store:
        store.append_many(
            [
                event(),
                event(
                    event_id="confirmed",
                    region_id="only-region",
                    record_time=REC + timedelta(seconds=1),
                    state="confirmed_by_model",
                    geometry=mapping(box(0, 0, 0.001, 0.001)),
                    sources=[source(source_id="different-source")],
                ),
            ]
        )
        payload = json.loads(
            store._db.execute(
                "SELECT payload FROM event_revisions WHERE event_id='event-a'"
            ).fetchone()[0]
        )
        store._db.execute("DELETE FROM event_sources WHERE sha256=?", [payload["source_refs"][0]])
        store._db.execute("DELETE FROM event_geometries WHERE sha256=?", [payload["geometry_ref"]])
        assert [row.event_id for row in store.query(states=["confirmed_by_model"])] == ["confirmed"]
        assert [row.event_id for row in store.query(region_id="only-region")] == ["confirmed"]
        assert [
            row.event_id
            for row in store.query(states=["confirmed_by_model"], region_id="only-region")
        ] == ["confirmed"]
        assert store.query(region_id="missing") == []
        with pytest.raises(ValueError, match="compact reference"):
            store.query()


def test_database_failure_rolls_back_reference_inserts_and_lazy_tables():
    import duckdb

    with EventStore() as store:
        store._db.execute(
            "CREATE TABLE event_sources (sha256 VARCHAR PRIMARY KEY, "
            "payload VARCHAR NOT NULL CHECK (NOT contains(payload, 'failure-marker')))"
        )
        before = table_names(store)
        good = event()
        bad = event(
            event_id="second",
            record_time=REC + timedelta(seconds=1),
            sources=[source(license="failure-marker")],
        )
        with pytest.raises(duckdb.ConstraintException):
            store.append_many([good, bad])
        assert table_names(store) == before
        assert store.count_revisions() == 0
        assert store._db.execute("SELECT count(*) FROM event_sources").fetchone()[0] == 0
        assert store.append(good)


def test_mixed_batch_retains_supporting_source_availability_guard():
    first = event(event_id="safe")
    bad = event(event_id="unavailable", record_time=REC + timedelta(seconds=1)).model_dump(
        mode="json"
    )
    bad["sources"][0]["published_at"] = APR.isoformat()
    bad["sources"][0]["availability_assumption"] = "known_publication"
    with EventStore() as store:
        with pytest.raises(ValueError, match="unavailable"):
            store.append_many([first, bad])
        assert store.count_revisions() == 0
        assert table_names(store) == {"event_revisions"}

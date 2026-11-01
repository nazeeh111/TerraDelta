"""Portable history reconstruction from synthetic records, with no study data."""
from datetime import timedelta
import importlib.util
import json
from pathlib import Path

import pytest

from test_store import APR, REC, event
from terradelta.store import EventStore, _canonical


def module():
    path = Path(__file__).resolve().parents[1] / "examples/restore_history.py"
    spec = importlib.util.spec_from_file_location("restore_history_example", path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def records():
    return [
        event(event_id="a", record_time=REC),
        event(event_id="b", record_time=REC + timedelta(seconds=1)),
        event(event_id="a", revision=2, record_time=REC + timedelta(seconds=2),
              detected_at=APR, state="confirmed_by_model"),
        event(event_id="a", revision=3, record_time=REC + timedelta(seconds=3),
              detected_at=APR, state="retracted", reason="later support changed"),
        event(event_id="b", revision=2, record_time=REC + timedelta(seconds=4),
              detected_at=APR, state="confirmed_by_model"),
    ]


def write_input(path, rows):
    path.write_text("".join(_canonical(row) + "\n" for row in rows))
    return path.read_bytes()


def test_existing_output_and_original_input_are_preserved(tmp_path):
    restore = module()
    source, output = tmp_path / "history.jsonl", tmp_path / "existing"
    original = write_input(source, records())
    output.mkdir()
    (output / "precious.txt").write_bytes(b"preserve me")
    with pytest.raises(FileExistsError):
        restore.restore_history(source, output, batch_size=2)
    assert (output / "precious.txt").read_bytes() == b"preserve me"
    assert source.read_bytes() == original
    assert not list(tmp_path.glob(".existing.restore-*"))


def test_later_malformed_batch_retains_staging_and_never_claims_success(tmp_path):
    restore = module()
    source, output = tmp_path / "broken.jsonl", tmp_path / "new-history"
    write_input(source, records()[:2])
    with source.open("ab") as stream:
        stream.write(b'{"not": "a complete event"}\n')
    original = source.read_bytes()
    with pytest.raises(restore.RestoreFailure, match="line 3") as caught:
        restore.restore_history(source, output, batch_size=2)
    assert not output.exists()
    assert source.read_bytes() == original
    staging = caught.value.staging
    receipt = json.loads((staging / "restore-receipt.json").read_text())
    assert receipt["complete"] is False and receipt["status"] == "failed"
    assert receipt["revisions_restored"] == 2
    with EventStore(staging / "events.duckdb") as store:
        assert store.count_revisions() == 2


def test_every_canonical_revision_time_and_historical_answer_roundtrips(tmp_path):
    restore = module()
    source, output, rows = tmp_path / "history.jsonl", tmp_path / "restored", records()
    original = write_input(source, rows)
    receipt = restore.restore_history(source, output, batch_size=2,
                                      study_status="Synthetic fixture; no accuracy claim")
    assert receipt["complete"] and receipt["status"] == "complete"
    assert receipt["canonical_revisions_verified"] == len(rows)
    assert receipt["every_payload_sha_and_record_time_matches"]
    assert all(check["matches_input"] for check in receipt["snapshot_checks"])
    assert source.read_bytes() == original
    assert not Path(receipt["staging_directory_used"]).exists()
    catalog = json.loads((output / "catalog-info.json").read_text())
    assert catalog["model_revision"] == rows[0].model_revision
    assert catalog["study_status"] == "Synthetic fixture; no accuracy claim"
    with EventStore(output / "events.duckdb") as store:
        for identity in ("a", "b"):
            assert [_canonical(e) for e in store.history(identity)] == [
                _canonical(e) for e in rows if e.event_id == identity
            ]
        for at in [REC - timedelta(seconds=1), *[e.record_time for e in rows]]:
            latest = {e.event_id: e for e in rows if e.record_time <= at}
            for include_retracted in (False, True):
                expected = sorted(_canonical(e) for e in latest.values()
                                  if include_retracted or e.state != "retracted")
                actual = sorted(_canonical(e) for e in store.query(
                    as_of=at, include_retracted=include_retracted))
                assert actual == expected


def test_racing_destination_is_not_replaced_even_when_empty(tmp_path, monkeypatch):
    restore = module()
    source, output = tmp_path / "history.jsonl", tmp_path / "racing"
    original = write_input(source, records())
    publish = restore.publish_exclusive

    def race(staging, target):
        target.mkdir()
        publish(staging, target)

    monkeypatch.setattr(restore, "publish_exclusive", race)
    with pytest.raises(restore.RestoreFailure):
        restore.restore_history(source, output, batch_size=2)
    assert output.is_dir() and list(output.iterdir()) == []
    assert source.read_bytes() == original
    stages = list(tmp_path.glob(".racing.restore-*"))
    assert len(stages) == 1
    assert json.loads((stages[0] / "restore-receipt.json").read_text())["complete"] is False

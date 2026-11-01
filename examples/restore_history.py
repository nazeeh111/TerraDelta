#!/usr/bin/env python3
"""Restore released revisions.jsonl into a NEW validated compact history directory.

Usage: python examples/restore_history.py revisions.jsonl --output demo/run
No model, imagery, labels or original DuckDB file is needed. Never overwrites
an output. Failed private staging directories and the input are preserved.
"""
from __future__ import annotations

import argparse
import ctypes
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import time
import uuid

try:
    import resource
except ImportError:  # Windows: restoration still works, process RSS is unreported.
    resource = None

from terradelta.schema import EventRevision, utc_datetime
from terradelta.store import EventStore


class RestoreFailure(RuntimeError):
    def __init__(self, message, staging):
        super().__init__(f"{message}; preserved staging: {staging}")
        self.staging = staging


def canonical(event):
    return json.dumps(event.model_dump(mode="json"), sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)  # Only this run's own progress/receipt is replaced.


def publish_exclusive(staging, output):
    """Atomic directory rename that refuses even a racing empty destination."""
    if os.name == "nt":
        os.rename(staging, output)  # Windows refuses an existing destination.
        return
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        rename = libc.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        result = rename(os.fsencode(staging), os.fsencode(output), 0x00000004)  # RENAME_EXCL
    elif sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        rename = libc.renameat2
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                           ctypes.c_char_p, ctypes.c_uint]
        result = rename(-100, os.fsencode(staging), -100, os.fsencode(output), 1)  # NOREPLACE
    else:
        raise RuntimeError("exclusive atomic directory publication unavailable; staging retained")
    if result:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(output))


def read_batches(path, batch_size):
    batch = []
    with path.open("rb") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                event = EventRevision.model_validate_json(line)
            except Exception as error:
                raise ValueError(f"invalid EventRevision at JSONL line {line_number}: {error}") from error
            batch.append(event)
            if len(batch) == batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


def snapshot_checks(store, audit, total, batch_size):
    first = audit.execute("SELECT record_time FROM expected WHERE seq=1").fetchone()[0]
    positions = {1, max(1, total // 2), total}
    for state in ("confirmed_by_model", "retracted"):
        row = audit.execute("SELECT min(seq) FROM expected WHERE state=?", [state]).fetchone()[0]
        if row is not None:
            positions.add(row)
    reports = []
    for position in [0, *sorted(positions)]:
        at = (utc_datetime(first) - timedelta(microseconds=1) if position == 0 else
              utc_datetime(audit.execute("SELECT record_time FROM expected WHERE seq=?",
                                        [position]).fetchone()[0]))
        token = store.snapshot_time(at)
        expected_token = None if position == 0 else at
        if token != expected_token:
            raise ValueError("restored record-time snapshot token differs from JSONL")
        for include_retracted in (False, True):
            expected = audit.execute(
                "WITH ranked AS (SELECT event_id,revision,state,row_number() OVER "
                "(PARTITION BY event_id ORDER BY seq DESC) rank FROM expected WHERE seq<=?) "
                "SELECT event_id,revision,state FROM ranked WHERE rank=1 "
                + ("" if include_retracted else "AND state!='retracted' ") + "ORDER BY event_id",
                [position],
            )
            actual = store._db.cursor().execute(
                "WITH ranked AS (SELECT event_id,revision,coalesce("
                "json_extract_string(payload,'$.event.state'),json_extract_string(payload,'$.state')) "
                "state,row_number() OVER (PARTITION BY event_id ORDER BY revision DESC) rank "
                "FROM event_revisions WHERE record_time<=?) "
                "SELECT event_id,revision,state FROM ranked WHERE rank=1 "
                + ("" if include_retracted else "AND state!='retracted' ") + "ORDER BY event_id",
                [at],
            )
            digest, count = hashlib.sha256(), 0
            while True:
                wanted, received = expected.fetchmany(batch_size), actual.fetchmany(batch_size)
                if wanted != received:
                    raise ValueError("restored snapshot IDs/revisions/states differ from JSONL")
                if not wanted:
                    break
                for row in wanted:
                    digest.update((json.dumps(row, separators=(",", ":")) + "\n").encode())
                count += len(wanted)
            actual.close()
            reports.append({"input_revision_position": position, "requested_as_of": at.isoformat(),
                            "snapshot_time": token.isoformat() if token else None,
                            "include_retracted": include_retracted, "events": count,
                            "ids_revisions_states_sha256": digest.hexdigest(), "matches_input": True})
    return reports


def restore_history(jsonl, output, *, batch_size=500, study_status=None):
    jsonl, output = Path(jsonl).absolute(), Path(output).absolute()
    if not 1 <= batch_size <= 500:
        raise ValueError("batch_size must be between 1 and 500")
    if os.path.lexists(output):
        raise FileExistsError(f"output exists and will be preserved: {output}")
    if not jsonl.is_file():
        raise FileNotFoundError(jsonl)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.restore-", dir=output.parent))
    receipt = {"format": "terradelta-history-restore-v1", "status": "in_progress", "complete": False,
               "input": str(jsonl), "output": str(output), "staging_directory_used": str(staging),
               "batch_size": batch_size, "revisions_restored": 0}
    started = time.monotonic()
    location = staging
    try:
        receipt["input_bytes"] = jsonl.stat().st_size
        receipt["input_sha256"] = file_sha256(jsonl)
        write_json(staging / "restore-receipt.json", receipt)
        with sqlite3.connect(staging / "restore-audit.sqlite3") as audit:
            audit.execute("PRAGMA cache_size=-8192")
            audit.execute("PRAGMA temp_store=FILE")
            audit.execute("CREATE TABLE expected (seq INTEGER PRIMARY KEY,event_id TEXT,"
                          "revision INTEGER,record_time TEXT,sha256 TEXT,state TEXT,model_revision TEXT)")
            audit.execute("CREATE INDEX expected_event_sequence ON expected(event_id,seq)")
            with EventStore(staging / "events.duckdb") as store:
                store._db.execute("SET memory_limit='256MB'")
                store._db.execute("SET threads=2")
                store._db.execute("SET temp_directory=?", [str(staging / "duckdb-spill")])
                first_model = None
                for batch in read_batches(jsonl, batch_size):
                    added = store.append_many(batch)
                    if added != len(batch):
                        raise ValueError("input repeats a revision; refusing to silently drop input records")
                    rows = []
                    for event in batch:
                        if first_model is None:
                            first_model = event.model_revision
                        receipt["revisions_restored"] += 1
                        rows.append((receipt["revisions_restored"], event.event_id, event.revision,
                                     event.record_time.isoformat(), hashlib.sha256(
                                         canonical(event).encode()).hexdigest(),
                                     event.state, event.model_revision))
                    audit.executemany("INSERT INTO expected VALUES (?,?,?,?,?,?,?)", rows)
                    audit.commit()
                    # Bound the store's validated source-object memoization too.
                    store._source_cache.clear()
                    write_json(staging / "restore-receipt.json", receipt)
                    print(json.dumps({"phase": "restore", "records": receipt["revisions_restored"]}),
                          flush=True)
                total = receipt["revisions_restored"]
                if not total:
                    raise ValueError("empty JSONL is not a complete event history")
                if store.count_revisions() != total:
                    raise ValueError("restored revision count differs from input")
                verified = 0
                for batch in read_batches(jsonl, batch_size):
                    rows = store._db.execute(
                        "SELECT event_id,revision,CAST(record_time AS VARCHAR),payload,sha256 "
                        "FROM event_revisions WHERE (event_id,revision) IN "
                        "(SELECT unnest(?),unnest(?)) ORDER BY record_time",
                        [[e.event_id for e in batch], [e.revision for e in batch]],
                    ).fetchall()
                    # Use the store's hash-verifying compact batch decoder, never
                    # history()/query() which could materialize unbounded histories.
                    received = store._decode_rows(rows)
                    if len(received) != len(batch):
                        raise ValueError("missing restored revision")
                    for wanted, actual, row in zip(batch, received, rows):
                        payload = canonical(wanted)
                        if (canonical(actual) != payload or actual.record_time != wanted.record_time
                                or row[4] != hashlib.sha256(payload.encode()).hexdigest()):
                            raise ValueError("canonical EventRevision/hash/record_time mismatch")
                        verified += 1
                    store._source_cache.clear()
                if file_sha256(jsonl) != receipt["input_sha256"]:
                    raise ValueError("input bytes changed during restoration/verification")
                snapshots = snapshot_checks(store, audit, total, batch_size)
                receipt.update({"canonical_revisions_verified": verified,
                                "every_payload_sha_and_record_time_matches": verified == total,
                                "snapshot_checks": snapshots})
            models = audit.execute("SELECT count(DISTINCT model_revision) FROM expected").fetchone()[0]
        catalog = {
            "model_revision": first_model if models == 1 else "Mixed processing revisions; inspect ledger",
            "processing_revisions_count": models,
            "study_status": study_status or "Restored history; restoration does not certify detection accuracy.",
            "restoration": "Lossless compact ledger restored from released EventRevision JSONL",
        }
        write_json(staging / "catalog-info.json", catalog)
        receipt.update({"status": "validated_pending_publication",
                        "events_duckdb_sha256": file_sha256(staging / "events.duckdb"),
                        "events_duckdb_bytes": (staging / "events.duckdb").stat().st_size,
                        "validation": "Every canonical payload, SHA256 and record time; independent disk-indexed JSONL snapshots",
                        "memory_bound": "Batch size<=500; bounded input+decoded batches; DuckDB256MiB; SQLite8MiBcache; source memoization cleared per batch",
                        "maximum_resident_bytes": (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                                                   * (1 if sys.platform == "darwin" else 1024)
                                                   if resource is not None else None),
                        "elapsed_seconds": time.monotonic() - started})
        write_json(staging / "restore-receipt.json", receipt)
        publish_exclusive(staging, output)
        location = output
        receipt.update(status="complete", complete=True)
        write_json(output / "restore-receipt.json", receipt)
        return receipt
    except BaseException as error:
        receipt.update(status="failed", complete=False, error_type=type(error).__name__,
                       error=str(error), elapsed_seconds=time.monotonic() - started)
        write_json(location / "restore-receipt.json", receipt)
        raise RestoreFailure(str(error), location) from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--study-status", help="Explicit study disclaimer recorded for the dashboard")
    args = parser.parse_args()
    try:
        result = restore_history(args.jsonl, args.output, batch_size=args.batch_size,
                                 study_status=args.study_status)
    except (RestoreFailure, FileExistsError, FileNotFoundError, ValueError) as error:
        print(json.dumps({"status": "failed", "complete": False, "error": str(error)}), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

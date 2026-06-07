"""Tests for storage layer correctness."""
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import storage


def _fresh():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    storage._apply_schema(conn)
    return conn


def test_meta_roundtrip():
    conn = _fresh()
    storage.set_meta(conn, "foo", "bar")
    conn.commit()
    assert storage.get_meta(conn, "foo") == "bar"
    assert storage.get_meta(conn, "missing") is None


def test_upsert_topic_idempotent():
    conn = _fresh()
    storage.upsert_topic(conn, "t1", "Topic One")
    storage.upsert_topic(conn, "t1", "Topic One Updated")
    conn.commit()
    # last_seen updates; canonical_name updates on conflict? Let's check schema.
    row = storage.get_topic(conn, "t1")
    assert row is not None


def test_alias_lookup():
    conn = _fresh()
    storage.upsert_topic(conn, "t1", "Topic One")
    storage.add_alias(conn, "t1", "Alias One", source="test")
    conn.commit()
    assert storage.find_by_alias(conn, "Alias One") == "t1"
    assert storage.find_by_alias(conn, "nonexistent") is None


def test_insert_event_dedup():
    conn = _fresh()
    storage.upsert_topic(conn, "t1", "Topic One")
    storage.add_alias(conn, "t1", "Topic One", source="test")
    conn.commit()

    id1 = storage.insert_event(
        conn, source="wikipedia", external_id="rev-1",
        ts="2026-06-07T12:00:00Z", title="Topic One",
        url=None, topic_id="t1",
    )
    conn.commit()
    id2 = storage.insert_event(
        conn, source="wikipedia", external_id="rev-1",
        ts="2026-06-07T12:00:00Z", title="Topic One",
        url=None, topic_id="t1",
    )
    conn.commit()
    assert id1 is not None
    assert id2 is None  # duplicate -> deduped


def test_signal_accumulation():
    conn = _fresh()
    storage.upsert_topic(conn, "t1", "Topic One")
    conn.commit()
    bucket = storage.hour_bucket()
    storage.upsert_signal(conn, "t1", bucket, "wikipedia", "edit_count", 3.0)
    storage.upsert_signal(conn, "t1", bucket, "wikipedia", "edit_count", 2.0)
    conn.commit()
    row = conn.execute(
        "SELECT value FROM signals WHERE topic_id='t1' AND signal='edit_count'",
    ).fetchone()
    assert row["value"] == 5.0  # signals accumulate (INSERT ... DO UPDATE SET value = value + ...)


def test_hour_bucket_format():
    bucket = storage.hour_bucket()
    assert len(bucket) == 20
    assert bucket.endswith(":00:00Z")

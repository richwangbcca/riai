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


# ---------------------------------------------------------------------------
# Time-window cutoffs
# ---------------------------------------------------------------------------

def test_cutoff_sql_matches_stored_timestamp_format():
    """Cutoffs must be in now_utc()'s format, or string comparison against a ts
    column diverges at the 'T' vs ' ' separator and widens the window by a day."""
    import re

    conn = _fresh()
    got = conn.execute(f"SELECT {storage.cutoff_sql('hours')}", ("-24",)).fetchone()[0]
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", got), got
    # same shape as what every ts column actually stores
    assert len(got) == len(storage.now_utc())


def test_cutoff_excludes_rows_sharing_the_cutoff_date():
    """The exact bug: a row older than the window but on the cutoff's calendar
    date used to compare greater than "YYYY-MM-DD HH:MM:SS" and sneak in."""
    from datetime import datetime, timedelta, timezone

    conn = _fresh()
    storage.upsert_topic(conn, "t1", "Topic One")
    now = datetime.now(timezone.utc)
    iso = lambda dt: dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    expected = 0
    for hours in (1, 23):  # inside a 24h window
        storage.insert_score(conn, "t1", iso(now - timedelta(hours=hours)))
        expected += 1
    for hours in (25, 30, 40, 47):  # outside it
        storage.insert_score(conn, "t1", iso(now - timedelta(hours=hours)))

    # Tightest case: one second past the cutoff, on the cutoff's own date. The
    # old comparison let this through whenever the cutoff wasn't exactly midnight
    # (when it is, there is no same-date-but-earlier instant to test).
    cutoff = now - timedelta(hours=24)
    if cutoff.time() != cutoff.min.time():
        storage.insert_score(conn, "t1", iso(cutoff - timedelta(seconds=1)))
    conn.commit()

    rows = storage.get_score_history(conn, "t1", hours=24)
    assert len(rows) == expected, [r["ts"] for r in rows]


def test_purge_respects_the_day_boundary():
    from datetime import datetime, timedelta, timezone

    conn = _fresh()
    now = datetime.now(timezone.utc)
    iso = lambda dt: dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    for days in (1, 3, 8, 12):
        storage.insert_event(
            conn, source="news", external_id=f"e{days}",
            ts=iso(now - timedelta(days=days)), title=f"t{days}", url=None,
        )
    conn.commit()

    assert storage.purge_old_events(conn, retain_days=7) == 2
    kept = {r["external_id"] for r in conn.execute("SELECT external_id FROM events")}
    assert kept == {"e1", "e3"}

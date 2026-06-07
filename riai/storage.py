"""
SQLite access layer. All DB interaction goes through this module.
Schema is applied from ../schema.sql on first open.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


_SCHEMA = Path(__file__).parent.parent / "schema.sql"


def _apply_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA.read_text())


def open_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    _apply_schema(conn)
    return conn


@contextmanager
def tx(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Simple transaction context manager."""
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def hour_bucket(ts: str | None = None) -> str:
    """Return the current (or given) UTC hour as a bucket string."""
    if ts is None:
        t = datetime.now(timezone.utc)
    else:
        t = datetime.fromisoformat(ts.rstrip("Z")).replace(tzinfo=timezone.utc)
    return t.strftime("%Y-%m-%dT%H:00:00Z")


# ---------------------------------------------------------------------------
# Meta (pipeline state)
# ---------------------------------------------------------------------------

def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta(key, value, updated_at) VALUES (?,?,?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
        (key, value, now_utc()),
    )


# ---------------------------------------------------------------------------
# Topics
# ---------------------------------------------------------------------------

def upsert_topic(
    conn: sqlite3.Connection,
    topic_id: str,
    canonical_name: str,
    wikipedia_title: str | None = None,
) -> None:
    ts = now_utc()
    conn.execute(
        """
        INSERT INTO topics(topic_id, canonical_name, wikipedia_title, first_seen, last_seen)
        VALUES (?,?,?,?,?)
        ON CONFLICT(topic_id) DO UPDATE SET
            last_seen = excluded.last_seen,
            wikipedia_title = COALESCE(excluded.wikipedia_title, topics.wikipedia_title)
        """,
        (topic_id, canonical_name, wikipedia_title, ts, ts),
    )


def touch_topic(conn: sqlite3.Connection, topic_id: str) -> None:
    conn.execute(
        "UPDATE topics SET last_seen = ? WHERE topic_id = ?",
        (now_utc(), topic_id),
    )


def get_topic(conn: sqlite3.Connection, topic_id: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM topics WHERE topic_id = ?", (topic_id,)
    ).fetchone()


def get_all_topics(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM topics").fetchall()


def set_low_confidence(conn: sqlite3.Connection, topic_id: str, flag: int) -> None:
    conn.execute(
        "UPDATE topics SET low_confidence = ? WHERE topic_id = ?",
        (flag, topic_id),
    )


# ---------------------------------------------------------------------------
# Aliases
# ---------------------------------------------------------------------------

def add_alias(
    conn: sqlite3.Connection, topic_id: str, alias: str, source: str | None = None
) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO topic_aliases(topic_id, alias, source) VALUES (?,?,?)",
        (topic_id, alias, source),
    )


def find_by_alias(conn: sqlite3.Connection, alias: str) -> str | None:
    row = conn.execute(
        "SELECT topic_id FROM topic_aliases WHERE alias = ?", (alias,)
    ).fetchone()
    return row["topic_id"] if row else None


def get_all_aliases(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    """Returns list of (alias, topic_id) pairs."""
    rows = conn.execute("SELECT alias, topic_id FROM topic_aliases").fetchall()
    return [(r["alias"], r["topic_id"]) for r in rows]


# ---------------------------------------------------------------------------
# Events (raw ingest)
# ---------------------------------------------------------------------------

def insert_event(
    conn: sqlite3.Connection,
    *,
    source: str,
    external_id: str | None,
    ts: str,
    title: str | None,
    url: str | None,
    topic_id: str | None = None,
    raw: dict[str, Any] | None = None,
) -> int | None:
    """Insert event; returns new rowid or None if duplicate."""
    try:
        cur = conn.execute(
            """
            INSERT INTO events(source, external_id, ts, title, url, topic_id, raw)
            VALUES (?,?,?,?,?,?,?)
            """,
            (
                source,
                external_id,
                ts,
                title,
                url,
                topic_id,
                json.dumps(raw) if raw else None,
            ),
        )
        return cur.lastrowid
    except sqlite3.IntegrityError:
        return None  # duplicate


def assign_topic(conn: sqlite3.Connection, event_id: int, topic_id: str) -> None:
    conn.execute(
        "UPDATE events SET topic_id = ? WHERE id = ?", (topic_id, event_id)
    )


def get_unmatched_events(
    conn: sqlite3.Connection, limit: int = 500
) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM events WHERE topic_id IS NULL ORDER BY ts DESC LIMIT ?",
        (limit,),
    ).fetchall()


def purge_old_events(conn: sqlite3.Connection, retain_days: int = 30) -> int:
    cutoff = datetime.now(timezone.utc)
    cutoff_str = cutoff.strftime(f"%Y-%m-%dT%H:%M:%SZ").replace(
        cutoff.strftime("%Y-%m-%dT%H:%M:%SZ"),
        f"{cutoff.year - (1 if cutoff.timetuple().tm_yday <= retain_days else 0)}"
        f"-{cutoff.strftime('%m-%dT%H:%M:%SZ')}",
    )
    # simpler: use SQLite date arithmetic
    cur = conn.execute(
        "DELETE FROM events WHERE ts < datetime('now', ? || ' days')",
        (f"-{retain_days}",),
    )
    return cur.rowcount


# ---------------------------------------------------------------------------
# Signals (hourly time-series)
# ---------------------------------------------------------------------------

def upsert_signal(
    conn: sqlite3.Connection,
    topic_id: str,
    bucket_ts: str,
    source: str,
    signal: str,
    value: float,
) -> None:
    conn.execute(
        """
        INSERT INTO signals(topic_id, bucket_ts, source, signal, value)
        VALUES (?,?,?,?,?)
        ON CONFLICT(topic_id, bucket_ts, source, signal)
        DO UPDATE SET value = value + excluded.value
        """,
        (topic_id, bucket_ts, source, signal, value),
    )


def get_signal_history(
    conn: sqlite3.Connection,
    topic_id: str,
    source: str,
    signal: str,
    days: int = 7,
) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT bucket_ts, value FROM signals
        WHERE topic_id = ? AND source = ? AND signal = ?
          AND bucket_ts >= datetime('now', ? || ' days')
        ORDER BY bucket_ts
        """,
        (topic_id, source, signal, f"-{days}"),
    ).fetchall()


def get_signal_baseline(
    conn: sqlite3.Connection,
    topic_id: str,
    source: str,
    signal: str,
    days: int = 7,
    current_hour_of_day: int | None = None,
) -> tuple[float, float]:
    """Return (mean, std) of signal values over the baseline window.
    If current_hour_of_day is given, restrict to matching hours (rhythm correction).
    """
    if current_hour_of_day is not None:
        rows = conn.execute(
            """
            SELECT value FROM signals
            WHERE topic_id = ? AND source = ? AND signal = ?
              AND bucket_ts >= datetime('now', ? || ' days')
              AND CAST(strftime('%H', bucket_ts) AS INTEGER) = ?
            """,
            (topic_id, source, signal, f"-{days}", current_hour_of_day),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT value FROM signals
            WHERE topic_id = ? AND source = ? AND signal = ?
              AND bucket_ts >= datetime('now', ? || ' days')
            """,
            (topic_id, source, signal, f"-{days}"),
        ).fetchall()

    values = [r["value"] for r in rows]
    if len(values) < 2:
        return 0.0, 1.0  # no baseline yet; avoid division by zero
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return mean, var ** 0.5


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------

def insert_score(
    conn: sqlite3.Connection,
    topic_id: str,
    ts: str,
    wikipedia_score: float = 0.0,
    news_score: float = 0.0,
    reddit_score: float = 0.0,
    search_score: float = 0.0,
    attention_index: float = 0.0,
    ewma: float | None = None,
    momentum: float | None = None,
    anomaly_z: float | None = None,
    emerging: int = 0,
) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO scores(
            topic_id, ts, wikipedia_score, news_score, reddit_score, search_score,
            attention_index, ewma, momentum, anomaly_z, emerging
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            topic_id, ts, wikipedia_score, news_score, reddit_score, search_score,
            attention_index, ewma, momentum, anomaly_z, emerging,
        ),
    )


def get_latest_score(
    conn: sqlite3.Connection, topic_id: str
) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM scores WHERE topic_id = ? ORDER BY ts DESC LIMIT 1",
        (topic_id,),
    ).fetchone()


def get_score_history(
    conn: sqlite3.Connection, topic_id: str, hours: int = 24
) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT * FROM scores
        WHERE topic_id = ? AND ts >= datetime('now', ? || ' hours')
        ORDER BY ts
        """,
        (topic_id, f"-{hours}"),
    ).fetchall()


def get_top_topics(
    conn: sqlite3.Connection, limit: int = 50
) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM top_topics LIMIT ?", (limit,)
    ).fetchall()


def get_emerging_topics(
    conn: sqlite3.Connection, limit: int = 20
) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM emerging_topics LIMIT ?", (limit,)
    ).fetchall()

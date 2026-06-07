"""
Scoring, normalization, and momentum computation.

Per topic per scoring run:
  1. Sum signals from the current hour bucket into per-source raw values.
  2. Normalize each source signal to 0-1 using a rolling z-score -> logistic squash.
  3. Composite as a weighted average -> attention_index.
  4. Compute momentum as attention_index - EWMA(trailing attention_index).
  5. Compute anomaly_z and set emerging flag.
"""
from __future__ import annotations

import logging
import math
import sqlite3
from datetime import datetime, timezone
from typing import Any

import storage

log = logging.getLogger(__name__)


def _logistic(z: float, k: float = 1.0, midpoint: float = 2.0) -> float:
    """Squash a z-score to (0, 1) via a logistic function."""
    try:
        return 1.0 / (1.0 + math.exp(-k * (z - midpoint)))
    except OverflowError:
        return 0.0 if z < midpoint else 1.0


def _zscore_normalize(
    conn: sqlite3.Connection,
    topic_id: str,
    source: str,
    signal: str,
    current_value: float,
    cfg: dict[str, Any],
) -> float:
    """
    Compute z-score of current_value against its own rolling baseline,
    then squash to 0-1 with a logistic.
    Returns 0.0 if no baseline exists yet (cold start).
    """
    baseline_days = cfg.get("baseline_days", 7)
    k = cfg.get("logistic_k", 1.0)
    midpoint = cfg.get("logistic_midpoint", 2.0)
    current_hour = datetime.now(timezone.utc).hour

    mean, std = storage.get_signal_baseline(
        conn, topic_id, source, signal,
        days=baseline_days, current_hour_of_day=current_hour,
    )
    if std < 1e-9:
        # No variance in baseline: use raw value relative to mean
        if current_value <= mean:
            return 0.0
        return _logistic(1.0, k, midpoint)

    z = (current_value - mean) / std
    return _logistic(z, k, midpoint)


def _compute_wikipedia_score(
    conn: sqlite3.Connection,
    topic_id: str,
    bucket: str,
    cfg: dict[str, Any],
) -> float:
    """Normalize edit_count and pageviews signals, average them."""
    scores: list[float] = []

    for signal in ("edit_count", "pageviews", "unique_editors"):
        rows = conn.execute(
            "SELECT value FROM signals WHERE topic_id=? AND source='wikipedia' "
            "AND signal=? AND bucket_ts=?",
            (topic_id, signal, bucket),
        ).fetchone()
        if rows:
            s = _zscore_normalize(conn, topic_id, "wikipedia", signal, rows["value"], cfg)
            scores.append(s)

    return sum(scores) / len(scores) if scores else 0.0


def _compute_news_score(
    conn: sqlite3.Connection,
    topic_id: str,
    bucket: str,
    cfg: dict[str, Any],
) -> float:
    scores: list[float] = []
    for signal in ("article_count", "publisher_count"):
        rows = conn.execute(
            "SELECT value FROM signals WHERE topic_id=? AND source='news' "
            "AND signal=? AND bucket_ts=?",
            (topic_id, signal, bucket),
        ).fetchone()
        if rows:
            s = _zscore_normalize(conn, topic_id, "news", signal, rows["value"], cfg)
            scores.append(s)
    return sum(scores) / len(scores) if scores else 0.0


def _compute_reddit_score(
    conn: sqlite3.Connection,
    topic_id: str,
    bucket: str,
    cfg: dict[str, Any],
) -> float:
    scores: list[float] = []
    for signal in ("post_velocity", "comment_velocity", "score_growth"):
        rows = conn.execute(
            "SELECT value FROM signals WHERE topic_id=? AND source='reddit' "
            "AND signal=? AND bucket_ts=?",
            (topic_id, signal, bucket),
        ).fetchone()
        if rows:
            s = _zscore_normalize(conn, topic_id, "reddit", signal, rows["value"], cfg)
            scores.append(s)
    return sum(scores) / len(scores) if scores else 0.0


def _update_ewma(
    conn: sqlite3.Connection,
    topic_id: str,
    current_index: float,
    alpha: float,
) -> float:
    """Compute new EWMA from the latest stored score."""
    prev = storage.get_latest_score(conn, topic_id)
    if prev is None or prev["ewma"] is None:
        return current_index  # cold start: initialize EWMA to current value
    return alpha * current_index + (1 - alpha) * prev["ewma"]


def _compute_anomaly_z(
    conn: sqlite3.Connection,
    topic_id: str,
    current_attention: float,
    baseline_days: int,
    current_hour: int,
) -> float:
    """
    Compute z-score of current attention_index against historical attention_index values.
    Uses scores table (not signals) for the composite-level anomaly.
    """
    rows = conn.execute(
        """
        SELECT attention_index FROM scores
        WHERE topic_id = ?
          AND ts >= datetime('now', ? || ' days')
          AND CAST(strftime('%H', ts) AS INTEGER) = ?
        ORDER BY ts DESC
        """,
        (topic_id, f"-{baseline_days}", current_hour),
    ).fetchall()

    values = [r["attention_index"] for r in rows]
    if len(values) < 3:
        return 0.0

    mean = sum(values) / len(values)
    std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
    if std < 1e-9:
        return 0.0
    return (current_attention - mean) / std


def run_scoring(
    conn: sqlite3.Connection,
    weights: dict[str, float],
    scoring_cfg: dict[str, Any],
) -> int:
    """
    Score all topics that have signal data in the current hour bucket.
    Returns the number of topics scored.
    """
    bucket = storage.hour_bucket()
    ts = storage.now_utc()
    zscore_threshold = scoring_cfg.get("zscore_threshold", 2.5)
    alpha = scoring_cfg.get("ewma_alpha", 0.3)
    baseline_days = scoring_cfg.get("baseline_days", 7)
    min_events = scoring_cfg.get("min_events_for_confidence", 10)
    current_hour = datetime.now(timezone.utc).hour

    # Find all topics with any signal in this bucket
    topic_rows = conn.execute(
        "SELECT DISTINCT topic_id FROM signals WHERE bucket_ts = ?", (bucket,)
    ).fetchall()

    scored = 0
    for row in topic_rows:
        tid = row["topic_id"]
        try:
            wp = _compute_wikipedia_score(conn, tid, bucket, scoring_cfg)
            nw = _compute_news_score(conn, tid, bucket, scoring_cfg)
            rd = _compute_reddit_score(conn, tid, bucket, scoring_cfg)
            sr = 0.0  # search: disabled / delayed

            attention = (
                weights.get("wikipedia", 0.40) * wp
                + weights.get("news", 0.35) * nw
                + weights.get("reddit", 0.20) * rd
                + weights.get("search", 0.05) * sr
            )

            ewma = _update_ewma(conn, tid, attention, alpha)
            momentum = attention - ewma
            anomaly_z = _compute_anomaly_z(conn, tid, attention, baseline_days, current_hour)
            emerging = 1 if anomaly_z >= zscore_threshold else 0

            storage.insert_score(
                conn, tid, ts,
                wikipedia_score=wp,
                news_score=nw,
                reddit_score=rd,
                search_score=sr,
                attention_index=attention,
                ewma=ewma,
                momentum=momentum,
                anomaly_z=anomaly_z,
                emerging=emerging,
            )

            # Clear low_confidence once enough events exist
            event_count = conn.execute(
                "SELECT COUNT(*) AS n FROM events WHERE topic_id = ?", (tid,)
            ).fetchone()["n"]
            if event_count >= min_events:
                storage.set_low_confidence(conn, tid, 0)

            scored += 1
        except Exception as exc:
            log.warning("Scoring failed for topic %s: %s", tid, exc)

    conn.commit()
    log.info("Scored %d topics for bucket %s", scored, bucket)
    return scored

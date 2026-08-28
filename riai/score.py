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

import bisect
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


# A topic without a usable baseline is ranked against the field instead of
# against itself, and confined to the bottom half of the range. The two
# questions are not the same: a z-score says "unusual for this topic", the
# fallback only says "large compared with everything else active right now".
# Capping at the logistic midpoint means a topic we cannot call unusual never
# outranks one we can.
_FALLBACK_CEILING = 0.5

Distributions = dict[tuple[str, str], list[float]]


def _load_bucket_distributions(conn: sqlite3.Connection, bucket: str) -> Distributions:
    """Sorted values per (source, signal) across every topic in this bucket.

    One query for the whole scoring run -- the per-topic alternative was
    thousands of aggregate scans over the same rows.
    """
    dists: Distributions = {}
    for row in conn.execute(
        "SELECT source, signal, value FROM signals WHERE bucket_ts = ?", (bucket,)
    ):
        dists.setdefault((row["source"], row["signal"]), []).append(row["value"])
    for values in dists.values():
        values.sort()
    return dists


def _percentile_rank(dists: Distributions, source: str, signal: str, value: float) -> float:
    """Where `value` falls among all topics' values for this signal, in [0,1].

    Ties share the midpoint of the range they span, so a signal where every
    topic reports the same number lands everyone at 0.5 rather than at 0 --
    uninformative data produces a tie rather than a false ordering.
    """
    values = dists.get((source, signal))
    if not values:
        return 0.0
    lo = bisect.bisect_left(values, value)
    hi = bisect.bisect_right(values, value)
    return (lo + hi) / (2 * len(values))


def _zscore_normalize(
    conn: sqlite3.Connection,
    topic_id: str,
    source: str,
    signal: str,
    current_value: float,
    cfg: dict[str, Any],
    dists: Distributions,
) -> float:
    """
    Compute z-score of current_value against its own rolling baseline,
    then squash to 0-1 with a logistic.

    Two cases have no usable baseline, and both used to return a constant --
    0.15 with no history, 0.269 when the history had no variance. Between them
    they covered most topics, so the index collapsed to a handful of values and
    ranked by which branch a topic fell into: one live run had a 14-way tie for
    sixth place. Zero variance is not rare either. A low-traffic article edited
    once an hour, every hour, has a standard deviation of zero permanently, so
    that branch never resolves with more history.

    Both now fall back to the topic's rank against the field. Raw values are
    never treated as z-scores -- pageview counts in the tens of thousands would
    otherwise produce absurd ones.
    """
    baseline_days = cfg.get("baseline_days", 7)
    k = cfg.get("logistic_k", 1.0)
    midpoint = cfg.get("logistic_midpoint", 2.0)
    current_hour = datetime.now(timezone.utc).hour

    mean, std = storage.get_signal_baseline(
        conn, topic_id, source, signal,
        days=baseline_days, current_hour_of_day=current_hour,
    )
    if mean is None or std < 1e-9:
        return _FALLBACK_CEILING * _percentile_rank(dists, source, signal, current_value)

    z = (current_value - mean) / std
    return _logistic(z, k, midpoint)


def _compute_wikipedia_score(
    conn: sqlite3.Connection,
    topic_id: str,
    bucket: str,
    cfg: dict[str, Any],
    dists: Distributions,
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
            s = _zscore_normalize(conn, topic_id, "wikipedia", signal, rows["value"], cfg, dists)
            scores.append(s)

    return sum(scores) / len(scores) if scores else 0.0


def _compute_news_score(
    conn: sqlite3.Connection,
    topic_id: str,
    bucket: str,
    cfg: dict[str, Any],
    dists: Distributions,
) -> float:
    scores: list[float] = []
    for signal in ("article_count", "publisher_count"):
        rows = conn.execute(
            "SELECT value FROM signals WHERE topic_id=? AND source='news' "
            "AND signal=? AND bucket_ts=?",
            (topic_id, signal, bucket),
        ).fetchone()
        if rows:
            s = _zscore_normalize(conn, topic_id, "news", signal, rows["value"], cfg, dists)
            scores.append(s)
    return sum(scores) / len(scores) if scores else 0.0


def _compute_reddit_score(
    conn: sqlite3.Connection,
    topic_id: str,
    bucket: str,
    cfg: dict[str, Any],
    dists: Distributions,
) -> float:
    scores: list[float] = []
    for signal in ("post_velocity", "comment_velocity", "score_growth"):
        rows = conn.execute(
            "SELECT value FROM signals WHERE topic_id=? AND source='reddit' "
            "AND signal=? AND bucket_ts=?",
            (topic_id, signal, bucket),
        ).fetchone()
        if rows:
            s = _zscore_normalize(conn, topic_id, "reddit", signal, rows["value"], cfg, dists)
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
        f"""
        SELECT attention_index FROM scores
        WHERE topic_id = ?
          AND ts >= {storage.cutoff_sql('days')}
          AND CAST(strftime('%H', ts) AS INTEGER) = ?
        ORDER BY ts DESC
        """,
        (topic_id, f"-{baseline_days}", current_hour),
    ).fetchall()

    values = [r["attention_index"] for r in rows]
    # Require at least 6 samples (~6 hours at hourly scoring) before trusting the z-score.
    # With fewer samples the std is too noisy and any small spike looks anomalous.
    if len(values) < 6:
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

    # Loaded once per run: every topic without a usable baseline is ranked
    # against this same snapshot of the field.
    dists = _load_bucket_distributions(conn, bucket)

    scored = 0
    for row in topic_rows:
        tid = row["topic_id"]
        try:
            wp = _compute_wikipedia_score(conn, tid, bucket, scoring_cfg, dists)
            nw = _compute_news_score(conn, tid, bucket, scoring_cfg, dists)
            rd = _compute_reddit_score(conn, tid, bucket, scoring_cfg, dists)
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

"""
RIAI poller — main entry point.

Runs the full pipeline:
  - Long-lived SSE connection to Wikipedia EventStreams (separate thread)
  - Periodic polling of Wikipedia Pageviews, News (GDELT + RSS), Reddit
  - Topic extraction + matching after each batch
  - Scoring + momentum on a configurable cadence
  - Static dashboard regeneration

Usage:
  cd riai/
  python poller.py [--config config.yaml] [--db riai.db]
"""
from __future__ import annotations

import argparse
import logging
import signal
import sqlite3
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

import storage
import score as scoring
import extract
import match as matcher
import enrich
import summarize
from sources import wikipedia as wp_source
from sources import news as news_source
from sources import reddit as reddit_source
from dashboard import build_static

log = logging.getLogger(__name__)

_SHUTDOWN = threading.Event()


def _load_config(path: str) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text())


# ---------------------------------------------------------------------------
# Ingest helpers
# ---------------------------------------------------------------------------

def _ingest_event(
    conn: sqlite3.Connection,
    *,
    source: str,
    external_id: str | None,
    ts: str,
    title: str,
    url: str | None,
    raw: dict[str, Any] | None,
    matching_cfg: dict[str, Any],
    wikipedia_title: str | None = None,
) -> str | None:
    """Persist raw event, match/create topic, accumulate signal bucket."""
    if not title:
        return None

    candidates = extract.extract_topics(title, source=source)
    if not candidates:
        return None

    best = candidates[0]
    try:
        topic_id = matcher.match_or_create(
            conn,
            candidate_text=best["text"],
            wikipedia_title=wikipedia_title,
            cfg=matching_cfg,
            source=source,
        )
    except Exception as exc:
        log.warning("match_or_create failed: %s", exc)
        return None

    event_id = storage.insert_event(
        conn,
        source=source,
        external_id=external_id,
        ts=ts,
        title=title,
        url=url,
        topic_id=topic_id,
        raw=raw,
    )

    return topic_id


def _bump_signal(
    conn: sqlite3.Connection,
    topic_id: str,
    source: str,
    signal: str,
    value: float = 1.0,
    bucket: str | None = None,
) -> None:
    b = bucket or storage.hour_bucket()
    storage.upsert_signal(conn, topic_id, b, source, signal, value)


# ---------------------------------------------------------------------------
# Wikipedia EventStreams thread
# ---------------------------------------------------------------------------

def _wikipedia_stream_thread(
    db_path: str,
    cfg: dict[str, Any],
    matching_cfg: dict[str, Any],
    last_event_id_holder: list[str | None],
) -> None:
    # Each thread gets its own connection. Sharing one connection across threads
    # causes exception state from one thread's failed statements to surface in
    # the other thread. WAL mode handles concurrent writers from separate connections.
    conn = storage.open_db(db_path)

    # Batch commits: flush every 20 events or every 5 seconds, whichever comes first.
    # Committing on every single edit (can be 30+/sec) holds the write lock too
    # frequently and causes "database is locked" on the main thread.
    _BATCH_SIZE = 20
    _BATCH_SECS = 5.0
    batch_count = 0
    last_commit_time = time.monotonic()

    def on_event(payload: dict[str, Any]) -> None:
        nonlocal batch_count, last_commit_time

        ev = wp_source.parse_edit_event(payload)
        if ev is None:
            return
        if ev["is_bot"]:
            return

        topic_id = _ingest_event(
            conn,
            source="wikipedia",
            external_id=ev["rev_id"],
            ts=ev["ts"],
            title=ev["title"],
            url=ev["url"],
            raw=None,
            matching_cfg=matching_cfg,
            wikipedia_title=ev["title"],
        )
        if topic_id:
            _bump_signal(conn, topic_id, "wikipedia", "edit_count")

        batch_count += 1
        now = time.monotonic()
        if batch_count >= _BATCH_SIZE or (now - last_commit_time) >= _BATCH_SECS:
            conn.commit()
            batch_count = 0
            last_commit_time = now

    log.info("Wikipedia EventStreams thread starting")
    try:
        wp_source.stream_edits(
            cfg=cfg.get("wikipedia", {}),
            on_event=on_event,
            last_event_id=last_event_id_holder[0],
        )
    except Exception as exc:
        log.error("EventStreams thread crashed: %s", exc)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Pageviews poll
# ---------------------------------------------------------------------------

def _poll_pageviews(
    conn: sqlite3.Connection,
    cfg: dict[str, Any],
    matching_cfg: dict[str, Any],
) -> None:
    log.info("Polling Wikipedia top pageviews...")
    top = wp_source.fetch_top_pageviews(cfg.get("wikipedia", {}))
    bucket = storage.hour_bucket()
    for item in top:
        topic_id = _ingest_event(
            conn,
            source="wikipedia",
            external_id=f"pv-{item['title']}-{bucket}",
            ts=bucket,
            title=item["title"],
            url=None,
            raw=None,
            matching_cfg=matching_cfg,
            wikipedia_title=item["title"],
        )
        if topic_id:
            _bump_signal(conn, topic_id, "wikipedia", "pageviews", float(item["views"]), bucket)
    conn.commit()
    log.info("Pageviews: ingested %d articles", len(top))


# ---------------------------------------------------------------------------
# News poll
# ---------------------------------------------------------------------------

def _poll_rss(
    conn: sqlite3.Connection,
    cfg: dict[str, Any],
    matching_cfg: dict[str, Any],
) -> None:
    news_cfg = cfg.get("news", {})
    bucket = storage.hour_bucket()
    log.info("Polling RSS feeds...")
    articles = news_source.poll_all_rss(news_cfg)
    topic_publisher_map: dict[str, set[str]] = {}
    for article in articles:
        if not article["title"]:
            continue
        topic_id = _ingest_event(
            conn,
            source="news",
            external_id=article["url"] or None,
            ts=article["ts"],
            title=article["title"],
            url=article["url"],
            raw=None,
            matching_cfg=matching_cfg,
        )
        if topic_id:
            _bump_signal(conn, topic_id, "news", "article_count", 1.0, bucket)
            if topic_id not in topic_publisher_map:
                topic_publisher_map[topic_id] = set()
            topic_publisher_map[topic_id].add(article.get("publisher", ""))
    for tid, publishers in topic_publisher_map.items():
        _bump_signal(conn, tid, "news", "publisher_count", float(len(publishers)), bucket)
    conn.commit()
    log.info("RSS: %d articles", len(articles))


def _poll_gdelt(
    conn: sqlite3.Connection,
    cfg: dict[str, Any],
    matching_cfg: dict[str, Any],
) -> None:
    news_cfg = cfg.get("news", {})
    bucket = storage.hour_bucket()
    log.info("Polling GDELT...")
    mentions = news_source.fetch_gdelt_mentions(news_cfg)
    for mention in mentions:
        if not mention.get("title"):
            continue
        topic_id = _ingest_event(
            conn,
            source="news",
            external_id=mention["url"] or None,
            ts=mention["ts"],
            title=mention["title"],
            url=mention["url"],
            raw=None,
            matching_cfg=matching_cfg,
        )
        if topic_id:
            _bump_signal(conn, topic_id, "news", "article_count", 1.0, bucket)
    conn.commit()
    log.info("GDELT: %d mentions", len(mentions))


# ---------------------------------------------------------------------------
# Reddit poll
# ---------------------------------------------------------------------------

def _poll_reddit(
    conn: sqlite3.Connection,
    cfg: dict[str, Any],
    matching_cfg: dict[str, Any],
) -> None:
    reddit_cfg = cfg.get("reddit", {})
    bucket = storage.hour_bucket()

    log.info("Polling Reddit...")
    posts = reddit_source.poll_all_subreddits(reddit_cfg)
    for post in posts:
        if not post["title"]:
            continue
        topic_id = _ingest_event(
            conn,
            source="reddit",
            external_id=post["post_id"],
            ts=post["ts"],
            title=post["title"],
            url=post["url"],
            raw=None,
            matching_cfg=matching_cfg,
        )
        if topic_id:
            _bump_signal(conn, topic_id, "reddit", "post_velocity", 1.0, bucket)
            _bump_signal(conn, topic_id, "reddit", "comment_velocity", float(post.get("num_comments", 0)), bucket)
            _bump_signal(conn, topic_id, "reddit", "score_growth", float(post.get("score", 0)), bucket)

    conn.commit()
    log.info("Reddit: %d posts", len(posts))


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(config_path: str = "config.yaml", db_path: str | None = None) -> None:
    cfg = _load_config(config_path)
    db = db_path or cfg.get("db_path", "riai.db")
    conn = storage.open_db(db)

    matching_cfg = cfg.get("matching", {})
    weights = cfg.get("weights", {})
    scoring_cfg = cfg.get("scoring", {})
    dashboard_cfg = cfg.get("dashboard", {})
    retention_cfg = cfg.get("retention", {})

    poll_interval_reddit = cfg.get("reddit", {}).get("poll_minutes", 5) * 60
    poll_interval_rss = cfg.get("news", {}).get("rss_poll_minutes", 5) * 60
    poll_interval_gdelt = cfg.get("news", {}).get("gdelt_poll_minutes", 30) * 60
    poll_interval_pv = cfg.get("wikipedia", {}).get("pageviews_poll_minutes", 60) * 60
    poll_interval_dashboard = dashboard_cfg.get("regenerate_minutes", 5) * 60
    poll_interval_summarize = cfg.get("summarize", {}).get("interval_minutes", 10) * 60

    last_event_id: list[str | None] = [storage.get_meta(conn, "wp_last_event_id")]

    # Start EventStreams SSE in background thread (runs forever).
    # Passes db path, not conn — thread opens its own connection to avoid
    # shared-connection state corruption between threads.
    wp_thread = threading.Thread(
        target=_wikipedia_stream_thread,
        args=(db, cfg, matching_cfg, last_event_id),
        daemon=True,
        name="wp-eventstreams",
    )
    wp_thread.start()

    # Track last poll times
    last: dict[str, float] = {
        "pageviews": 0.0,
        "rss": 0.0,
        "gdelt": 0.0,
        "reddit": 0.0,
        "score": 0.0,
        "summarize": 0.0,
        "dashboard": 0.0,
        "purge": 0.0,
    }

    log.info("RIAI poller started. DB: %s", db)

    def _handle_signal(*_):
        log.info("Shutdown signal received")
        _SHUTDOWN.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    while not _SHUTDOWN.is_set():
        now = time.monotonic()

        if now - last["pageviews"] >= poll_interval_pv:
            try:
                _poll_pageviews(conn, cfg, matching_cfg)
            except Exception as exc:
                log.error("Pageviews poll error: %s", exc)
            last["pageviews"] = now

        if now - last["rss"] >= poll_interval_rss:
            try:
                _poll_rss(conn, cfg, matching_cfg)
            except Exception as exc:
                log.error("RSS poll error: %s", exc)
            last["rss"] = now

        if now - last["gdelt"] >= poll_interval_gdelt:
            try:
                _poll_gdelt(conn, cfg, matching_cfg)
            except Exception as exc:
                log.error("GDELT poll error: %s", exc)
            last["gdelt"] = now

        if now - last["reddit"] >= poll_interval_reddit:
            try:
                _poll_reddit(conn, cfg, matching_cfg)
            except Exception as exc:
                log.error("Reddit poll error: %s", exc)
            last["reddit"] = now

        # Score every 5 minutes (same cadence as dashboard)
        if now - last["score"] >= poll_interval_dashboard:
            try:
                scoring.run_scoring(conn, weights, scoring_cfg)
            except Exception as exc:
                log.error("Scoring error: %s", exc)
            try:
                enrich.enrich_missing_titles(conn, cfg, limit=30)
            except Exception as exc:
                log.error("Enrichment error: %s", exc)
            last["score"] = now

        if now - last["summarize"] >= poll_interval_summarize:
            try:
                summarize.generate_summaries(conn, cfg)
            except Exception as exc:
                log.error("Summarize error: %s", exc)
            last["summarize"] = now

        if now - last["dashboard"] >= poll_interval_dashboard:
            try:
                build_static.build(
                    conn,
                    out_path=dashboard_cfg.get("output_path", "dashboard/index.html"),
                    top_n=dashboard_cfg.get("top_n", 50),
                    emerging_n=dashboard_cfg.get("emerging_n", 20),
                    sparkline_hours=dashboard_cfg.get("sparkline_hours", 24),
                )
            except Exception as exc:
                log.error("Dashboard build error: %s", exc)
            last["dashboard"] = now

        # Daily purge
        if now - last["purge"] >= 86400:
            try:
                n = storage.purge_old_events(conn, retain_days=retention_cfg.get("raw_events_days", 30))
                log.info("Purged %d old events", n)
                conn.commit()
            except Exception as exc:
                log.error("Purge error: %s", exc)
            last["purge"] = now

        _SHUTDOWN.wait(timeout=10)

    log.info("Poller shut down cleanly.")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="RIAI poller")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--db", default=None, help="Override db_path from config")
    args = parser.parse_args()
    run(config_path=args.config, db_path=args.db)


if __name__ == "__main__":
    main()

"""
News sources: GDELT (global news event stream) + RSS feeds.
GDELT is polled every ~15 min; RSS feeds every ~5 min.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

import feedparser
import requests

log = logging.getLogger(__name__)

# GDELT DOC 2.0 API — returns article list for a time window, no auth needed
# timespan=15min gives articles ingested in the last 15 minutes
_GDELT_DOC_API = (
    "https://api.gdeltproject.org/api/v2/doc/doc"
    "?query=&mode=artlist&maxrecords=250&format=json&timespan=15min"
)


# ---------------------------------------------------------------------------
# RSS feeds
# ---------------------------------------------------------------------------

def fetch_rss(feed_cfg: dict[str, str], ua: str) -> list[dict[str, Any]]:
    """
    Parse one RSS feed. Returns list of article dicts:
      {"title", "url", "ts", "source_name", "publisher"}
    """
    url = feed_cfg["url"]
    name = feed_cfg.get("name", urlparse(url).netloc)
    try:
        parsed = feedparser.parse(url, request_headers={"User-Agent": ua})
        items = []
        for entry in parsed.entries:
            title = entry.get("title", "").strip()
            link = entry.get("link", "")
            published = _rss_ts(entry)
            if title:
                items.append({
                    "title": title,
                    "url": link,
                    "ts": published,
                    "source_name": name,
                    "publisher": urlparse(link).netloc,
                })
        return items
    except Exception as exc:
        log.warning("RSS fetch failed [%s]: %s", name, exc)
        return []


def poll_all_rss(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    ua = cfg.get("user_agent", "RIAI/1.0")
    feeds = cfg.get("rss_feeds", [])
    results: list[dict[str, Any]] = []
    for feed in feeds:
        results.extend(fetch_rss(feed, ua))
    return results


def _rss_ts(entry: Any) -> str:
    """Extract publication time from a feedparser entry, fall back to now."""
    for attr in ("published_parsed", "updated_parsed", "created_parsed"):
        t = getattr(entry, attr, None)
        if t:
            try:
                return datetime(*t[:6], tzinfo=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            except Exception:
                pass
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# GDELT
# ---------------------------------------------------------------------------

def fetch_gdelt_mentions(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Fetch recent articles from the GDELT DOC 2.0 API (last 15 min).
    Returns list of {"title", "url", "ts", "source_name"}.
    No auth or file downloads needed — plain JSON.
    """
    ua = cfg.get("user_agent", "RIAI/1.0")
    try:
        resp = requests.get(
            _GDELT_DOC_API, headers={"User-Agent": ua}, timeout=20
        )
        if resp.status_code == 429:
            log.warning("GDELT rate-limited (429); skipping this cycle")
            return []
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError) as exc:
        log.warning("GDELT DOC API fetch failed: %s", exc)
        return []

    articles = data.get("articles") or []
    results: list[dict[str, Any]] = []
    for a in articles:
        title = (a.get("title") or "").strip()
        url = (a.get("url") or "").strip()
        if not title and url:
            title = _title_from_url(url)
        if not title:
            continue
        # seendate format: YYYYMMDDTHHMMSSZ
        ts = _gdelt_doc_ts(a.get("seendate", ""))
        results.append({
            "title": title,
            "url": url,
            "ts": ts,
            "source_name": a.get("domain", ""),
        })
    return results


def _gdelt_doc_ts(raw: str) -> str:
    """Parse GDELT DOC API timestamp YYYYMMDDTHHMMSSZ -> ISO UTC."""
    try:
        return datetime.strptime(raw, "%Y%m%dT%H%M%SZ").strftime("%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _title_from_url(url: str) -> str:
    """Best-effort: extract a human-readable title from a URL path."""
    try:
        path = urlparse(url).path.rstrip("/")
        slug = path.split("/")[-1]
        slug = slug.split(".")[0]  # strip extension
        return slug.replace("-", " ").replace("_", " ").strip()
    except Exception:
        return ""

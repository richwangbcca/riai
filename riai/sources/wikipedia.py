"""
Wikipedia sources:
  - EventStreams (SSE): real-time edits -> early signal
  - Pageviews API: hourly reader counts -> attention confirmation
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Callable, Iterator

import requests

log = logging.getLogger(__name__)

_EVENTSTREAMS_URL = "https://stream.wikimedia.org/v2/stream/recentchange"
_PAGEVIEWS_BASE = "https://wikimedia.org/api/rest_v1/metrics/pageviews"


# ---------------------------------------------------------------------------
# EventStreams (SSE)
# ---------------------------------------------------------------------------

def stream_edits(
    cfg: dict[str, Any],
    on_event: Callable[[dict[str, Any]], None],
    last_event_id: str | None = None,
) -> None:
    """
    Long-lived SSE loop. Calls on_event for each qualifying main-namespace
    English Wikipedia edit. Reconnects on errors with exponential backoff.
    Never returns unless interrupted.
    """
    url = cfg.get("eventstreams_url", _EVENTSTREAMS_URL)
    lang = cfg.get("lang", "en")
    excluded_ns = set(cfg.get("excluded_namespaces", []))
    ua = cfg.get("user_agent", "RIAI/1.0")
    headers = {"User-Agent": ua, "Accept": "text/event-stream"}
    if last_event_id:
        headers["Last-Event-ID"] = last_event_id

    backoff = 2
    while True:
        try:
            log.info("Connecting to Wikipedia EventStreams...")
            with requests.get(url, headers=headers, stream=True, timeout=90) as resp:
                resp.raise_for_status()
                backoff = 2  # reset on successful connection
                for raw_line in _iter_sse_lines(resp):
                    if not raw_line.startswith("data:"):
                        continue
                    try:
                        payload = json.loads(raw_line[5:].strip())
                    except json.JSONDecodeError:
                        continue

                    if payload.get("wiki") != f"{lang}wiki":
                        continue
                    if payload.get("namespace", -1) in excluded_ns:
                        continue
                    if payload.get("namespace", -1) != 0:
                        continue
                    if payload.get("type") not in ("edit", "new"):
                        continue

                    on_event(payload)

        except (requests.RequestException, OSError) as exc:
            log.warning("EventStreams connection lost: %s. Reconnecting in %ds", exc, backoff)
            time.sleep(backoff)
            backoff = min(backoff * 2, 120)


def _iter_sse_lines(resp: requests.Response) -> Iterator[str]:
    """Yield raw SSE text lines from a streaming response."""
    for raw_bytes in resp.iter_lines():
        if isinstance(raw_bytes, bytes):
            yield raw_bytes.decode("utf-8", errors="replace")
        else:
            yield raw_bytes


def parse_edit_event(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Extract the fields we care about from a recentchange SSE payload."""
    title = payload.get("title", "").strip()
    if not title:
        return None
    return {
        "title": title,
        "ts": _to_iso(payload.get("timestamp")),
        "rev_id": str(payload.get("revision", {}).get("new", "")),
        "url": payload.get("meta", {}).get("uri", ""),
        "is_bot": payload.get("bot", False),
        "is_new": payload.get("type") == "new",
        "comment": payload.get("comment", ""),
        "user": payload.get("user", ""),
    }


def _to_iso(ts: int | str | None) -> str:
    if ts is None:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return str(ts)


# ---------------------------------------------------------------------------
# Pageviews API
# ---------------------------------------------------------------------------

def fetch_pageviews(
    title: str,
    cfg: dict[str, Any],
    hours: int = 24,
) -> list[dict[str, Any]]:
    """
    Fetch hourly pageviews for a Wikipedia article over the last `hours` hours.
    Returns list of {"hour": "YYYY-MM-DDTHH:00:00Z", "views": int}.
    """
    base = cfg.get("pageviews_base", _PAGEVIEWS_BASE)
    lang = cfg.get("lang", "en")
    ua = cfg.get("user_agent", "RIAI/1.0")

    now = datetime.now(timezone.utc)
    # pageviews API uses YYYYMMDD/HH format
    end = now.strftime("%Y%m%d") + "/" + now.strftime("%H")
    # go back `hours` hours
    from datetime import timedelta
    start_dt = now - timedelta(hours=hours)
    start = start_dt.strftime("%Y%m%d") + "/" + start_dt.strftime("%H")

    safe_title = title.replace(" ", "_")
    url = (
        f"{base}/per-article/en.wikipedia/all-access/all-agents/"
        f"{requests.utils.quote(safe_title, safe='')}/hourly/{start}/{end}"
    )
    try:
        resp = requests.get(url, headers={"User-Agent": ua}, timeout=15)
        if resp.status_code == 404:
            return []  # article not found in pageviews
        resp.raise_for_status()
        items = resp.json().get("items", [])
        result = []
        for item in items:
            ts_str = item.get("timestamp", "")
            if len(ts_str) == 10:  # YYYYMMDDHH
                hour = f"{ts_str[:4]}-{ts_str[4:6]}-{ts_str[6:8]}T{ts_str[8:10]}:00:00Z"
            else:
                hour = ts_str
            result.append({"hour": hour, "views": item.get("views", 0)})
        return result
    except (requests.RequestException, ValueError) as exc:
        log.warning("Pageviews fetch failed for %r: %s", title, exc)
        return []


def fetch_top_pageviews(cfg: dict[str, Any], date: str | None = None) -> list[dict[str, Any]]:
    """
    Fetch the top 1000 viewed English Wikipedia articles for a given date (YYYY-MM-DD).
    Defaults to yesterday (most recent complete day).
    Returns list of {"title": str, "views": int, "rank": int}.
    """
    base = cfg.get("pageviews_base", _PAGEVIEWS_BASE)
    ua = cfg.get("user_agent", "RIAI/1.0")

    if date is None:
        from datetime import timedelta
        yesterday = datetime.now(timezone.utc) - timedelta(days=1)
        date = yesterday.strftime("%Y/%m/%d")
    else:
        date = date.replace("-", "/")

    url = f"{base}/top/en.wikipedia/all-access/{date}"
    try:
        resp = requests.get(url, headers={"User-Agent": ua}, timeout=15)
        resp.raise_for_status()
        articles = resp.json().get("items", [{}])[0].get("articles", [])
        return [
            {"title": a["article"].replace("_", " "), "views": a["views"], "rank": a["rank"]}
            for a in articles
            # filter out meta-pages that slipped through
            if not a["article"].startswith(("Main_Page", "Special:", "Wikipedia:"))
        ]
    except (requests.RequestException, ValueError, KeyError) as exc:
        log.warning("Top pageviews fetch failed: %s", exc)
        return []

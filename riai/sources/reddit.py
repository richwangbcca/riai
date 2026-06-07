"""
Reddit source: polling via Reddit's RSS feeds.
Reddit blocked unauthenticated JSON API access; RSS feeds remain public.
feedparser already handles RSS parsing, so no new dependency needed.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

import feedparser

log = logging.getLogger(__name__)

_BASE = "https://www.reddit.com"


def fetch_subreddit_rss(
    subreddit: str,
    cfg: dict[str, Any],
    limit: int = 25,
) -> list[dict[str, Any]]:
    """
    Fetch hot posts from r/<subreddit> via RSS.
    Returns list of {"title", "url", "ts", "subreddit", "post_id"}.
    """
    ua = cfg.get("user_agent", "RIAI/1.0")
    url = f"{_BASE}/r/{subreddit}/hot.rss?limit={limit}"
    try:
        parsed = feedparser.parse(url, request_headers={"User-Agent": ua})
        if parsed.get("status", 200) == 429:
            log.warning("Reddit rate-limited on r/%s, backing off", subreddit)
            time.sleep(60)
            return []
        posts = []
        for entry in parsed.entries:
            title = entry.get("title", "").strip()
            link = entry.get("link", "")
            if not title:
                continue
            ts = _rss_ts(entry)
            # Reddit RSS post IDs are in the <id> tag or derivable from URL
            post_id = entry.get("id", link).split("/comments/")[-1].split("/")[0]
            posts.append({
                "title": title,
                "url": link,
                "ts": ts,
                "subreddit": subreddit,
                "post_id": post_id,
                # RSS doesn't carry score/comment counts; use 0 as placeholder
                "score": 0,
                "num_comments": 0,
            })
        return posts
    except Exception as exc:
        log.warning("Reddit RSS fetch failed for r/%s: %s", subreddit, exc)
        return []


def _rss_ts(entry: Any) -> str:
    for attr in ("published_parsed", "updated_parsed"):
        t = getattr(entry, attr, None)
        if t:
            try:
                return datetime(*t[:6], tzinfo=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            except Exception:
                pass
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def poll_all_subreddits(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    subreddits = cfg.get("subreddits", [])
    limit = cfg.get("posts_per_sub", 25)
    results: list[dict[str, Any]] = []
    for sub in subreddits:
        posts = fetch_subreddit_rss(sub, cfg, limit=limit)
        results.extend(posts)
        time.sleep(1)  # stay polite
    return results

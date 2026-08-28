"""
Reddit source: polling via Reddit's RSS feeds.
Reddit blocked unauthenticated JSON API access; RSS feeds remain public.
feedparser already handles RSS parsing, so no new dependency needed.
"""
from __future__ import annotations

import logging
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
            # Never sleep here: this runs on the main poll loop, and blocking it
            # delays scoring, the dashboard rebuild and the config reload too.
            # The caller's own interval is the backoff.
            log.debug("Reddit rate-limited on r/%s, skipping this cycle", subreddit)
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


_next_sub = 0


def poll_next_subreddit(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Fetch one subreddit per call, cycling through the configured list.

    Reddit throttles this IP to roughly one RSS fetch per minute no matter how
    the requests are spaced -- measured 1 success in 6 at both 3s and 10s
    spacing. Fetching every subreddit each cycle therefore 429'd on all but one
    while the caller slept 60s per failure, stalling the whole poll loop for
    ~9 minutes to collect a single subreddit's worth of posts.

    One request per cycle stays under the throttle and never blocks. At the
    default 5-minute interval each subreddit comes round about every 50
    minutes; trim `reddit.subreddits` to tighten that.
    """
    global _next_sub
    subreddits = cfg.get("subreddits", [])
    if not subreddits:
        return []
    sub = subreddits[_next_sub % len(subreddits)]
    _next_sub += 1
    return fetch_subreddit_rss(sub, cfg, limit=cfg.get("posts_per_sub", 25))

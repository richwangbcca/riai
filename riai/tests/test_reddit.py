"""Reddit polling: one subreddit per cycle, never blocking the poll loop."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sources import reddit


def test_cycles_through_subreddits_one_per_call(monkeypatch):
    fetched = []
    monkeypatch.setattr(
        reddit, "fetch_subreddit_rss", lambda sub, cfg, limit: fetched.append(sub) or []
    )
    monkeypatch.setattr(reddit, "_next_sub", 0)

    cfg = {"subreddits": ["news", "worldnews", "science"]}
    for _ in range(4):
        reddit.poll_next_subreddit(cfg)

    assert fetched == ["news", "worldnews", "science", "news"]  # wraps around


def test_empty_subreddit_list_does_not_divide_by_zero(monkeypatch):
    monkeypatch.setattr(reddit, "_next_sub", 0)
    assert reddit.poll_next_subreddit({"subreddits": []}) == []


def test_rate_limited_fetch_returns_immediately(monkeypatch):
    """A 429 used to sleep(60) on the main loop, stalling scoring and the
    dashboard along with it."""
    class _Parsed(dict):
        entries = []

    monkeypatch.setattr(reddit.feedparser, "parse", lambda *a, **kw: _Parsed(status=429))

    start = time.monotonic()
    assert reddit.fetch_subreddit_rss("news", {}) == []
    assert time.monotonic() - start < 1.0  # used to be a hard 60s sleep

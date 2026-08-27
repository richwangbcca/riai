"""Tests for LLM summarization topic selection (no live API calls)."""
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import storage
import summarize


class _FakeResponse:
    def __init__(self, summaries):
        self._payload = {
            "candidates": [
                {"content": {"parts": [{"text": json.dumps(summaries)}]}}
            ]
        }
        self.text = json.dumps(self._payload)

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


_SCORE_TS = "2026-01-01T00:00:00Z"  # shared: the query only reads MAX(ts) from scores


def _seed(conn, topic_id, name, index, headlines):
    storage.upsert_topic(conn, topic_id, name)
    conn.execute("UPDATE topics SET low_confidence = 0 WHERE topic_id = ?", (topic_id,))
    storage.insert_score(conn, topic_id, _SCORE_TS, attention_index=index)
    for i, h in enumerate(headlines):
        storage.insert_event(
            conn,
            source="news",
            external_id=f"{topic_id}-{i}",
            ts=storage.now_utc(),
            title=h,
            url=None,
            topic_id=topic_id,
        )
    conn.commit()


def _run(monkeypatch, conn, cfg, n_summaries):
    """Run generate_summaries against a stubbed API; returns the prompt sent."""
    sent = {}

    def fake_post(url, json=None, **kwargs):
        sent["prompt"] = json["contents"][0]["parts"][0]["text"]
        sent["headers"] = kwargs.get("headers", {})
        sent["url"] = url
        return _FakeResponse([f"summary {i}" for i in range(n_summaries)])

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(summarize.requests, "post", fake_post)
    summarize.generate_summaries(conn, cfg)
    return sent.get("prompt")


def _fresh_db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    storage._apply_schema(conn)
    return conn


def test_topics_without_headlines_are_skipped(monkeypatch):
    conn = _fresh_db()
    _seed(conn, "has-news", "Has News", 0.9, ["Something specific happened"])
    _seed(conn, "no-news", "No News", 0.8, [])

    prompt = _run(monkeypatch, conn, {}, n_summaries=1)

    assert "Has News" in prompt
    assert "No News" not in prompt

    rows = dict(conn.execute("SELECT topic_id, summary FROM topic_summaries"))
    assert rows == {"has-news": "summary 0"}


def test_no_api_call_when_nothing_has_headlines(monkeypatch):
    conn = _fresh_db()
    _seed(conn, "no-news", "No News", 0.8, [])

    def explode(*a, **kw):
        raise AssertionError("should not call the API with no topics")

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(summarize.requests, "post", explode)
    assert summarize.generate_summaries(conn, {}) == 0


def test_top_n_comes_from_config(monkeypatch):
    conn = _fresh_db()
    for i in range(5):
        _seed(conn, f"t{i}", f"Topic {i}", 0.9 - i * 0.1, [f"Headline {i}"])

    prompt = _run(monkeypatch, conn, {"summarize": {"top_n": 2}}, n_summaries=2)

    assert "Topic 0" in prompt and "Topic 1" in prompt
    assert "Topic 2" not in prompt


def test_api_key_is_sent_as_header_not_in_url(monkeypatch):
    """The key must stay out of the URL — request URLs end up in warning logs."""
    conn = _fresh_db()
    _seed(conn, "has-news", "Has News", 0.9, ["Something happened"])

    sent = {}

    def fake_post(url, json=None, **kwargs):
        sent["url"] = url
        sent["headers"] = kwargs.get("headers", {})
        sent["params"] = kwargs.get("params")
        return _FakeResponse(["a summary"])

    monkeypatch.setenv("GEMINI_API_KEY", "secret-key-value")
    monkeypatch.setattr(summarize.requests, "post", fake_post)
    summarize.generate_summaries(conn, {})

    assert sent["headers"].get("x-goog-api-key") == "secret-key-value"
    assert "secret-key-value" not in sent["url"]
    assert not (sent["params"] or {})

"""
Batched LLM summarization: generates one-line "why is this trending?" blurbs
for the top N topics, using recent article headlines as context.

One API call per refresh cycle. Never per-event.
Requires GEMINI_API_KEY in the environment.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
from typing import Any

import requests

import storage

log = logging.getLogger(__name__)

_GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

_INSTRUCTIONS = """\
You are helping users of a news attention tracker understand why topics are trending.

For each topic below I will give you the topic name and up to 5 recent headlines \
associated with it. Write exactly one sentence (under 20 words) explaining why it \
is getting attention right now. Be specific — name the event, person, or development \
driving the attention. If the headlines are too vague to be specific, write a \
brief general description.

Respond with a JSON array of strings, one per topic, in the same order as the input.\
"""


def _fetch_topic_headlines(conn: sqlite3.Connection, topic_id: str, n: int = 5) -> list[str]:
    rows = conn.execute(
        f"""
        SELECT DISTINCT title FROM events
        WHERE topic_id = ? AND title IS NOT NULL
          AND ts >= {storage.cutoff_sql('hours')}
        ORDER BY ts DESC
        LIMIT ?
        """,
        (topic_id, "-24", n),
    ).fetchall()
    return [r["title"] for r in rows if r["title"]]


def _build_topic_block(name: str, headlines: list[str]) -> str:
    lines = [f"Topic: {name}", "Recent headlines:"]
    lines.extend(f"  - {h}" for h in headlines)
    return "\n".join(lines)


def generate_summaries(conn: sqlite3.Connection, cfg: dict[str, Any]) -> int:
    """
    Generate summaries for the top `summarize.top_n` topics by attention index.
    Topics with no recent headlines are skipped (no context = hallucinated blurb).
    Stores results in topic_summaries. Returns number of topics updated.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        log.warning("GEMINI_API_KEY not set; skipping summarization")
        return 0

    summarize_cfg = cfg.get("summarize", {})
    model = summarize_cfg.get("model", "gemini-3.6-flash")
    top_n = summarize_cfg.get("top_n", 20)

    # Fetch top topics from latest scoring run
    rows = conn.execute(
        """
        SELECT s.topic_id, t.canonical_name
        FROM scores s
        JOIN topics t USING (topic_id)
        WHERE s.ts = (SELECT MAX(ts) FROM scores)
          AND t.low_confidence = 0
        ORDER BY s.attention_index DESC
        LIMIT ?
        """,
        (top_n,),
    ).fetchall()

    topics: list[tuple[str, str]] = []
    topic_blocks: list[str] = []
    for r in rows:
        headlines = _fetch_topic_headlines(conn, r["topic_id"])
        if not headlines:
            continue
        topics.append((r["topic_id"], r["canonical_name"]))
        topic_blocks.append(_build_topic_block(r["canonical_name"], headlines))

    if not topics:
        log.info("No topics with recent headlines; skipping summarization")
        return 0

    dynamic_content = "\n\n".join(topic_blocks)
    prompt = f"{_INSTRUCTIONS}\n\n{dynamic_content}"

    try:
        response = requests.post(
            _GEMINI_URL.format(model=model),
            headers={"x-goog-api-key": api_key},  # not a query param: URLs land in logs
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "responseMimeType": "application/json",
                    "responseSchema": {"type": "ARRAY", "items": {"type": "STRING"}},
                },
            },
            timeout=30,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        log.warning("LLM summarization API call failed: %s", exc)
        return 0

    try:
        data = response.json()
        raw = data["candidates"][0]["content"]["parts"][0]["text"]
        summaries: list[str] = json.loads(raw)
    except (KeyError, IndexError, ValueError, json.JSONDecodeError) as exc:
        log.warning("Failed to parse LLM summary response: %s\nRaw: %.200s", exc, response.text)
        return 0

    if len(summaries) != len(topics):
        log.warning(
            "LLM returned %d summaries for %d topics; skipping", len(summaries), len(topics)
        )
        return 0

    now = storage.now_utc()
    for (tid, _name), summary in zip(topics, summaries):
        summary = summary.strip()
        if not summary:
            continue
        conn.execute(
            """
            INSERT INTO topic_summaries(topic_id, summary, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(topic_id) DO UPDATE SET summary = excluded.summary,
                                                updated_at = excluded.updated_at
            """,
            (tid, summary, now),
        )
    conn.commit()
    log.info("Generated summaries for %d topics via %s", len(summaries), model)
    return len(summaries)

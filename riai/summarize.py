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
import re
import sqlite3
from typing import Any

import requests

import storage

log = logging.getLogger(__name__)

_GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

_INSTRUCTIONS = """\
You are helping users of a news attention tracker understand why topics are trending.

For each topic below I will give you the topic name and whatever recent evidence we \
have for it. Write exactly one sentence (under 20 words) explaining why it is getting \
attention right now. Be specific — name the event, person, or development driving it.

The evidence comes in two forms, and they are not equally strong:

- "News/Reddit coverage" are published articles and posts, with a count of how many \
appeared in the last 24 hours. This is the strongest evidence available. A high count \
means something real happened, so lead with it and treat everything else as detail.

- "Recent Wikipedia edit summaries" are notes editors wrote describing their own \
changes to the article. Most edits are routine upkeep — typos, copyedits, formatting, \
wikilinks, categories, reverts, reference fixes. Upkeep is not why anything trends. \
Ignore those and use the substantive edits (a death date, a result, a new section, a \
rename). If upkeep is all there is, do not describe the cleanup; say attention is \
rising without a clear cause.

The two are asymmetric. Heavy coverage is strong positive evidence. No coverage is \
not evidence of the opposite — plenty of topics draw real attention with nothing \
written about them yet, so a missing coverage section is normal and means only that \
you must not imply articles exist.

Never invent an event the evidence does not mention. If the evidence is too thin to \
say why, describe what the topic is and note that activity is rising — a vague but \
true sentence beats a specific but invented one.

Respond with a JSON array of strings, one per topic, in the same order as the input.\
"""


def _fetch_coverage(
    conn: sqlite3.Connection, topic_id: str, name: str, n: int = 5
) -> tuple[list[str], int]:
    """Recent news/Reddit headlines, newest first, plus how many there are.

    Wikipedia events are deliberately excluded: their title is the article name,
    not a headline, so "1958 in film" would be handed to the model as a news
    headline about "1977 in film". Wikipedia speaks through _fetch_edit_comments
    instead.

    The count matters separately from the sample. Heavy coverage is positive
    evidence that something actually happened, and the model can't infer that
    from a list capped at `n`. Light coverage is not evidence of the opposite —
    a topic can draw real attention with no articles at all.
    """
    rows = conn.execute(
        f"""
        SELECT title, MAX(ts) AS latest_ts
        FROM events
        WHERE topic_id = ? AND source IN ('news', 'reddit') AND title IS NOT NULL
          AND lower(title) != lower(?)
          AND ts >= {storage.cutoff_sql('hours')}
        GROUP BY title
        ORDER BY latest_ts DESC
        """,
        (topic_id, name, "-24"),
    ).fetchall()
    titles = [r["title"] for r in rows if r["title"]]
    return titles[:n], len(titles)


_SECTION_RE = re.compile(r"/\*\s*(.*?)\s*\*/")


def _fetch_edit_comments(
    conn: sqlite3.Connection, topic_id: str, n: int = 5
) -> list[str]:
    """Recent Wikipedia edit summaries — what editors say they changed.

    A topic can spike on edits and pageviews with no news coverage at all; that
    is a real attention signal, not noise. The edit summary is the only text
    those events carry, so it's the only honest answer to "why is this trending".
    """
    rows = conn.execute(
        f"""
        SELECT json_extract(raw, '$.comment') AS comment, MAX(ts) AS latest_ts
        FROM events
        WHERE topic_id = ? AND source = 'wikipedia' AND raw IS NOT NULL
          AND ts >= {storage.cutoff_sql('hours')}
        GROUP BY comment
        ORDER BY latest_ts DESC
        LIMIT ?
        """,
        (topic_id, "-24", n),
    ).fetchall()

    out = []
    for r in rows:
        # "/* Death */ add date" -> "Death: add date"; the section name is often
        # the most informative part, so keep it rather than stripping the marker.
        comment = _SECTION_RE.sub(r"\1:", r["comment"] or "").strip()
        if len(comment) > 2:
            out.append(comment)
    return out


def _build_topic_block(
    name: str, headlines: list[str], coverage_count: int, comments: list[str]
) -> str:
    lines = [f"Topic: {name}"]
    if headlines:
        lines.append(f"News/Reddit coverage ({coverage_count} in last 24h):")
        lines.extend(f"  - {h}" for h in headlines)
    if comments:
        lines.append("Recent Wikipedia edit summaries:")
        lines.extend(f"  - {c}" for c in comments)
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
        headlines, coverage = _fetch_coverage(
            conn, r["topic_id"], r["canonical_name"]
        )
        comments = _fetch_edit_comments(conn, r["topic_id"])
        if not headlines and not comments:
            continue
        topics.append((r["topic_id"], r["canonical_name"]))
        topic_blocks.append(
            _build_topic_block(r["canonical_name"], headlines, coverage, comments)
        )

    if not topics:
        log.info("No topics with recent context; skipping summarization")
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

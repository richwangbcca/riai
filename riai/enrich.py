"""
Wikipedia title enrichment for topics that were first seen via non-Wikipedia sources.

Queries the Wikipedia OpenSearch API to find the best-matching article title,
then updates the topic record and adds a Wikipedia alias.

Run from the poller on a slow cadence (e.g. once per scoring cycle) against
only the top-N topics that still lack a wikipedia_title. Never in the hot path.
"""
from __future__ import annotations

import logging
import sqlite3
import time
from typing import Any

import requests

log = logging.getLogger(__name__)

_SEARCH_URL = "https://en.wikipedia.org/w/api.php"


def _search_wikipedia_title(query: str, ua: str) -> str | None:
    """
    Use the Wikipedia OpenSearch API to find the best-matching article title.
    Returns the canonical title string, or None if no confident match found.
    """
    try:
        resp = requests.get(
            _SEARCH_URL,
            params={
                "action": "opensearch",
                "search": query,
                "limit": 3,
                "namespace": 0,
                "format": "json",
                "redirects": "resolve",
            },
            headers={"User-Agent": ua},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        # OpenSearch returns [query, [titles], [descriptions], [urls]]
        titles = data[1] if len(data) > 1 else []
        if not titles:
            return None
        return titles[0]  # first result is the best match
    except (requests.RequestException, ValueError, IndexError) as exc:
        log.debug("Wikipedia title search failed for %r: %s", query, exc)
        return None


def enrich_missing_titles(
    conn: sqlite3.Connection,
    cfg: dict[str, Any],
    limit: int = 30,
) -> int:
    """
    Find up to `limit` high-scoring topics without a wikipedia_title and try
    to resolve one. Returns the number of topics updated.

    Pulls from the latest scoring run so we only enrich topics that are
    actually visible, not every topic ever seen.
    """
    ua = cfg.get("wikipedia", {}).get("user_agent", "RIAI/1.0")

    rows = conn.execute(
        """
        SELECT t.topic_id, t.canonical_name
        FROM topics t
        JOIN scores s ON s.topic_id = t.topic_id
        WHERE t.wikipedia_title IS NULL
          AND s.ts = (SELECT MAX(ts) FROM scores)
        ORDER BY s.attention_index DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()

    updated = 0
    for row in rows:
        tid = row["topic_id"]
        name = row["canonical_name"]

        title = _search_wikipedia_title(name, ua)
        if not title:
            time.sleep(0.2)
            continue

        conn.execute(
            "UPDATE topics SET wikipedia_title = ? WHERE topic_id = ?",
            (title, tid),
        )
        # Add as alias so the matcher uses it next time
        conn.execute(
            "INSERT OR IGNORE INTO topic_aliases(topic_id, alias, source) VALUES (?,?,?)",
            (tid, title, "wikipedia_enrichment"),
        )
        conn.commit()
        updated += 1
        log.debug("Enriched %r -> %r", name, title)
        time.sleep(0.1)  # ~10 req/s, well within Wikipedia's limits

    if updated:
        log.info("Wikipedia enrichment: resolved titles for %d/%d topics", updated, len(rows))
    return updated

"""
Topic matching and canonicalization.
Anchors on Wikipedia titles; fuzzy-matches new topics to the existing registry.
Unmatched topics become new entries.

Pipeline:
  1. Exact alias lookup (O(1))
  2. Fuzzy match via RapidFuzz against all known aliases + canonical names
  3. Embedding cosine similarity for semantic matches (lazy-loaded)
  4. Unmatched -> new topic
"""
from __future__ import annotations

import hashlib
import logging
import re
import sqlite3
from typing import Any

log = logging.getLogger(__name__)

_embedder = None  # lazy-loaded sentence-transformer
_embeddings_cache: dict[str, Any] = {}  # topic_id -> embedding


def _get_embedder(model_name: str = "all-MiniLM-L6-v2"):
    global _embedder
    if _embedder is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embedder = SentenceTransformer(model_name)
            log.info("Loaded sentence-transformer: %s", model_name)
        except Exception as exc:
            log.warning("Sentence-transformer unavailable (%s); embedding match disabled", exc)
            _embedder = False
    return _embedder if _embedder else None


def slugify(text: str) -> str:
    """Convert a topic name to a stable slug for use as topic_id."""
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text)
    text = text.strip("-")
    return text[:80]


def make_topic_id(text: str) -> str:
    """Stable slug; add a hash suffix if slug alone would be ambiguous."""
    slug = slugify(text)
    if not slug:
        slug = hashlib.md5(text.encode()).hexdigest()[:8]
    return slug


def match_or_create(
    conn: sqlite3.Connection,
    candidate_text: str,
    wikipedia_title: str | None,
    cfg: dict[str, Any],
    *,
    source: str = "unknown",
) -> str:
    """
    Find the best matching topic_id for candidate_text, or create a new one.
    Returns the topic_id.
    """
    from storage import (
        add_alias, find_by_alias, get_all_aliases, upsert_topic,
        get_all_topics, touch_topic,
    )

    fuzzy_threshold = cfg.get("fuzzy_threshold", 85)
    embedding_threshold = cfg.get("embedding_threshold", 0.82)
    model_name = cfg.get("embedding_model", "all-MiniLM-L6-v2")

    text = candidate_text.strip()
    if not text:
        raise ValueError("candidate_text is empty")

    # -- 1. Exact alias lookup --
    topic_id = find_by_alias(conn, text)
    if topic_id:
        touch_topic(conn, topic_id)
        return topic_id

    # Also try Wikipedia title if provided
    if wikipedia_title and wikipedia_title != text:
        topic_id = find_by_alias(conn, wikipedia_title)
        if topic_id:
            add_alias(conn, topic_id, text, source=source)
            touch_topic(conn, topic_id)
            return topic_id

    # -- 2. Fuzzy match --
    try:
        from rapidfuzz import process as fuzz_process, fuzz
        all_aliases = get_all_aliases(conn)  # [(alias, topic_id)]
        if all_aliases:
            alias_texts = [a[0] for a in all_aliases]
            result = fuzz_process.extractOne(
                text, alias_texts, scorer=fuzz.token_sort_ratio
            )
            if result and result[1] >= fuzzy_threshold:
                matched_alias = result[0]
                topic_id = next(tid for alias, tid in all_aliases if alias == matched_alias)
                add_alias(conn, topic_id, text, source=source)
                if wikipedia_title:
                    add_alias(conn, topic_id, wikipedia_title, source="wikipedia")
                touch_topic(conn, topic_id)
                return topic_id
    except ImportError:
        log.debug("rapidfuzz not available; skipping fuzzy match")
    except Exception as exc:
        log.debug("Fuzzy match error: %s", exc)

    # -- 3. Embedding similarity --
    embedder = _get_embedder(model_name)
    if embedder:
        try:
            all_topics = get_all_topics(conn)
            if all_topics:
                import numpy as np

                query_emb = embedder.encode(text, normalize_embeddings=True)
                best_score = 0.0
                best_id = None
                for topic in all_topics:
                    tid = topic["topic_id"]
                    if tid not in _embeddings_cache:
                        _embeddings_cache[tid] = embedder.encode(
                            topic["canonical_name"], normalize_embeddings=True
                        )
                    score = float(np.dot(query_emb, _embeddings_cache[tid]))
                    if score > best_score:
                        best_score = score
                        best_id = tid

                if best_id and best_score >= embedding_threshold:
                    add_alias(conn, best_id, text, source=source)
                    if wikipedia_title:
                        add_alias(conn, best_id, wikipedia_title, source="wikipedia")
                    touch_topic(conn, best_id)
                    return best_id
        except Exception as exc:
            log.debug("Embedding match error: %s", exc)

    # -- 4. Create new topic --
    canonical = wikipedia_title or text
    new_id = make_topic_id(canonical)
    # Ensure uniqueness (slug collision)
    existing = conn.execute(
        "SELECT topic_id FROM topics WHERE topic_id = ?", (new_id,)
    ).fetchone()
    if existing and existing["topic_id"] != new_id:
        suffix = hashlib.md5(canonical.encode()).hexdigest()[:4]
        new_id = f"{new_id}-{suffix}"

    upsert_topic(conn, new_id, canonical, wikipedia_title=wikipedia_title)
    add_alias(conn, new_id, canonical, source="canonical")
    if text != canonical:
        add_alias(conn, new_id, text, source=source)
    if wikipedia_title and wikipedia_title != text and wikipedia_title != canonical:
        add_alias(conn, new_id, wikipedia_title, source="wikipedia")

    # Invalidate embedding cache for new topic
    _embeddings_cache.pop(new_id, None)

    log.debug("New topic created: %s -> %r", new_id, canonical)
    return new_id


def invalidate_embedding(topic_id: str) -> None:
    _embeddings_cache.pop(topic_id, None)

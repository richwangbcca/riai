"""Tests for topic matching and slug generation."""
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import storage
import match


def _fresh_conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    storage._apply_schema(conn)
    return conn


def test_slugify():
    assert match.slugify("Hello World") == "hello-world"
    assert match.slugify("GPT-4o") == "gpt-4o"
    assert match.slugify("Björk") == "björk"  # unicode kept, Python \w matches it


def test_make_topic_id_stable():
    a = match.make_topic_id("Mount Everest")
    b = match.make_topic_id("Mount Everest")
    assert a == b


def test_match_or_create_new_topic():
    conn = _fresh_conn()
    cfg = {"fuzzy_threshold": 85, "embedding_threshold": 0.82}
    tid = match.match_or_create(conn, "Climate Change", None, cfg, source="test")
    conn.commit()
    assert tid is not None
    topic = storage.get_topic(conn, tid)
    assert topic is not None
    assert topic["canonical_name"] == "Climate Change"


def test_match_or_create_exact_alias():
    conn = _fresh_conn()
    cfg = {"fuzzy_threshold": 85, "embedding_threshold": 0.82}
    # Create once
    tid1 = match.match_or_create(conn, "World Cup", None, cfg, source="test")
    conn.commit()
    # Match again via exact alias
    tid2 = match.match_or_create(conn, "World Cup", None, cfg, source="test")
    conn.commit()
    assert tid1 == tid2


def test_match_or_create_wikipedia_title_anchor():
    conn = _fresh_conn()
    cfg = {"fuzzy_threshold": 85, "embedding_threshold": 0.82}
    tid = match.match_or_create(
        conn, "Taylor Swift", "Taylor Swift", cfg, source="wikipedia"
    )
    conn.commit()
    topic = storage.get_topic(conn, tid)
    assert topic["wikipedia_title"] == "Taylor Swift"


def test_match_or_create_fuzzy(monkeypatch):
    """With RapidFuzz available, near-duplicate names should merge."""
    try:
        import rapidfuzz  # noqa: F401
    except ImportError:
        return  # skip if not installed

    conn = _fresh_conn()
    cfg = {"fuzzy_threshold": 85, "embedding_threshold": 0.82}

    tid1 = match.match_or_create(conn, "Donald Trump", None, cfg, source="test")
    conn.commit()
    # Very close variant — should fuzzy-match to the same topic
    tid2 = match.match_or_create(conn, "Donald J. Trump", None, cfg, source="test")
    conn.commit()
    assert tid1 == tid2

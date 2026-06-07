"""Tests for scoring and normalization (deterministic given fixed inputs)."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import score


def test_logistic_midpoint():
    v = score._logistic(2.0, k=1.0, midpoint=2.0)
    assert abs(v - 0.5) < 1e-9


def test_logistic_high_z():
    v = score._logistic(10.0, k=1.0, midpoint=2.0)
    assert v > 0.99


def test_logistic_low_z():
    v = score._logistic(-5.0, k=1.0, midpoint=2.0)
    assert v < 0.01


def test_logistic_no_overflow():
    # very extreme values should not raise
    v_hi = score._logistic(1000.0, k=1.0, midpoint=2.0)
    v_lo = score._logistic(-1000.0, k=1.0, midpoint=2.0)
    assert 0.0 <= v_hi <= 1.0
    assert 0.0 <= v_lo <= 1.0


def test_scoring_pipeline_smoke():
    """Smoke test: run_scoring on an in-memory DB with synthetic data."""
    import sqlite3
    import storage
    import score as scoring

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    storage._apply_schema(conn)

    # Create a topic
    storage.upsert_topic(conn, "foo-bar", "Foo Bar", wikipedia_title="Foo Bar")
    storage.add_alias(conn, "foo-bar", "Foo Bar", source="test")

    # Inject signal in current hour bucket
    bucket = storage.hour_bucket()
    storage.upsert_signal(conn, "foo-bar", bucket, "wikipedia", "edit_count", 5.0)
    conn.commit()

    weights = {"wikipedia": 0.40, "news": 0.35, "reddit": 0.20, "search": 0.05}
    scoring_cfg = {
        "baseline_days": 7,
        "zscore_threshold": 2.5,
        "ewma_alpha": 0.3,
        "logistic_k": 1.0,
        "logistic_midpoint": 2.0,
        "min_events_for_confidence": 10,
    }

    n = scoring.run_scoring(conn, weights, scoring_cfg)
    assert n == 1

    row = storage.get_latest_score(conn, "foo-bar")
    assert row is not None
    assert 0.0 <= row["attention_index"] <= 1.0
    assert row["emerging"] in (0, 1)

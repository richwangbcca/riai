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


def _dists(values_by_signal):
    return {k: sorted(v) for k, v in values_by_signal.items()}


def test_percentile_rank_orders_within_the_field():
    d = _dists({("wikipedia", "edit_count"): [1.0, 1.0, 2.0, 50.0]})
    rank = lambda v: score._percentile_rank(d, "wikipedia", "edit_count", v)
    assert rank(1.0) < rank(2.0) < rank(50.0)
    assert 0.0 <= rank(1.0) and rank(50.0) <= 1.0


def test_percentile_rank_ties_share_the_midpoint():
    """All-identical values must tie, not be ordered arbitrarily at zero."""
    d = _dists({("wikipedia", "edit_count"): [3.0] * 8})
    assert score._percentile_rank(d, "wikipedia", "edit_count", 3.0) == 0.5


def test_percentile_rank_handles_unseen_signal():
    assert score._percentile_rank({}, "reddit", "post_velocity", 9.0) == 0.0


def test_baseline_without_variance_falls_back_to_the_field():
    """A low-traffic page edited once an hour has std=0 forever. That used to
    return the constant logistic(1.0) for every such topic regardless of size,
    which is what produced multi-way ties at the top of the index."""
    import sqlite3
    import storage

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    storage._apply_schema(conn)
    bucket = storage.hour_bucket()

    # Three topics, flat identical histories (std == 0), different current values.
    for tid, current in (("small", 1.0), ("mid", 5.0), ("big", 99.0)):
        storage.upsert_topic(conn, tid, tid)
        storage.upsert_signal(conn, tid, bucket, "wikipedia", "edit_count", current)
    conn.commit()

    dists = score._load_bucket_distributions(conn, bucket)
    cfg = {"baseline_days": 7, "logistic_k": 1.0, "logistic_midpoint": 2.0}
    norm = lambda tid, v: score._zscore_normalize(
        conn, tid, "wikipedia", "edit_count", v, cfg, dists
    )

    small, mid, big = norm("small", 1.0), norm("mid", 5.0), norm("big", 99.0)
    assert small < mid < big, "no-baseline topics must be ordered, not tied"
    assert big <= score._FALLBACK_CEILING, "must not outrank a confirmed anomaly"

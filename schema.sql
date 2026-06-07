-- RIAI SQLite schema (v1)
-- Single embedded datastore. Apply with:  sqlite3 riai.db < schema.sql
-- Timestamps are ISO-8601 UTC strings (sortable, portable). Raw payloads are JSON TEXT.
--
-- Data flow this schema supports:
--   poller -> events (raw, ephemeral ~30d)
--   match  -> events.topic_id set; topics / topic_aliases populated
--   score  -> signals (durable hourly time-series) + scores (per run)
-- Old raw events are purged after retention; signals is the long-term history,
-- so the rolling z-score baseline (7d, matched by hour-of-day) reads from signals.

PRAGMA journal_mode = WAL;        -- let the dashboard read while the poller writes
PRAGMA foreign_keys = ON;
PRAGMA synchronous = NORMAL;

-- ---------------------------------------------------------------------------
-- Pipeline state / small key-value store (last poll times, offsets, version)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS meta (
    key        TEXT PRIMARY KEY,
    value      TEXT,
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);
INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', '1');

-- ---------------------------------------------------------------------------
-- Canonical topic registry
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS topics (
    topic_id        TEXT PRIMARY KEY,            -- slug, e.g. 'gpt-x'
    canonical_name  TEXT NOT NULL,               -- display name
    wikipedia_title TEXT,                         -- nullable; anchor when available
    first_seen      TEXT NOT NULL,
    last_seen       TEXT NOT NULL,
    low_confidence  INTEGER NOT NULL DEFAULT 1,   -- 1 until enough baseline history
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);
CREATE INDEX IF NOT EXISTS idx_topics_last_seen ON topics(last_seen);
CREATE INDEX IF NOT EXISTS idx_topics_wp_title  ON topics(wikipedia_title);

-- Alias table (incl. Wikipedia redirects); used by the matcher
CREATE TABLE IF NOT EXISTS topic_aliases (
    topic_id TEXT NOT NULL REFERENCES topics(topic_id) ON DELETE CASCADE,
    alias    TEXT NOT NULL,
    source   TEXT,                                -- where the alias was seen
    PRIMARY KEY (topic_id, alias)
);
CREATE INDEX IF NOT EXISTS idx_aliases_alias ON topic_aliases(alias);

-- ---------------------------------------------------------------------------
-- Raw events (ephemeral, ~30 day retention, then rolled into signals)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    source       TEXT NOT NULL,                   -- 'wikipedia' | 'news' | 'reddit' | 'trends'
    external_id  TEXT,                            -- revision id / reddit id / gdelt id (dedup)
    ts           TEXT NOT NULL,                   -- event time (UTC)
    ingested_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    title        TEXT,
    url          TEXT,
    topic_id     TEXT REFERENCES topics(topic_id) ON DELETE SET NULL,  -- null until matched
    raw          TEXT,                            -- original payload as JSON
    UNIQUE (source, external_id)                  -- idempotent ingest
);
CREATE INDEX IF NOT EXISTS idx_events_ts          ON events(ts);
CREATE INDEX IF NOT EXISTS idx_events_source_ts   ON events(source, ts);
CREATE INDEX IF NOT EXISTS idx_events_topic_ts    ON events(topic_id, ts);
CREATE INDEX IF NOT EXISTS idx_events_unmatched   ON events(topic_id) WHERE topic_id IS NULL;

-- ---------------------------------------------------------------------------
-- Durable per-signal time series (hourly buckets) -- the long-term history
-- One row per (topic, hour, source, signal). Feeds normalization & baselines.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS signals (
    topic_id  TEXT NOT NULL REFERENCES topics(topic_id) ON DELETE CASCADE,
    bucket_ts TEXT NOT NULL,                      -- hour bucket, UTC (e.g. 2026-06-07T18:00:00Z)
    source    TEXT NOT NULL,                      -- 'wikipedia' | 'news' | 'reddit' | 'search'
    signal    TEXT NOT NULL,                      -- e.g. 'edit_count','unique_editors','pageviews',
                                                  --      'article_count','publisher_count',
                                                  --      'post_velocity','comment_velocity','score_growth'
    value     REAL NOT NULL,
    PRIMARY KEY (topic_id, bucket_ts, source, signal)
);
CREATE INDEX IF NOT EXISTS idx_signals_lookup ON signals(topic_id, source, signal, bucket_ts);
CREATE INDEX IF NOT EXISTS idx_signals_bucket ON signals(bucket_ts);

-- ---------------------------------------------------------------------------
-- Computed scores (one row per topic per scoring run)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS scores (
    topic_id        TEXT NOT NULL REFERENCES topics(topic_id) ON DELETE CASCADE,
    ts              TEXT NOT NULL,                -- scoring run time (UTC)
    wikipedia_score REAL NOT NULL DEFAULT 0,      -- each 0..1
    news_score      REAL NOT NULL DEFAULT 0,
    reddit_score    REAL NOT NULL DEFAULT 0,
    search_score    REAL NOT NULL DEFAULT 0,
    attention_index REAL NOT NULL DEFAULT 0,      -- weighted composite
    ewma            REAL,                          -- adaptive baseline
    momentum        REAL,                          -- attention_index - ewma (velocity)
    anomaly_z       REAL,                          -- z-score vs own baseline
    emerging        INTEGER NOT NULL DEFAULT 0,    -- 0/1 flag
    PRIMARY KEY (topic_id, ts)
);
CREATE INDEX IF NOT EXISTS idx_scores_ts        ON scores(ts);
CREATE INDEX IF NOT EXISTS idx_scores_emerging  ON scores(emerging, ts);
CREATE INDEX IF NOT EXISTS idx_scores_momentum  ON scores(ts, momentum);
CREATE INDEX IF NOT EXISTS idx_scores_attention ON scores(ts, attention_index);

-- ---------------------------------------------------------------------------
-- Convenience views for the dashboard / API (latest run only)
-- ---------------------------------------------------------------------------
CREATE VIEW IF NOT EXISTS latest_run AS
    SELECT MAX(ts) AS ts FROM scores;

CREATE VIEW IF NOT EXISTS top_topics AS
    SELECT s.*, t.canonical_name, t.wikipedia_title, t.low_confidence
    FROM scores s
    JOIN topics t USING (topic_id)
    WHERE s.ts = (SELECT ts FROM latest_run)
    ORDER BY s.attention_index DESC;

CREATE VIEW IF NOT EXISTS emerging_topics AS
    SELECT s.*, t.canonical_name, t.wikipedia_title, t.low_confidence
    FROM scores s
    JOIN topics t USING (topic_id)
    WHERE s.ts = (SELECT ts FROM latest_run) AND s.emerging = 1
    ORDER BY s.momentum DESC;

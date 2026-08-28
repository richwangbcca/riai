"""
Regenerate the static HTML dashboard from current DB state.
Run standalone or called from the poller every N minutes.

Usage: python -m dashboard.build_static --db riai.db --out dashboard/index.html
"""
from __future__ import annotations

import argparse
import html
import json
import logging
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import storage

log = logging.getLogger(__name__)


def _sparkline_svg(values: list[float], width: int = 80, height: int = 20) -> str:
    """Minimal inline SVG sparkline from a list of floats."""
    if not values or max(values) == min(values):
        flat_y = height // 2
        return (
            f'<svg width="{width}" height="{height}" class="spark">'
            f'<line x1="0" y1="{flat_y}" x2="{width}" y2="{flat_y}" '
            f'stroke="#555" stroke-width="1"/></svg>'
        )
    mn, mx = min(values), max(values)
    span = mx - mn or 1
    step = width / (len(values) - 1) if len(values) > 1 else width
    pts = []
    for i, v in enumerate(values):
        x = i * step
        y = height - ((v - mn) / span) * (height - 2) - 1
        pts.append(f"{x:.1f},{y:.1f}")
    poly = " ".join(pts)
    return (
        f'<svg width="{width}" height="{height}" class="spark">'
        f'<polyline points="{poly}" fill="none" stroke="#4af" stroke-width="1.5"/>'
        f"</svg>"
    )


def _momentum_badge(momentum: float | None) -> str:
    if momentum is None:
        return ""
    if momentum > 0.05:
        return '<span class="badge up">▲</span>'
    if momentum < -0.05:
        return '<span class="badge dn">▼</span>'
    return '<span class="badge flat">—</span>'


def _format_score(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v:.3f}"


def build(
    conn: sqlite3.Connection,
    out_path: str = "dashboard/index.html",
    top_n: int = 50,
    emerging_n: int = 20,
    sparkline_hours: int = 24,
) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    top = storage.get_top_topics(conn, limit=top_n)
    emerging = storage.get_emerging_topics(conn, limit=emerging_n)

    # Build sparkline data per topic (top + emerging, deduplicated)
    all_tids = list({r["topic_id"] for r in list(top) + list(emerging)})
    spark_data: dict[str, list[float]] = {}
    for tid in all_tids:
        history = storage.get_score_history(conn, tid, hours=sparkline_hours)
        spark_data[tid] = [r["attention_index"] for r in history]

    # Load summaries
    summary_rows = conn.execute("SELECT topic_id, summary FROM topic_summaries").fetchall()
    summaries: dict[str, str] = {r["topic_id"]: r["summary"] for r in summary_rows}

    # --- HTML ---
    def esc(s: str | None) -> str:
        return html.escape(s or "")

    rows_top = []
    for i, r in enumerate(top, 1):
        tid = r["topic_id"]
        spark = _sparkline_svg(spark_data.get(tid, []))
        wiki_url = (
            f"https://en.wikipedia.org/wiki/{r['wikipedia_title'].replace(' ', '_')}"
            if r["wikipedia_title"] else ""
        )
        name_cell = (
            f'<a href="{esc(wiki_url)}" target="_blank" rel="noopener">{esc(r["canonical_name"])}</a>'
            if wiki_url else esc(r["canonical_name"])
        )
        lc = ' <sup title="Low confidence: not enough history">~</sup>' if r["low_confidence"] else ""
        summary = summaries.get(tid, "")
        summary_cell = f'<br><span class="summary">{esc(summary)}</span>' if summary else ""
        rows_top.append(
            f"<tr>"
            f"<td>{i}</td>"
            f"<td>{name_cell}{lc}{summary_cell}</td>"
            f"<td>{_format_score(r['attention_index'])}</td>"
            f"<td>{_momentum_badge(r['momentum'])}&nbsp;{_format_score(r['momentum'])}</td>"
            f"<td>{spark}</td>"
            f"</tr>"
        )

    rows_em = []
    for r in emerging:
        tid = r["topic_id"]
        spark = _sparkline_svg(spark_data.get(tid, []))
        wiki_url = (
            f"https://en.wikipedia.org/wiki/{r['wikipedia_title'].replace(' ', '_')}"
            if r["wikipedia_title"] else ""
        )
        name_cell = (
            f'<a href="{esc(wiki_url)}" target="_blank" rel="noopener">{esc(r["canonical_name"])}</a>'
            if wiki_url else esc(r["canonical_name"])
        )
        lc = ' <sup title="Low confidence">~</sup>' if r["low_confidence"] else ""
        summary = summaries.get(tid, "")
        summary_cell = f'<br><span class="summary">{esc(summary)}</span>' if summary else ""
        rows_em.append(
            f"<tr>"
            f"<td>{name_cell}{lc}{summary_cell}</td>"
            f"<td>{_format_score(r['attention_index'])}</td>"
            f"<td>{_format_score(r['momentum'])}</td>"
            f"<td>{_format_score(r['anomaly_z'])}</td>"
            f"<td>{spark}</td>"
            f"</tr>"
        )

    top_html = "\n".join(rows_top) or "<tr><td colspan='5'>No data yet</td></tr>"
    em_html = "\n".join(rows_em) or "<tr><td colspan='5'>No emerging topics detected</td></tr>"

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="300">
<title>RIAI — Real-Time Internet Attention Index</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:system-ui,sans-serif;background:#0d1117;color:#c9d1d9;font-size:14px}}
header{{padding:16px 24px;border-bottom:1px solid #30363d;display:flex;justify-content:space-between;align-items:center}}
header h1{{font-size:18px;font-weight:600;letter-spacing:.5px}}
.ts{{color:#8b949e;font-size:12px}}
main{{padding:24px;max-width:1100px;margin:0 auto}}
section{{margin-bottom:40px}}
h2{{font-size:15px;font-weight:600;margin-bottom:12px;color:#58a6ff}}
table{{width:100%;border-collapse:collapse}}
th{{text-align:left;padding:6px 10px;border-bottom:1px solid #30363d;color:#8b949e;font-weight:500;font-size:12px;text-transform:uppercase;letter-spacing:.5px}}
td{{padding:6px 10px;border-bottom:1px solid #21262d;vertical-align:middle}}
tr:hover td{{background:#161b22}}
a{{color:#58a6ff;text-decoration:none}}
a:hover{{text-decoration:underline}}
.badge{{font-size:11px;border-radius:3px;padding:1px 4px}}
.badge.up{{color:#3fb950}}
.badge.dn{{color:#f85149}}
.badge.flat{{color:#8b949e}}
.spark{{vertical-align:middle;display:inline-block}}
.notice{{color:#8b949e;font-size:12px;margin-top:8px;font-style:italic}}
sup{{font-size:10px;color:#8b949e}}
.summary{{font-size:12px;color:#8b949e;font-style:italic;display:-webkit-box;-webkit-box-orient:vertical;-webkit-line-clamp:2;line-clamp:2;overflow:hidden}}
.explainer{{font-size:13px;color:#8b949e;line-height:1.65;max-width:760px}}
.explainer h3{{font-size:13px;color:#c9d1d9;font-weight:600;margin:18px 0 4px}}
.explainer h3:first-of-type{{margin-top:0}}
.explainer p{{margin-bottom:6px}}
.explainer .cav{{color:#6e7681}}
footer{{padding:16px 24px;border-top:1px solid #30363d;color:#8b949e;font-size:11px;margin-top:40px}}
</style>
</head>
<body>
<header>
  <h1>RIAI — Real-Time Internet Attention Index</h1>
  <span class="ts">Updated {now} &nbsp;·&nbsp; auto-refresh 5 min</span>
</header>
<main>
  <p class="notice">
    Constructed heuristic index, not objective measurement.
    Sources skew English-language. No manipulation resistance.
    Numbers are relative, not absolute.
  </p>

  <section style="margin-top:24px">
    <h2>Emerging Topics</h2>
    <p class="notice">Topics rising sharply above their own baseline right now.</p>
    <table>
      <thead><tr>
        <th>Topic</th><th>Attention</th><th>Momentum</th><th>Anomaly Z</th><th>24h</th>
      </tr></thead>
      <tbody>
        {em_html}
      </tbody>
    </table>
  </section>

  <section>
    <h2>Top Topics by Attention</h2>
    <table>
      <thead><tr>
        <th>#</th><th>Topic</th><th>Attention</th><th>Momentum</th><th>24h</th>
      </tr></thead>
      <tbody>
        {top_html}
      </tbody>
    </table>
  </section>

  <section class="explainer">
    <h2>How to read this page</h2>

    <h3>Attention</h3>
    <p>
      A 0–1 score for how <em>unusual</em> a topic's activity is right now — not how big it is.
      Each hour, a topic's edit counts, article counts and post counts are compared against what
      that same topic normally does at that same hour of day over the past week. Matching its own
      normal scores near zero; running about two standard deviations above it scores around 0.5,
      and further above that climbs toward 1.
    </p>
    <p>
      So a small topic doing ten times its usual traffic will outrank a famous one having an
      ordinary day. A high score means "unusually busy for itself", never "most popular".
    </p>
    <p class="cav">
      The four sources are weighted Wikipedia 40%, news 35%, Reddit 20%, search 5% (defaults; set in
      <code>config.yaml</code>). Search is not implemented and contributes nothing, so in practice
      scores top out near 0.95. A topic with no baseline to compare against yet gets a fixed low
      placeholder for that signal rather than a real score, and a topic seen in only a handful of
      events is separately marked ~.
    </p>

    <h3>Momentum</h3>
    <p>
      Whether a topic is still climbing or already past its peak. It is the current attention score
      minus that topic's own recent smoothed average, so it asks "is right now busier than the last
      few hours were?" rather than comparing against other topics.
    </p>
    <p>
      <span class="badge up">▲</span> rising · <span class="badge dn">▼</span> fading ·
      <span class="badge flat">—</span> holding steady. A topic can have high attention and negative
      momentum at the same time: still busy, but quieter than it just was. Brand-new topics sit near
      zero until they have enough history to be compared against.
    </p>

    <h3>The {sparkline_hours}h graph</h3>
    <p>
      The topic's attention score at every scoring run over the last {sparkline_hours} hours, oldest
      on the left. It is there to show the <em>shape</em> of a run — one clean spike, a slow build,
      or a jagged on-and-off pattern.
    </p>
    <p class="cav">
      Each line is scaled to its own highest and lowest point, so heights are not comparable between
      rows: a dramatic-looking peak in one row may be a far smaller move than a gentle rise in
      another. A flat line means the score did not change, or there is not yet enough history.
    </p>

    <h3>Emerging &amp; Anomaly Z</h3>
    <p>
      A topic is flagged emerging when its current attention is at least 2.5 standard deviations
      above its own attention at this hour on previous days — that multiple is the Anomaly Z column.
      It needs at least six prior samples to qualify, which is why genuinely new events often take
      several hours to appear here.
    </p>
  </section>
</main>
<footer>
  RIAI — data from Wikipedia EventStreams, Pageviews API, GDELT, RSS feeds, Reddit.
  &nbsp;~&nbsp; = low confidence (insufficient history).
  Scores are z-score-normalized per topic, composited, and squashed to [0,1].
  This is a heuristic — validate empirically before trusting it.
</footer>
</body>
</html>"""

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(page, encoding="utf-8")
    log.info("Dashboard written to %s", out_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Build RIAI static dashboard")
    parser.add_argument("--db", default="riai.db")
    parser.add_argument("--out", default="dashboard/index.html")
    parser.add_argument("--top", type=int, default=50)
    parser.add_argument("--emerging", type=int, default=20)
    parser.add_argument("--hours", type=int, default=24)
    args = parser.parse_args()

    conn = storage.open_db(args.db)
    build(conn, out_path=args.out, top_n=args.top, emerging_n=args.emerging, sparkline_hours=args.hours)


if __name__ == "__main__":
    main()

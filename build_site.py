"""Build the website version of the Workbench (for GitHub Pages) into the site/ folder.

GitHub Actions runs this about every 10 minutes (.github/workflows/update-site.yml), so the website stays live
without your computer. Each build also writes version.json; open pages check it once a minute and load the new
data.js by themselves. You can also run it yourself and open site/index.html in a browser:

    .\\.venv\\Scripts\\python.exe build_site.py --db workbench.db             # snapshot of your local data
    .\\.venv\\Scripts\\python.exe build_site.py --db workbench.db --refresh   # fetch fresh data first (a few minutes)

Safety: the database you pass is only read (never changed, unless you name the same file with --save-db). The script
works on a temporary copy and deletes the copy's settings first, so your Gemini key can never end up on the website.
Gemini is used only when the environment variable WORKBENCH_GEMINI_KEY is set (in GitHub Actions: the optional
GEMINI_API_KEY repository secret), and the key is removed again before anything is written.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLACEHOLDER = "<!-- STATIC_DATA -->"
KEEP_SETTINGS = ("news_sync_at", "news_visit_last", "news_visit_prev", "new_per_day", "score_version",
                 "gh_releases_last", "arxiv_max_results", "hf_limit", "custom_feeds",
                 "site_gemini_last", "site_gemini_report", "x_provider", "x_accounts", "x_daily_cap")
KEEP_PREFIXES = ("feed_baseline:", "xstate_")   # undated-feed memory; X read positions and today's read count


def copy_database(src: Path, dst: Path) -> None:
    s = sqlite3.connect(f"file:{src}?mode=ro", uri=True)   # read-only, safe while the local server is running
    d = sqlite3.connect(str(dst))
    s.backup(d)
    s.close()
    prefixes = " AND ".join("substr(key, 1, ?) != ?" for _ in KEEP_PREFIXES)
    d.execute(f"DELETE FROM settings WHERE key NOT IN ({','.join('?' for _ in KEEP_SETTINGS)}) AND {prefixes}",
              (*KEEP_SETTINGS, *[v for p in KEEP_PREFIXES for v in (len(p), p)]))
    d.commit()
    d.close()


def wait_while(lock, timeout: float) -> None:
    t0 = time.time()
    time.sleep(1)
    while lock.locked() and time.time() - t0 < timeout:
        time.sleep(2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=str(HERE / "workbench.db"), help="database to start from (may not exist yet)")
    ap.add_argument("--save-db", help="save the updated database here afterwards (GitHub Actions caches it)")
    ap.add_argument("--site", default=str(HERE / "site"), help="output folder (default: site)")
    ap.add_argument("--refresh", action="store_true", help="fetch all sources before building")
    ap.add_argument("--figures", type=int, default=6, help="with --refresh: load arXiv figures for this many top papers (15 s each)")
    ap.add_argument("--analyze", type=int, default=2, help="with a Gemini key: write breakdowns for this many top papers")
    ap.add_argument("--gemini-every", type=float, default=0, metavar="HOURS",
                    help="with a Gemini key: use Gemini at most this often (0 = every build); frequent builds would use up the free quota")
    ap.add_argument("--max-papers", type=int, default=400)
    ap.add_argument("--alerts", default=str(HERE / "alerts" / "state.json"), help="alerts.py state file (optional)")
    args = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="wb-site-"))
    work = tmp / "work.db"
    src = Path(args.db).resolve()
    if src.exists():
        copy_database(src, work)
    else:
        print(f"No database at {src} yet: starting a fresh one.")

    key = os.environ.get("WORKBENCH_GEMINI_KEY", "").strip()
    os.environ["WORKBENCH_DB"] = str(work)
    os.environ["WORKBENCH_NO_FETCH"] = "1"
    os.environ["WORKBENCH_SITE_URL"] = ""   # this build *is* the website: never read alerts/X back from it
    # The X key (secret X_API_KEY -> WORKBENCH_X_KEY) is read from the environment only, never stored in the database.
    if not key:
        os.environ["WORKBENCH_GEMINI_BASE"] = "http://127.0.0.1:9/gemini-disabled"   # nothing can reach Gemini
    sys.path.insert(0, str(HERE))
    import server as S   # noqa: E402  (after the environment is set)

    S.init_db()
    S.seed_starter()
    S.backfill_scores()
    # Highest alerts (new Claude/Gemini/GPT models, breakthroughs) also go on the News page
    raw_alerts = load_alerts(Path(args.alerts))
    n_alerts = S.ingest_alerts(raw_alerts)
    if n_alerts:
        print(f"Alerts: {n_alerts} alert(s) added to the News page.")
    if not args.refresh:
        S.rescore_news()
    gemini_report: dict = {"enabled": bool(key), "note": "no Gemini step in this build (republish only)" if key else "no GEMINI_API_KEY secret"}
    # Builds run every few minutes; Gemini (free quota) only joins every --gemini-every hours. In between, the
    # key is not even loaded and the website keeps showing the report of the last Gemini build.
    last_gemini = S.parse_iso(S.get_setting("site_gemini_last"))
    gemini_due = bool(key) and args.refresh and (
        last_gemini is None or (S.now_utc() - last_gemini).total_seconds() >= args.gemini_every * 3600)
    try:
        if gemini_due:
            S.set_setting("gemini_api_key", key)
        if args.refresh:
            print("Fetching fresh data: news feeds, Hugging Face, arXiv, Hacker News, GitHub...")
            # force=False: GitHub releases keep their own 30-minute minimum (server.RELEASES_MIN_AGE)
            S.start_refresh("site", None, force=False)
            wait_while(S._refresh_lock, 1800)
            fetch_figures(S, args.figures)
        if gemini_due:
            gemini_report = run_gemini(S, args.analyze, key)
            S.set_setting("site_gemini_last", S.iso())
            S.set_setting("site_gemini_report", json.dumps(gemini_report, ensure_ascii=False))
        elif key and args.refresh:
            try:
                gemini_report = json.loads(S.get_setting("site_gemini_report") or "")
            except ValueError:
                gemini_report = {"enabled": True, "note": "Gemini joins a build every few hours; none since this database started"}
            print(f"Gemini: skipped (last used {S.get_setting('site_gemini_last')}, at most every {args.gemini_every:g} h).")
    finally:
        S.set_setting("gemini_api_key", None)
    if S.get_setting("gemini_api_key"):
        raise SystemExit("Refusing to continue: the working database still holds a Gemini key.")

    data = build_snapshot(S, args.max_papers)
    data["alerts"] = raw_alerts
    data["gemini"] = gemini_report
    write_site(Path(args.site), data)
    if args.save_db:
        with S.db() as c:
            c.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        dst = Path(args.save_db)
        dst.parent.mkdir(parents=True, exist_ok=True)
        copy_database(work, dst)
        print(f"Saved the database to {dst}.")
    shutil.rmtree(tmp, ignore_errors=True)
    return 0


def fetch_figures(S, n: int) -> None:
    if n <= 0:
        return
    latest = "COALESCE(latest_at, published, first_seen)"
    with S.db() as c:
        todo = [r[0] for r in c.execute(
            f"""SELECT arxiv_id FROM papers WHERE figures_checked IS NULL
                ORDER BY COALESCE(importance, 0) / (1.0 + MAX(0, julianday('now') - julianday({latest})) / 2.0) DESC
                LIMIT ?""", (n,))]
    for i, aid in enumerate(todo):
        try:
            S.fetch_figures(aid)
            print(f"Figures: arXiv:{aid} ({i + 1}/{len(todo)})")
        except Exception as exc:
            print(f"Figures: arXiv:{aid} failed ({exc})")


class _GeminiLog(logging.Handler):
    """Collects the app's Gemini warnings during a build, so the website can show whether the key works."""

    def __init__(self, key: str):
        super().__init__(logging.WARNING)
        self.key, self.lines = key, []

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        if "gemini" in msg.lower() and len(self.lines) < 12:
            self.lines.append(msg.replace(self.key, "***")[:240] if self.key else msg[:240])


def _gemini_counts(S) -> dict:
    with S.db() as c:
        return {"triaged": c.execute("SELECT COUNT(*) FROM news WHERE brief IS NOT NULL").fetchone()[0],
                "analyses": c.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]}


def _wait_out_rate_limit(S, cap: float = 90) -> None:
    left = S._gemini_pause_until - time.time()
    if left > 0:
        print(f"Gemini: rate limited, waiting {min(left, cap):.0f} s before the next step.")
        time.sleep(min(left, cap))
        S._gemini_pause_until = 0.0


def run_gemini(S, n_analyze: int, key: str = "") -> dict:
    """With a key: triage the top events, write the Frontier Brief and break down a few top papers.
    Returns a small report (counts and Gemini error messages, never the key) that goes on the website."""
    handler = _GeminiLog(key)
    S.log.addHandler(handler)
    before = _gemini_counts(S)
    try:
        _run_gemini_steps(S, n_analyze)
    finally:
        S.log.removeHandler(handler)
    after = _gemini_counts(S)
    with S.db() as c:
        queue = [{"paper": r["arxiv_id"], "status": r["status"], "kind": r["kind"], "message": (r["message"] or "")[:200]}
                 for r in c.execute("SELECT arxiv_id, status, kind, message FROM analysis_queue WHERE status != 'running'")]
    brief = S.frontier_brief_get()
    report = {"enabled": True, "checked_at": S.iso(),
              "triaged_this_build": after["triaged"] - before["triaged"],
              "breakdowns_this_build": after["analyses"] - before["analyses"],
              "brief": {"status": brief.get("status"), "model": brief.get("model"), "generated_at": brief.get("generated_at")},
              "problems": handler.lines, "queue": queue[:10]}
    print("Gemini report:", json.dumps(report, ensure_ascii=False))
    return report


def _run_gemini_steps(S, n_analyze: int) -> None:
    _wait_out_rate_limit(S)   # the refresh may already have written a brief and hit the per-minute limit
    try:
        S.process_briefs(5)
    except Exception as exc:
        print("Gemini triage skipped:", exc)
    _wait_out_rate_limit(S)
    if S.start_brief_regen(force=True) or S._brief_lock.locked():
        wait_while(S._brief_lock, 300)
    _wait_out_rate_limit(S)
    if n_analyze > 0:
        latest = "COALESCE(p.latest_at, p.published, p.first_seen)"
        with S.db() as c:
            top = [r[0] for r in c.execute(
                f"""SELECT p.arxiv_id FROM papers p WHERE length(p.abstract) > 200
                    AND NOT EXISTS (SELECT 1 FROM analyses a WHERE a.arxiv_id = p.arxiv_id)
                    AND NOT EXISTS (SELECT 1 FROM analysis_queue q WHERE q.arxiv_id = p.arxiv_id AND q.status = 'error')
                    ORDER BY COALESCE(p.importance, 0) / (1.0 + MAX(0, julianday('now') - julianday({latest})) / 2.0) DESC
                    LIMIT ?""", (n_analyze,))]
        for aid in top:
            S.queue_set(aid, "queued", None, "Queued for Gemini (website build).")
        try:
            S.process_queue()
        except Exception as exc:
            print("Gemini breakdowns skipped:", exc)


def build_snapshot(S, max_papers: int) -> dict:
    with S.db() as c:
        rows = c.execute("SELECT * FROM news ORDER BY COALESCE(trend_score, 0) DESC, COALESCE(published, first_seen) DESC").fetchall()
        raw = {r["id"]: dict(r) for r in rows}
        news_total = len(rows)
    groups = S.group_stories([S._news_out(r) for r in rows])

    quickstart = {}
    for g in groups:
        r = raw[g["id"]]
        if g["kind"] in ("model", "repo") and r.get("readme"):
            g["readme_excerpt"] = r["readme"][:1500]    # for in-browser Gemini triage
        specs = json.loads(r.get("specs") or "{}")
        if specs.get("commands"):
            g["commands"] = specs["commands"][:4]
        if len(quickstart) < 300 and (g["kind"] in ("model", "repo") or (g.get("metrics") or {}).get("runnable")):
            try:
                quickstart[g["id"]] = S.quickstart_for(r)
            except Exception as exc:
                print("Quickstart skipped for", g["id"][:60], exc)

    brief = S.frontier_brief_get()
    brief = {k: brief.get(k) for k in ("status", "brief", "items", "model", "generated_at", "sync_id")}

    picks: dict[str, None] = {}
    latest = "COALESCE(latest_at, published, first_seen)"
    with S.db() as c:
        for q in (
            "SELECT arxiv_id FROM papers WHERE starter = 1",
            "SELECT arxiv_id FROM analyses",
            "SELECT arxiv_id FROM papers WHERE sources LIKE '%\"news\"%'",
            f"SELECT arxiv_id FROM papers ORDER BY COALESCE(importance, 0) / (1.0 + MAX(0, julianday('now') - julianday({latest})) / 2.0) DESC LIMIT 250",
            f"SELECT arxiv_id FROM papers ORDER BY {latest} DESC LIMIT 200",
            "SELECT arxiv_id FROM papers ORDER BY COALESCE(importance, 0) DESC LIMIT 120",
            "SELECT arxiv_id FROM papers WHERE hf_upvotes IS NOT NULL ORDER BY hf_upvotes DESC LIMIT 120",
        ):
            for r in c.execute(q):
                picks.setdefault(r[0])
        for aid in {a for g in groups for a in g.get("arxiv_ids") or []}:
            if c.execute("SELECT 1 FROM papers WHERE arxiv_id=?", (aid,)).fetchone():
                picks.setdefault(aid)
    papers = []
    for aid in list(picks)[:max_papers]:
        p = S.paper_detail(aid)
        p["analyzed"] = bool(p.get("analysis"))
        p["analysis_origin"] = (p.get("analysis") or {}).get("origin")
        p["queue_status"] = None
        p.pop("queue", None)
        p.pop("questions", None)
        papers.append(p)

    with S.db() as c:
        qrows = c.execute(f"SELECT {S.Q_COLS} FROM {S.Q_FROM} ORDER BY q.id").fetchall()
    questions = []
    for r in qrows:
        q = S.question_out(r)
        q.pop("preview", None)
        questions.append(q)

    sources = [{k: s.get(k) for k in ("name", "label", "status", "message", "items", "last_attempt", "last_success")}
               for s in S.status()["sources"]]
    return {
        "version": 1,
        "generated_at": S.iso(),
        "repo": os.environ.get("GITHUB_REPOSITORY", ""),
        "news_sync_at": S.get_setting("news_sync_at"),
        "sources": sources,
        "feeds": S.news_sources(),
        "categories": S.NEWS_CATEGORIES,
        "news": groups,
        "news_total": news_total,
        "quickstart": quickstart,
        "runnable": S.news_runnable()["items"],
        "brief": brief,
        "papers": papers,
        "repos": S.repos(),
        "graph": S.graph(),
        "questions": questions,
        "settings": {"new_per_day": S.setting_int("new_per_day", 0, 500)},
        "x": x_snapshot(S),
    }


def x_snapshot(S) -> dict:
    """X posts for the website's X page (and x.json, which the desktop app reads when it has no key of its own)."""
    items, total = S.x_items(limit=400)
    info = S.x_info()   # provider, cap, accounts, read count: never the key
    with S.db() as c:
        src = c.execute("SELECT status, message, last_success FROM sources WHERE name='x'").fetchone()
    return {**info, "items": items, "total": total, "source": dict(src) if src else None}


def load_alerts(path: Path) -> list[dict]:
    """The newest highest alerts (model releases, breakthroughs) recorded by alerts.py."""
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    keep = ("kind", "vendor", "key", "name", "title", "url", "source", "published", "detected", "evidence")
    return [{k: a.get(k) for k in keep} for a in state.get("alerts", [])[:20]]


def write_site(site: Path, data: dict) -> None:
    html = (HERE / "index.html").read_text(encoding="utf-8")
    if PLACEHOLDER not in html:
        raise SystemExit(f"index.html has no {PLACEHOLDER} marker.")
    stamp = data["generated_at"].replace(":", "").replace("-", "")
    html = html.replace(PLACEHOLDER, f'<script src="data.js?v={stamp}"></script>', 1)
    if site.exists():
        shutil.rmtree(site)
    site.mkdir(parents=True)
    text = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    (site / "data.js").write_text("window.WB_DATA = " + text + ";\n", encoding="utf-8")
    # Open pages fetch this small file once a minute and load the new data.js when generated_at changes.
    (site / "version.json").write_text(json.dumps({"generated_at": data["generated_at"], "news_sync_at": data.get("news_sync_at"),
                                                   "news": len(data["news"])}), encoding="utf-8")
    # Small side files for the desktop app (server.py reads them when it has no X key / no alerts job of its own)
    (site / "alerts.json").write_text(json.dumps({"generated_at": data["generated_at"], "alerts": data.get("alerts") or []},
                                                 ensure_ascii=False), encoding="utf-8")
    (site / "x.json").write_text(json.dumps({"generated_at": data["generated_at"], **(data.get("x") or {})},
                                            ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    (site / "index.html").write_text(html, encoding="utf-8")
    (site / ".nojekyll").write_text("", encoding="utf-8")
    print(f"Website written to {site}: {len(data['news'])} news events, {len(data['papers'])} papers, "
          f"{len(data['questions'])} review questions ({len(text) / 1e6:.2f} MB of data).")


if __name__ == "__main__":
    raise SystemExit(main())

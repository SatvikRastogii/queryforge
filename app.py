"""Stage 7: FastAPI on port 7860 (Hugging Face Spaces requirement).

Three pages:
  GET  /replay  — the completed 20-generation run: chart + per-generation table,
                  read from evals/results.csv and evals/baselines.json, plus
                  buttons into /custom and /live.
  POST /live    — a short 3-generation live search on the fixed TPC-H workload,
                  so a visitor can watch the loop actually run. Takes a few
                  minutes (it runs real benchmarks); renders a measured result page.
  GET/POST /custom — the same search against user-submitted SQL, with the
                  dataset schema and example queries shown on the form.

Everything shown is measured. If an artifact is missing, the page says so
rather than inventing numbers.
"""

import contextlib
import csv
import hashlib
import html
import json
import threading
import time
from pathlib import Path
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, Form, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse

import graph
import oracle
import specs
from oracle import WORKLOAD

EVALS = Path(__file__).parent / "evals"
app = FastAPI(title="QueryForge")

# In-process cache for /custom: sha1 of the sorted queries -> the measured
# result dict. No Redis/disk — a demo tool, lost on restart. A hit returns a
# REAL past measurement (labeled cached), never a fabricated number.
_CUSTOM_CACHE: dict[str, dict] = {}
MAX_CUSTOM_QUERIES = 10

# --------------------------------------------------------------------------
# Guardrails: /live and /custom both run oracle.reset_indexes() -> build ->
# benchmark against the SAME shared Postgres schema. Two concurrent runs would
# race on that shared state and silently corrupt each other's measurements —
# the one failure mode CLAUDE.md singles out (measurement must be ground
# truth). So only one run at a time, plus a cooldown between runs so a public
# deployment can't be hammered into burning Groq's daily token quota.
# --------------------------------------------------------------------------
_BENCH_LOCK = threading.Lock()
_last_run_finished_at = 0.0
RUN_COOLDOWN_S = 30.0


class BenchmarkBusy(Exception):
    """A run is already in progress, or the cooldown since the last one
    finished hasn't elapsed yet."""


@contextlib.contextmanager
def _run_slot():
    global _last_run_finished_at
    if time.monotonic() - _last_run_finished_at < RUN_COOLDOWN_S:
        raise BenchmarkBusy(
            f"a benchmark run just finished — wait {RUN_COOLDOWN_S:.0f}s between runs"
        )
    if not _BENCH_LOCK.acquire(blocking=False):
        raise BenchmarkBusy("a benchmark run is already in progress — try again shortly")
    try:
        yield
    finally:
        _last_run_finished_at = time.monotonic()
        _BENCH_LOCK.release()


MAX_CUSTOM_BODY_BYTES = 50_000  # /custom's raw form body, checked via Content-Length


@app.middleware("http")
async def _limit_custom_body_size(request: Request, call_next):
    # ponytail: trusts the client's Content-Length header rather than counting
    # streamed bytes: a client that lies (omits it, or uses chunked transfer)
    # bypasses this. Upgrade to a byte-counting body wrapper if that's ever
    # exploited; not worth it for a demo endpoint today.
    if request.method == "POST" and request.url.path == "/custom":
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_CUSTOM_BODY_BYTES:
            return JSONResponse(
                {"error": f"request body too large (max {MAX_CUSTOM_BODY_BYTES} bytes)"},
                status_code=413,
            )
    return await call_next(request)

# Standard, documented TPC-H table semantics — schema DOCUMENTATION for users
# who don't know the benchmark, not a measurement. Row counts shown alongside
# are measured live (specs.row_counts()); these one-liners are not.
_TPCH_TABLE_DESC = {
    "region": "5 geographic regions (Africa, America, Asia, Europe, Middle East).",
    "nation": "25 nations, each belonging to one region.",
    "supplier": "Suppliers of parts; each based in one nation.",
    "customer": "Customers who place orders; each in one nation and market segment.",
    "part": "Parts that can be ordered (brand, type, size, container).",
    "partsupp": "Which supplier supplies which part — at what cost, in what qty.",
    "orders": "Customer orders (status, total price, order date, priority).",
    "lineitem": "Line items within orders — the large fact table (one row per part "
    "per order: quantity, price, discount, ship/receipt dates).",
}


@app.get("/")
def root():
    return RedirectResponse("/replay")


@app.get("/fitness.png")
def fitness():
    p = EVALS / "fitness.png"
    if not p.exists():
        return JSONResponse({"error": "no fitness.png yet — run evals/run_search.py"}, 404)
    return FileResponse(p)


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


_PAGE_CSS = """
:root{color-scheme:light dark}
body{font:14px/1.5 system-ui,sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem}
h1{margin-bottom:.2rem}.sub{color:#888;margin-top:0}
table{border-collapse:collapse;width:100%;margin:1rem 0;font-variant-numeric:tabular-nums}
th,td{border:1px solid #8883;padding:4px 8px;text-align:right}th{text-align:right}
td:first-child,th:first-child{text-align:left}
img{max-width:100%;border:1px solid #8883;border-radius:6px}
.kpi{display:inline-block;margin-right:2rem}.kpi b{font-size:1.6rem;display:block}
code{background:#8882;padding:1px 4px;border-radius:3px}
.note{background:#8881;border-left:3px solid #888;padding:.6rem 1rem;border-radius:4px}
.btn{display:inline-block;font-size:15px;padding:.5rem 1.2rem;border-radius:6px;border:1px solid #8886;
  background:#8882;cursor:pointer;text-decoration:none;color:inherit}
.btn:hover{background:#8884}
"""


@app.get("/replay", response_class=HTMLResponse)
def replay():
    rows = _read_csv(EVALS / "results.csv")
    baselines = (
        json.loads((EVALS / "baselines.json").read_text()) if (EVALS / "baselines.json").exists() else None
    )

    if not rows:
        body = ("<div class='note'>No <code>results.csv</code> yet. Run "
                "<code>python evals/run_search.py</code> to generate the full run, "
                "then reload.</div>")
        return f"<!doctype html><meta charset=utf-8><style>{_PAGE_CSS}</style>" \
               f"<h1>QueryForge</h1><p class=sub>LLM-guided Postgres index search</p>{body}"

    # The true no-index baseline is results.csv's generation-0 row (a measured
    # history entry, not a guess). Older results.csv files predate that row —
    # fall back to baselines.json's "none", and if NEITHER exists, don't
    # compute a KPI at all rather than silently comparing against the wrong thing.
    base_row = next((r for r in rows if r["generation"] == "0"), None)
    if base_row is not None:
        baseline_ms = float(base_row["p50_total_ms"])
    elif baselines:
        baseline_ms = float(baselines["none"]["p50_total_ms"])
    else:
        baseline_ms = None

    best_ms = min(float(r["best_ms"]) for r in rows)
    improve = (1 - best_ms / baseline_ms) * 100 if baseline_ms else None

    baseline_cell = f"{baseline_ms:,.0f} ms" if baseline_ms is not None else "no baseline data"
    improve_cell = f"{improve:.1f}%" if improve is not None else "n/a"
    kpis = (
        f"<div class='kpi'>baseline<b>{baseline_cell}</b></div>"
        f"<div class='kpi'>best<b>{best_ms:,.0f} ms</b></div>"
        f"<div class='kpi'>faster<b>{improve_cell}</b></div>"
        f"<div class='kpi'>budget<b>{graph.STORAGE_BUDGET_MB:.0f} MB</b></div>"
    )

    bl_table = ""
    if baselines:
        rb = baselines["random"]["best_within_budget"]
        rb_cell = f"{rb['p50_total_ms']:,.0f} ms ({rb['storage_mb']:.1f} MB)" if rb else "none fit budget"
        bl_table = (
            "<h2>Baselines (same oracle)</h2><table>"
            "<tr><th>baseline</th><th>p50 total</th><th>storage</th></tr>"
            f"<tr><td>none</td><td>{baselines['none']['p50_total_ms']:,.0f} ms</td><td>0 MB</td></tr>"
            f"<tr><td>naive ({baselines['naive']['n_indexes']} idx)</td>"
            f"<td>{baselines['naive']['p50_total_ms']:,.0f} ms</td>"
            f"<td>{baselines['naive']['storage_mb']:.1f} MB</td></tr>"
            f"<tr><td>random best (&le;budget)</td><td colspan=2>{rb_cell}</td></tr>"
            "</table>"
        )

    head = ("generation", "p50_total_ms", "best_ms", "storage_mb", "n_indexes",
            "failed_ddl_count", "rejected_count", "regressed_queries",
            "tokens_prompt", "tokens_completion", "wall_s")
    thead = "".join(f"<th>{h}</th>" for h in head)
    trows = "".join(
        "<tr>" + "".join(f"<td>{r[h]}</td>" for h in head) + "</tr>" for r in rows
    )

    return f"""<!doctype html><meta charset=utf-8><title>QueryForge</title>
<style>{_PAGE_CSS}</style>
<h1>QueryForge</h1>
<p class=sub>LLM-guided search over Postgres index configurations, judged by a real benchmark.</p>
<div class=note>
An LLM proposes sets of <code>CREATE INDEX</code> statements for a fixed 22-query SQL workload
(TPC-H). Each proposal is <b>measured for real</b> against a live Postgres database with
<code>EXPLAIN (ANALYZE, BUFFERS)</code> — never simulated, never estimated. The LLM never grades
itself; a stopwatch does. What's below is a completed 20-generation search.
</div>
<div>{kpis}</div>
<p class=sub><b>baseline</b> = total workload latency with no indexes &middot;
<b>best</b> = total latency under the LLM's best found index set &middot;
<b>faster</b> = improvement over baseline &middot;
<b>budget</b> = the storage cap every configuration had to fit under.</p>
<img src="/fitness.png" alt="fitness curve">
{bl_table}
<h2>Per-generation results</h2>
<table><tr>{thead}</tr>{trows}</table>
<p class=note><code>regressed_queries</code> = queries slower under the current best than with no indexes. It is non-zero on purpose: every real win is a trade-off.</p>

<h2>Try it yourself</h2>
<div class=note>
<p><a class=btn href="/custom">Run your own SQL →</a> &nbsp; Paste your own read-only queries
against the TPC-H tables and watch the same search run live, with the schema/dataset shown so you
know what you can query.</p>
<p><form method="post" action="/live" style="display:inline;margin:0">
<button type="submit" class=btn>Run a live search on the TPC-H workload →</button>
</form> &nbsp; <span class=sub>takes a few minutes — runs a real benchmark and real LLM calls, not
a replay of the numbers above.</span></p>
</div>
"""


@app.get("/live")
def live_get():
    # /live only runs via the POST form on /replay -- a browser GET here
    # (e.g. someone navigating to the URL directly) lands back on the page
    # with the actual button, rather than a bare 405.
    return RedirectResponse("/replay")


@app.post("/live", response_class=HTMLResponse)
def live():
    """Run a real 3-generation search against the fixed TPC-H workload and
    render a measured result page (same rendering as /custom's)."""
    try:
        with _run_slot():
            compiled = graph.build_graph()
            initial = {
                "workload": WORKLOAD,
                "storage_budget_mb": graph.STORAGE_BUDGET_MB,
                "max_generations": 3,
            }
            # A fresh thread_id per call. build_graph() already makes a new
            # MemorySaver per request, so today the checkpointer is never
            # shared and a fixed id would still start clean — but a unique id
            # is the defensive default: it keeps state isolated even if the
            # graph/checkpointer is ever hoisted to be built once and reused.
            # (This is not fixing an observed resumption bug.)
            with graph.search_trace(mode="live", generations=3):
                final = compiled.invoke(
                    initial,
                    config={"configurable": {"thread_id": str(uuid4())}, "recursion_limit": 200},
                )
            graph.flush_traces()
    except BenchmarkBusy as e:
        return _custom_error_page([str(e)], status_code=429)

    base_entry = graph._find_config_entry(final, [])
    best_entry = graph._find_config_entry(final, final["best_config"])
    result = {
        "baseline_ms": final["baseline_ms"],
        "best_ms": final["best_ms"],
        "percent_faster": round((1 - final["best_ms"] / final["baseline_ms"]) * 100, 1),
        "best_config": final["best_config"],
        "generations_run": final["generation"],
        "base_pq": base_entry["per_query_ms"] if base_entry else {},
        "best_pq": best_entry["per_query_ms"] if best_entry else {},
        "queries_by_id": WORKLOAD,
    }
    return _custom_result_page(result, cached=False)


# --------------------------------------------------------------------------
# /custom — run the search against user-submitted queries
# --------------------------------------------------------------------------
def _page(title: str, body: str) -> str:
    return (
        f"<!doctype html><meta charset=utf-8><title>{title}</title>"
        f"<style>{_PAGE_CSS}</style>{body}"
    )


def _insights_panel() -> str:
    """Schema reference for users who don't know TPC-H: measured row counts +
    the full column list + a one-line description per table."""
    schema = specs.full_schema()
    counts = specs.row_counts()
    rows = []
    for table in sorted(schema):
        cols = ", ".join(f"{c} <span class=sub>({t})</span>" for c, t in schema[table])
        desc = _TPCH_TABLE_DESC.get(table, "")
        rows.append(
            f"<tr><td><b>{table}</b><br><span class=sub>{html.escape(desc)}</span></td>"
            f"<td>{counts.get(table, 0):,}</td><td style='text-align:left'>{cols}</td></tr>"
        )
    return (
        "<details open><summary><b>Dataset — TPC-H tables you can query</b> "
        "(row counts measured live)</summary>"
        "<table><tr><th>table</th><th>rows</th><th>columns (type)</th></tr>"
        + "".join(rows)
        + "</table></details>"
    )


# (label, SQL) — the same examples documented in README's "Example queries for /custom".
# Each is real, runnable SQL against the tables above; the last is a 3-query
# workload so the search has to find indexes that help the whole set.
_EXAMPLE_QUERIES = [
    ("Selective filter (lineitem)",
     "SELECT l_orderkey, l_quantity FROM lineitem WHERE l_shipdate = date '1994-03-15';"),
    ("Compound filter (orders)",
     "SELECT o_orderkey, o_totalprice FROM orders\n"
     "WHERE o_orderdate >= date '1995-01-01' AND o_orderstatus = 'O';"),
    ("Group-by after filter (part)",
     "SELECT p_brand, count(*) FROM part WHERE p_size = 15 GROUP BY p_brand;"),
    ("Join (orders × customer)",
     "SELECT o_orderkey, o_totalprice\n"
     "FROM orders JOIN customer ON o_custkey = c_custkey\n"
     "WHERE c_mktsegment = 'AUTOMOBILE' AND o_orderdate < date '1995-03-15';"),
    ("Multi-query workload (3 queries)",
     "SELECT l_returnflag, l_linestatus, sum(l_quantity) FROM lineitem "
     "WHERE l_shipdate <= date '1998-09-01' GROUP BY l_returnflag, l_linestatus;\n"
     "SELECT o_orderpriority, count(*) FROM orders "
     "WHERE o_orderdate >= date '1993-07-01' GROUP BY o_orderpriority;\n"
     "SELECT ps_partkey, min(ps_supplycost) FROM partsupp GROUP BY ps_partkey;"),
]


def _example_buttons() -> str:
    # json.dumps makes a safe JS string literal, but its own double quotes
    # then need HTML-attribute escaping too, since onclick="..." is itself
    # double-quoted -- without html.escape() here, a quote inside the SQL
    # closes the onclick attribute early and truncates the handler.
    buttons = "".join(
        f'<button type="button" class=btn style="font-size:12px;padding:.3rem .6rem;margin:2px 6px 2px 0" '
        f"onclick=\"document.getElementById('q').value={html.escape(json.dumps(sql))}\">"
        f"{html.escape(label)}</button>"
        for label, sql in _EXAMPLE_QUERIES
    )
    return f'<p class=sub>Try an example:</p><p>{buttons}</p>'


@app.get("/custom", response_class=HTMLResponse)
def custom_form():
    body = f"""
<h1>QueryForge — custom workload</h1>
<p class=sub>Paste your own read-only SQL against the TPC-H tables. The same loop
runs for real: an LLM proposes indexes, a live Postgres benchmark measures them,
the measurement is ground truth.</p>
<div class=note>Rules: <b>SELECT / WITH queries only</b> (no INSERT/UPDATE/DELETE/DDL),
separate multiple queries with <code>;</code>, up to <b>{MAX_CUSTOM_QUERIES}</b> queries.
Each query is checked against a read-only allowlist and validated with Postgres
<code>EXPLAIN</code> before anything runs. A run does 3 generations of real
benchmarking and takes a few minutes.</div>
{_insights_panel()}
<form method="post" action="/custom">
  <textarea id="q" name="queries" rows="14" style="width:100%;font-family:ui-monospace,monospace;font-size:13px"
    placeholder="Enter your own query, e.g. SELECT l_orderkey, l_quantity FROM lineitem WHERE l_shipdate = date '1994-03-15';"></textarea>
  {_example_buttons()}
  <p><button type="submit" class=btn>Run the search</button></p>
</form>
<p class=sub><a href="/replay">← back to the full 20-generation replay</a></p>
"""
    return _page("QueryForge — custom", body)


def _custom_error_page(errors: list[str], status_code: int = 400) -> HTMLResponse:
    items = "".join(f"<li><code>{html.escape(e)}</code></li>" for e in errors)
    body = f"""
<h1>Query rejected</h1>
<div class=note>Nothing was benchmarked — no query ran and no LLM was called.
Fix the issues below and resubmit.</div>
<ul>{items}</ul>
<p><a href="/custom">← back to the form</a></p>
"""
    return HTMLResponse(_page("QueryForge — rejected", body), status_code=status_code)


def _custom_result_page(result: dict, cached: bool) -> HTMLResponse:
    tag = ("<div class=note>Served from cache — this exact query set was measured "
           "earlier this session; these are those real numbers, not re-run.</div>"
           if cached else "")
    base = result["baseline_ms"]
    best = result["best_ms"]
    pct = result["percent_faster"]
    kpis = (
        f"<div class='kpi'>baseline<b>{base:,.0f} ms</b></div>"
        f"<div class='kpi'>best<b>{best:,.0f} ms</b></div>"
        f"<div class='kpi'>faster<b>{pct:.1f}%</b></div>"
        f"<div class='kpi'>generations<b>{result['generations_run']}</b></div>"
    )
    cfg = result["best_config"]
    cfg_html = (
        "<ul>" + "".join(f"<li><code>{html.escape(d)}</code></li>" for d in cfg) + "</ul>"
        if cfg else "<p class=sub>No within-budget index helped — the baseline "
        "(no indexes) was best for this workload.</p>"
    )
    base_pq = result.get("base_pq", {})
    best_pq = result.get("best_pq", {})
    pq_rows = ""
    for qid in base_pq:
        b = base_pq[qid]
        w = best_pq.get(qid, b)
        delta = "→" if abs(w - b) < 1e-6 else ("↓ faster" if w < b else "↑ slower")
        q_sql = html.escape(result["queries_by_id"].get(qid, ""))
        pq_rows += (
            f"<tr><td>{qid}</td><td>{b:,.0f} ms</td><td>{w:,.0f} ms</td>"
            f"<td>{delta}</td><td style='text-align:left'><code>{q_sql[:90]}</code></td></tr>"
        )
    pq_table = (
        "<h2>Per-query latency (no indexes → best config)</h2>"
        "<table><tr><th>query</th><th>baseline</th><th>best</th><th></th><th>SQL</th></tr>"
        f"{pq_rows}</table>"
    ) if pq_rows else ""
    body = f"""
<h1>Custom workload result</h1>
{tag}
<div>{kpis}</div>
<h2>Best index configuration ({len(cfg)} indexes)</h2>
{cfg_html}
{pq_table}
<p class=note>Every number here was measured by the same oracle used for the
headline benchmark: <code>EXPLAIN (ANALYZE, BUFFERS)</code> execution time, median
of 3 timed passes, budget {graph.STORAGE_BUDGET_MB:.0f} MB. The LLM never graded itself.</p>
<p><a href="/custom">← run another</a></p>
"""
    return HTMLResponse(_page("QueryForge — custom result", body))


@app.post("/custom", response_class=HTMLResponse)
def custom_run(queries: str = Form(...)):
    # Split on ';' (how people write multi-statement SQL). Not quote-aware: a
    # ';' inside a string literal would mis-split, but a bad fragment then
    # fails validate_select()/explain_check() and is reported as invalid — it
    # is never silently treated as a passing query.
    qlist = [q.strip() for q in queries.split(";") if q.strip()]
    if not qlist:
        return _custom_error_page(["No query submitted — the box was empty."])
    if len(qlist) > MAX_CUSTOM_QUERIES:
        return _custom_error_page(
            [f"Too many queries: {len(qlist)} submitted, limit is {MAX_CUSTOM_QUERIES}."]
        )

    # Two real gates, both before any LLM/graph code runs (CLAUDE.md rule 3:
    # surface failures, never a silent fallback).
    errors: list[str] = []
    for i, q in enumerate(qlist, 1):
        if not specs.validate_select(q):
            errors.append(
                f"q{i}: rejected by the read-only allowlist (must be a single "
                f"SELECT/WITH query, no writes or DDL). SQL: {q[:100]}"
            )
            continue
        err = oracle.explain_check(q)
        if err is not None:
            errors.append(f"q{i}: {err}")
    if errors:
        return _custom_error_page(errors)

    workload = {f"q{i}": q for i, q in enumerate(qlist, 1)}
    key = hashlib.sha1("\x1e".join(sorted(qlist)).encode()).hexdigest()
    cached = key in _CUSTOM_CACHE
    if cached:
        return _custom_result_page(_CUSTOM_CACHE[key], cached=True)

    try:
        with _run_slot():
            compiled = graph.build_graph()
            initial = {
                "workload": workload,
                "storage_budget_mb": graph.STORAGE_BUDGET_MB,
                "max_generations": 3,
            }
            with graph.search_trace(mode="custom", generations=3):
                final = compiled.invoke(
                    initial,
                    config={"configurable": {"thread_id": str(uuid4())}, "recursion_limit": 200},
                )
            graph.flush_traces()
    except BenchmarkBusy as e:
        return _custom_error_page([str(e)], status_code=429)

    base_entry = graph._find_config_entry(final, [])
    best_entry = graph._find_config_entry(final, final["best_config"])
    result = {
        "baseline_ms": final["baseline_ms"],
        "best_ms": final["best_ms"],
        "percent_faster": round((1 - final["best_ms"] / final["baseline_ms"]) * 100, 1),
        "best_config": final["best_config"],
        "generations_run": final["generation"],
        "base_pq": base_entry["per_query_ms"] if base_entry else {},
        "best_pq": best_entry["per_query_ms"] if best_entry else {},
        "queries_by_id": workload,
    }
    _CUSTOM_CACHE[key] = result
    return _custom_result_page(result, cached=False)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
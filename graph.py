"""Stage 4: the LangGraph search.

The loop is a closed feedback cycle around the oracle:

    baseline -> propose -> validate -> benchmark -> archive -> analyze
                   ^__________|            (reject)
                   |__________________________|   (loop)

The LLM proposes indexes; a strict gate screens them; the oracle MEASURES them;
the archive records the measurement and the frontier; and the measurement plus
every failure flows back into the next prompt. The LLM never scores itself —
the stopwatch does.

Design choices worth stating in an interview:
  - State carries an append-only `history` via an operator.add reducer, so no
    node can clobber the cross-generation memory the propose node reads from.
  - The storage budget is enforced on the oracle's MEASURED storage_mb, never
    on a guess: an over-budget config is not accepted as best, and its measured
    overage is fed back as a failure so the LLM shrinks the next proposal.
  - A malicious value in any LLM field can't escape: it becomes part of a DDL
    string that specs.validate() re-checks (allowlist + no-stacking) before it
    ever reaches Postgres, and the agent role couldn't run it even if it did.
"""

import contextlib
import logging
import operator
import os
import re
import time
from typing import Annotated, Literal, TypedDict

import psycopg
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field, ValidationError

import archive
import oracle
import specs

load_dotenv()

AGENT_DSN = os.environ.get(
    "PG_AGENT_DSN", "postgresql://queryforge_agent:agentpw@localhost:5432/queryforge"
)

MODEL_PROPOSE = "llama-3.1-8b-instant"       # high-volume path — cheap model
MODEL_ANALYZE = "llama-3.3-70b-versatile"    # one final call only — big model

MAX_GENERATIONS = 20
MAX_STAGNATION = 5
MAX_CONSECUTIVE_REJECTS = 3
TINY_TABLE_ROWS = 1_000     # below this, an index is almost never worth it
STORAGE_BUDGET_MB = 25.0    # the B in "size(I) <= B" — one source of truth


# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------
class ForgeState(TypedDict):
    # fixed inputs (workload + budget set by the caller; schema_ddl by baseline)
    schema_ddl: str
    workload: dict[str, str]
    storage_budget_mb: float
    max_generations: int  # optional; defaults to MAX_GENERATIONS if unset

    # current generation's working set
    generation: int
    candidate: list[str]
    last_result: dict
    last_failures: list[dict]

    # the search frontier
    baseline_ms: float
    best_ms: float
    best_config: list[str]
    stagnation: int
    consecutive_rejects: int

    route: str  # "ok" / "reject", written by validate, read by the gate edge

    # append-only audit log — the reducer makes every node's return APPEND
    history: Annotated[list[dict], operator.add]


# --------------------------------------------------------------------------
# Structured LLM output
# --------------------------------------------------------------------------
class ProposedIndex(BaseModel):
    table: str = Field(min_length=1)
    columns: list[str] = Field(min_length=1)
    include: list[str] = []
    # Code-enforced, not just prompt text ("Rules: btree only" in
    # _build_propose_prompt): Pydantic rejects any other value at parse time,
    # which propose_node already treats as a rejected proposal fed back to
    # the LLM as a failure — the same path a malformed JSON response takes.
    method: Literal["btree"] = "btree"

    def to_spec(self) -> specs.IndexSpec:
        return specs.IndexSpec(
            table=self.table,
            columns=tuple(self.columns),
            include=tuple(self.include),
            method=self.method,
        )


class Proposal(BaseModel):
    reasoning: str
    indexes: list[ProposedIndex]


# --------------------------------------------------------------------------
# Groq client (lazy — importing this module must not require a key)
# --------------------------------------------------------------------------
_groq_client = None

MAX_RATE_LIMIT_RETRIES = 5


def _client():
    global _groq_client
    if _groq_client is None:
        from groq import Groq

        _groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _groq_client


def _groq_call(**kwargs):
    """Groq chat call with bounded, LOUD retry on genuine 429 rate limits
    (accumulated tokens-per-minute). We retry only RateLimitError — a request
    that is simply too large fails fast and is allowed to surface. Retries are
    logged and capped; after the cap the error propagates. Nothing is hidden."""
    from groq import RateLimitError

    for attempt in range(MAX_RATE_LIMIT_RETRIES):
        try:
            return _client().chat.completions.create(**kwargs)
        except RateLimitError as e:
            wait = min(2 ** attempt, 30)
            logging.warning(
                "Groq 429 rate limit (attempt %d/%d), sleeping %ds: %s",
                attempt + 1, MAX_RATE_LIMIT_RETRIES, wait, e,
            )
            time.sleep(wait)
    return _client().chat.completions.create(**kwargs)  # last try — let it raise


def _chat(_trace_meta: dict | None = None, **kwargs):
    """_groq_call, optionally recorded as a Langfuse `generation` observation
    (model, prompt, completion, and real token usage). Tracing never changes
    the call or its result — with keys absent this is just _groq_call."""
    if not _LANGFUSE_ON:
        return _groq_call(**kwargs)
    with _lf_client().start_as_current_observation(
        name=f"groq:{kwargs.get('model')}",
        as_type="generation",
        model=kwargs.get("model"),
        input=kwargs.get("messages"),
        model_parameters={"temperature": kwargs.get("temperature")},
        metadata=_trace_meta,
    ) as gen:
        resp = _groq_call(**kwargs)
        gen.update(
            output=resp.choices[0].message.content,
            usage_details={
                "input": resp.usage.prompt_tokens,
                "output": resp.usage.completion_tokens,
            },
        )
        return resp


# --------------------------------------------------------------------------
# Langfuse tracing — env-gated. With keys, every node becomes a traced span;
# without them, `traced` is the identity function and one line is logged. The
# search behaves identically either way — tracing never changes results.
# --------------------------------------------------------------------------
_LANGFUSE_ON = bool(
    os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")
)
if _LANGFUSE_ON:
    from langfuse import get_client as _lf_client
    from langfuse import observe as _observe

    logging.info("Langfuse tracing ENABLED")

    def traced(fn):
        return _observe(name=fn.__name__)(fn)
else:
    logging.info("Langfuse tracing disabled (no LANGFUSE_* keys) — running without it")

    def traced(fn):
        return fn


def flush_traces() -> None:
    """Send any buffered traces before the process exits (Langfuse batches)."""
    if _LANGFUSE_ON:
        _lf_client().flush()


def search_trace(**metadata):
    """A root span grouping one whole search run into a single Langfuse trace,
    or a no-op context manager when tracing is off."""
    if _LANGFUSE_ON:
        return _lf_client().start_as_current_observation(
            name="queryforge-search", as_type="span", metadata=metadata or None
        )
    return contextlib.nullcontext()


# --------------------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------------------
_SYSTEM = (
    "You are a PostgreSQL indexing expert doing physical database design. "
    "Given a fixed read-only workload and a storage budget, you propose a set "
    "of B-tree indexes that minimize total execution time. You reason about "
    "join keys, selective predicates, and sort/group columns, and you avoid "
    "indexing tiny tables or columns that only appear in output. You respond "
    "with JSON only."
)

_ANALYZE_SYSTEM = (
    "You are a database performance engineer writing a short, honest summary of "
    "an index-tuning experiment for a colleague. Reply in plain prose — no JSON, "
    "no markdown headings, just 4-6 sentences. Do not invent numbers."
)


def _build_schema_text(workload: dict[str, str]) -> str:
    cols = specs.candidate_columns(workload)
    with psycopg.connect(AGENT_DSN) as conn:
        counts = {
            t: n
            for t, n in [
                (t, conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0]) for t in cols
            ]
        }
    lines = ["TABLES — row count | candidate columns worth indexing:"]
    for table, cs in cols.items():
        n = counts[table]
        hint = "   <-- tiny table, indexing is pointless" if n < TINY_TABLE_ROWS else ""
        lines.append(f"  {table:<10} {n:>8,} | {', '.join(cs)}{hint}")
    return "\n".join(lines)


# Parse our own deterministic DDL back into a name (for size lookup) and a
# compact human form (for the prompt). Safe because we generate the DDL.
_DDL_NAME = re.compile(r"CREATE INDEX (\w+)", re.I)
_DDL_SHAPE = re.compile(
    r"ON (\w+) USING \w+ \(([^)]*)\)(?:\s+INCLUDE \(([^)]*)\))?", re.I
)


def _ddl_name(ddl: str) -> str:
    m = _DDL_NAME.match(ddl.strip())
    return m.group(1) if m else ddl


def _ddl_human(ddl: str) -> str:
    m = _DDL_SHAPE.search(ddl)
    if not m:
        return ddl
    s = f"{m.group(1)}({m.group(2)})"
    if m.group(3):
        s += f"+INCLUDE({m.group(3)})"
    return s


def _find_config_entry(state: ForgeState, ddls: list[str]) -> dict | None:
    for h in state["history"]:
        if h.get("ddls") == ddls and "per_query_ms" in h:
            return h
    return None


def _format_top(state: ForgeState) -> str:
    tops = archive.top_k(state["history"], 3)
    if not tops:
        return "(none within budget yet — propose a SMALL config to establish one)"
    blocks = []
    for i, h in enumerate(tops, 1):
        sizes = h.get("per_index_mb", {})
        idx = "\n".join(
            f"     {_ddl_human(d)} = {sizes.get(_ddl_name(d), 0):.1f} MB" for d in h["ddls"]
        ) or "     (no indexes)"
        blocks.append(
            f"  #{i}  p50={h['p50_total_ms']:.0f} ms  total storage={h['storage_mb']:.1f} MB\n{idx}"
        )
    return "\n".join(blocks)


def _format_best_per_query(state: ForgeState) -> str:
    entry = _find_config_entry(state, state["best_config"])
    pq = entry["per_query_ms"] if entry else {}
    if not pq:
        return "(not available)"
    ranked = sorted(pq.items(), key=lambda kv: kv[1], reverse=True)
    head = "  ".join(f"{q}={ms:.0f}" for q, ms in ranked[:8])
    return f"  slowest first: {head}\n  (baseline total {state['baseline_ms']:.0f} ms, best total {state['best_ms']:.0f} ms)"


def _format_failures(failures: list[dict]) -> str:
    if not failures:
        return "(none)"
    return "\n".join(f"  DDL: {f['ddl']}\n  ERROR: {f['error']}" for f in failures)


def _size_guidance(state: ForgeState) -> str:
    """Real per-index size range, computed from every index THIS run has
    actually measured so far — never a hardcoded number (CLAUDE.md: 'do not
    invent, estimate, or extrapolate any metric'). Honest and data-free on
    generation 1, before anything has been benchmarked yet."""
    single, multi = [], []
    for h in archive.benchmarked(state["history"]):
        for name, mb in h.get("per_index_mb", {}).items():
            ddl = next((d for d in h.get("ddls", []) if _ddl_name(d) == name), None)
            if ddl and "INCLUDE" not in ddl.upper() and ddl.count(",") == 0:
                single.append(mb)
            else:
                multi.append(mb)
    if not single and not multi:
        return ("No index sizes measured on this database yet — after your first "
                "proposal, real measured sizes will appear here. Start small "
                "(single-column indexes) so the first measurement is informative.")
    parts = []
    if single:
        parts.append(f"single-column indexes measured so far: {min(single):.1f}-{max(single):.1f} MB")
    if multi:
        parts.append(f"multi-column/INCLUDE indexes measured so far: {min(multi):.1f}-{max(multi):.1f} MB")
    return "Measured on THIS database, this run: " + "; ".join(parts) + "."


def _format_workload(state: ForgeState) -> str:
    """Per-query access fingerprint + measured baseline cost — the LLM's view
    of the workload, in place of raw SQL (see specs.query_fingerprints)."""
    base = _find_config_entry(state, [])
    base_pq = base["per_query_ms"] if base else {}
    lines = []
    for qid, tables in specs.query_fingerprints(state["workload"]).items():
        cost = base_pq.get(qid)
        cost_s = f"{cost:.0f}ms" if cost is not None else "?"
        access = ", ".join(f"{t}[{','.join(cs)}]" for t, cs in tables.items())
        lines.append(f"  {qid} (base {cost_s}): {access}")
    return "\n".join(lines)


def _build_propose_prompt(state: ForgeState) -> str:
    best_entry = _find_config_entry(state, state["best_config"])
    best_storage = best_entry["storage_mb"] if best_entry else 0.0
    return f"""Propose an index configuration that lowers total workload latency.

## 1. SCHEMA (row counts + the only columns you may index)
{state['schema_ddl']}

## 2. WORKLOAD — what each query accesses, with measured baseline cost
Columns are those used in WHERE / join / GROUP BY / ORDER BY. Columns listed
together for one query are candidates for a single composite index.
{_format_workload(state)}

## 3. STORAGE BUDGET
Budget: {state['storage_budget_mb']:.1f} MB. Current best uses {best_storage:.1f} MB.
{_size_guidance(state)}
Prefer single-column indexes over wide multi-column ones; you must stay under
budget or the whole config is rejected.

## 4. BEST CONFIGURATIONS MEASURED SO FAR (ground truth)
{_format_top(state)}

## 5. PER-QUERY LATENCY OF THE CURRENT BEST (measured, ms)
{_format_best_per_query(state)}
Target the slowest queries — that is where index gains are largest.

## 6. FAILURES FROM THE LAST ATTEMPT — fix these, do not repeat them
{_format_failures(state['last_failures'])}

Respond with JSON ONLY, exactly this shape:
{{"reasoning": "<one short paragraph>",
  "indexes": [{{"table": "lineitem", "columns": ["l_partkey"], "include": [], "method": "btree"}}]}}

Rules: btree only; use only columns from section 1, and every column of an
index must belong to that index's own table (never INCLUDE a column from a
different table). Prefer SINGLE-COLUMN indexes on the join keys and selective
predicates of the SLOWEST queries — a few of those usually capture most of the
gain. Propose a COMPLETE configuration (the full set you want built), not a
diff. STAY UNDER THE STORAGE BUDGET (see section 3 for measured sizes): a config
over budget scores nothing, so when in doubt propose FEWER, smaller indexes.
Never index tiny tables; never add an index whose leading column duplicates
another you already propose."""


def _build_analyze_prompt(state: ForgeState) -> str:
    base_entry = _find_config_entry(state, [])
    best_entry = _find_config_entry(state, state["best_config"])
    base_pq = base_entry["per_query_ms"] if base_entry else {}
    best_pq = best_entry["per_query_ms"] if best_entry else {}
    regressed = [q for q in base_pq if best_pq.get(q, 0) > base_pq.get(q, 0)]
    best_ddl = "\n".join(f"  {d}" for d in state["best_config"]) or "  (no indexes)"
    return f"""Summarize this index-tuning search for an engineer. Be concrete and honest.

Baseline (no indexes) total p50: {state['baseline_ms']:.0f} ms
Best configuration total p50   : {state['best_ms']:.0f} ms
Improvement                    : {(1 - state['best_ms'] / state['baseline_ms']) * 100:.1f}%
Storage used by best           : {best_entry['storage_mb'] if best_entry else 0:.1f} MB
Generations run                : {state['generation']}
Queries that REGRESSED vs baseline: {regressed or 'none'}

Best configuration:
{best_ddl}

Write 4-6 sentences: what the search found, which queries drove the win, any
regressions and the likely trade-off behind them, and whether the result looks
credible. Do not invent numbers beyond those given."""


# --------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------
@traced
def baseline_node(state: ForgeState) -> dict:
    workload = state["workload"]
    result = oracle.benchmark([], workload=workload)  # no indexes — the reference line
    return {
        "schema_ddl": _build_schema_text(workload),
        "baseline_ms": result["p50_total_ms"],
        "best_ms": result["p50_total_ms"],
        "best_config": [],
        "generation": 0,
        "stagnation": 0,
        "consecutive_rejects": 0,
        "candidate": [],
        "last_result": result,
        "last_failures": [],
        "history": [
            {
                "event": "baseline",
                "generation": 0,
                "ddls": [],
                "p50_total_ms": result["p50_total_ms"],
                "storage_mb": result["storage_mb"],
                "per_query_ms": result["per_query_ms"],
                "timed_out": result["timed_out"],
            }
        ],
    }


@traced
def propose_node(state: ForgeState) -> dict:
    gen = state["generation"] + 1
    resp = _chat(
        _trace_meta={"node": "propose", "generation": gen},
        model=MODEL_PROPOSE,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": _build_propose_prompt(state)},
        ],
        response_format={"type": "json_object"},
        temperature=0.5,
        max_tokens=1200,
    )
    content = resp.choices[0].message.content
    tok = {
        "tokens_prompt": resp.usage.prompt_tokens,
        "tokens_completion": resp.usage.completion_tokens,
    }
    try:
        proposal = Proposal.model_validate_json(content)
        # IndexSpec.__post_init__ raises ValueError if any field (table,
        # column, include, method) isn't a bare SQL identifier — caught here
        # alongside a Pydantic parse failure, since both mean "this proposal
        # cannot safely become DDL" and get the same treatment: reject and
        # feed the exact reason back, never silently drop or crash the node.
        ddls = sorted({spec.to_ddl() for spec in (p.to_spec() for p in proposal.indexes)})
    except (ValidationError, ValueError) as e:
        # A malformed proposal is a rejected proposal — surface it, feed it back,
        # never silently retry. The empty candidate makes the gate reject.
        return {
            "candidate": [],
            "last_failures": [
                {"ddl": content[:400], "error": f"LLM output did not parse: {e}"}
            ],
            "history": [{"event": "propose", "generation": gen, "parse_error": str(e), **tok}],
        }
    return {
        "candidate": ddls,
        "history": [
            {"event": "propose", "generation": gen, "reasoning": proposal.reasoning, **tok}
        ],
    }


@traced
def validate_node(state: ForgeState) -> dict:
    candidate = state["candidate"]
    reasons: list[str] = []

    if not candidate:
        reasons.append("empty proposal (no indexes)")
    for stmt in candidate:
        if not specs.validate(stmt):
            reasons.append(f"blocked by allowlist: {stmt}")
    if candidate:
        if archive.seen_exact(state["history"], candidate):
            reasons.append("configuration already benchmarked (exact duplicate)")
        elif archive.near_duplicate(state["history"], candidate):
            reasons.append("configuration nearly identical to one already benchmarked")

    if not reasons:
        return {
            "route": "ok",
            "consecutive_rejects": 0,
            "history": [{"event": "validate", "generation": state["generation"] + 1, "decision": "ok"}],
        }

    rejects = state["consecutive_rejects"] + 1
    summary = {"ddl": " ; ".join(candidate) or "(empty proposal)", "error": "; ".join(reasons)}

    if rejects >= MAX_CONSECUTIVE_REJECTS:
        # Anti-spin: stop burning tokens, fall back to the best-known config.
        return {
            "route": "ok",
            "candidate": list(state["best_config"]),
            "consecutive_rejects": 0,
            "last_failures": [summary],
            "history": [
                {"event": "validate", "generation": state["generation"] + 1,
                 "decision": "forced_ok", "reasons": reasons}
            ],
        }
    return {
        "route": "reject",
        "consecutive_rejects": rejects,
        "last_failures": [summary],
        "history": [
            {"event": "reject", "generation": state["generation"] + 1, "reasons": reasons}
        ],
    }


@traced
def benchmark_node(state: ForgeState) -> dict:
    result = oracle.benchmark(state["candidate"], workload=state["workload"])  # the stopwatch — ground truth
    return {"last_result": result}


@traced
def archive_node(state: ForgeState) -> dict:
    result = state["last_result"]
    candidate = state["candidate"]
    budget = state["storage_budget_mb"]
    p50 = result["p50_total_ms"]
    storage = result["storage_mb"]
    gen = state["generation"] + 1
    per_index = result.get("per_index_mb", {})

    # ONLY the DDLs that actually got an index built in Postgres — a statement
    # in `candidate` that failed (recorded in failed_ddl) has no entry in
    # per_index_mb and must never be reported as part of a measured config;
    # everything downstream (best_config, history, the LLM prompt) treats
    # `built_ddls` as ground truth, so a never-created index can't leak into it.
    built_ddls = [d for d in candidate if _ddl_name(d) in per_index]

    within_budget = storage <= budget
    improved = within_budget and p50 < state["best_ms"]

    # Feedback for the next prompt: the real DDL errors, plus a MEASURED
    # over-budget note naming each BUILT index's actual size so the LLM can
    # cut the biggest ones — never a failed statement, which would misread as
    # "a real 0MB index" instead of "this statement never ran".
    failures = list(result["failed_ddl"])
    if not within_budget:
        detail = ", ".join(f"{_ddl_human(d)}={per_index[_ddl_name(d)]:.1f}MB" for d in built_ddls)
        failures.append(
            {
                "ddl": " ; ".join(built_ddls) or "(no indexes built)",
                "error": f"configuration measured {storage:.1f} MB, over the "
                f"{budget:.1f} MB budget. Per-index sizes: {detail}. Drop the "
                f"largest indexes until the total is under {budget:.1f} MB.",
            }
        )

    updates: dict = {
        "generation": gen,
        "last_failures": failures,
        "history": [
            {
                "event": "benchmark",
                "generation": gen,
                "ddls": built_ddls,
                "p50_total_ms": p50,
                "storage_mb": storage,
                "per_index_mb": per_index,
                "per_query_ms": result["per_query_ms"],
                "within_budget": within_budget,
                "failed_ddl": result["failed_ddl"],
                "timed_out": result["timed_out"],
            }
        ],
    }
    if improved:
        updates.update(best_ms=p50, best_config=built_ddls, stagnation=0)
    else:
        updates["stagnation"] = state["stagnation"] + 1
    return updates


@traced
def analyze_node(state: ForgeState) -> dict:
    resp = _chat(
        _trace_meta={"node": "analyze", "generation": state["generation"]},
        model=MODEL_ANALYZE,
        messages=[
            {"role": "system", "content": _ANALYZE_SYSTEM},
            {"role": "user", "content": _build_analyze_prompt(state)},
        ],
        temperature=0.3,
        max_tokens=700,
    )
    return {
        "history": [
            {
                "event": "analyze",
                "generation": state["generation"],
                "analysis": resp.choices[0].message.content,
                "tokens_prompt": resp.usage.prompt_tokens,
                "tokens_completion": resp.usage.completion_tokens,
            }
        ]
    }


# --------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------
def gate(state: ForgeState) -> str:
    return state["route"]


def should_continue(state: ForgeState) -> str:
    cap = state.get("max_generations", MAX_GENERATIONS)
    if state["generation"] >= cap:
        return "stop"
    # Stagnation may only END the search AFTER a within-budget config has been
    # accepted (best_config non-empty). Otherwise, if early proposals are all
    # over budget, best never improves, stagnation climbs, and the search would
    # quit before it ever lands a valid config — which is exactly what happened.
    if state["best_config"] and state["stagnation"] >= MAX_STAGNATION:
        return "stop"
    return "loop"


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------
def build_graph():
    g = StateGraph(ForgeState)
    g.add_node("baseline", baseline_node)
    g.add_node("propose", propose_node)
    g.add_node("validate", validate_node)
    g.add_node("benchmark", benchmark_node)
    g.add_node("archive", archive_node)
    g.add_node("analyze", analyze_node)

    g.set_entry_point("baseline")
    g.add_edge("baseline", "propose")
    g.add_edge("propose", "validate")
    g.add_conditional_edges("validate", gate, {"ok": "benchmark", "reject": "propose"})
    g.add_edge("benchmark", "archive")
    g.add_conditional_edges("archive", should_continue, {"loop": "propose", "stop": "analyze"})
    g.add_edge("analyze", END)

    return g.compile(checkpointer=MemorySaver())

"""Stage 2: the oracle. A real Postgres benchmark, not a cost model.

This is the ground truth of the whole project. Everything downstream trusts
these numbers, so the numbers must be measured, deterministic, and honest:
  - timing comes from Postgres itself (EXPLAIN ANALYZE "Execution Time"),
    never Python wall-clock, so round-trip/serialization noise is excluded;
  - a bad DDL statement is recorded and stepped over, never hidden;
  - a query that blows the timeout is recorded and scored at the penalty,
    never silently dropped.

Public surface:
    benchmark(index_ddl) -> dict     # apply indexes, measure the workload
    reset_indexes()                  # drop all non-PK, non-unique indexes
    storage_mb()                     # size of those indexes, in MiB
"""

import json
import os
import statistics
from pathlib import Path

import psycopg
from psycopg import errors as pgerr
from dotenv import load_dotenv

load_dotenv()

AGENT_DSN = os.environ.get(
    "PG_AGENT_DSN", "postgresql://queryforge_agent:agentpw@localhost:5432/queryforge"
)
WORKLOAD_DIR = Path(__file__).parent / "workload"

TIMEOUT_MS = 30_000          # statement_timeout; also the penalty score on a hit
TIMED_PASSES = 3            # per-query result is the MEDIAN of these

# The set of indexes the agent is allowed to create and we therefore measure:
# everything in schema public that is NOT a primary-key or unique-constraint
# index. Those two are structural (created by load_data.py) and off-limits.
_MANAGED_INDEX_FILTER = """
    n.nspname = 'public'
    AND NOT i.indisprimary
    AND NOT i.indisunique
"""


def _load_workload() -> dict[str, str]:
    queries: dict[str, str] = {}
    for i in range(1, 23):
        text = (WORKLOAD_DIR / f"q{i}.sql").read_text(encoding="utf-8")
        queries[f"q{i}"] = text.strip().rstrip(";")
    return queries


WORKLOAD = _load_workload()


def reset_indexes() -> None:
    """Drop every managed (non-PK, non-unique) index in schema public.

    Called at the start of every benchmark so a run is deterministic given
    only its DDL — never contaminated by indexes a previous run left behind.
    """
    with psycopg.connect(AGENT_DSN, autocommit=True) as conn:
        names = conn.execute(
            f"""
            SELECT c.relname
            FROM pg_index i
            JOIN pg_class c ON c.oid = i.indexrelid
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE {_MANAGED_INDEX_FILTER}
            """
        ).fetchall()
        for (name,) in names:
            conn.execute(f'DROP INDEX IF EXISTS "{name}"')


def _storage_mb(conn: psycopg.Connection) -> float:
    """MiB of managed index storage, measured on an existing connection."""
    row = conn.execute(
        f"""
        SELECT COALESCE(SUM(pg_relation_size(i.indexrelid)), 0) / 1048576.0
        FROM pg_index i
        JOIN pg_class c ON c.oid = i.indexrelid
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE {_MANAGED_INDEX_FILTER}
        """
    ).fetchone()
    return float(row[0])


def storage_mb() -> float:
    """MiB of managed index storage, standalone (opens its own connection)."""
    with psycopg.connect(AGENT_DSN) as conn:
        return _storage_mb(conn)


def _index_sizes(conn: psycopg.Connection) -> dict[str, float]:
    """{index_name: MiB} for each managed index — the per-index breakdown that
    lets the agent see which specific index is eating the storage budget."""
    rows = conn.execute(
        f"""
        SELECT c.relname, pg_relation_size(i.indexrelid) / 1048576.0
        FROM pg_index i
        JOIN pg_class c ON c.oid = i.indexrelid
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE {_MANAGED_INDEX_FILTER}
        """
    ).fetchall()
    return {name: round(float(mb), 3) for name, mb in rows}


def _apply_indexes(conn: psycopg.Connection, index_ddl: list[str]) -> list[dict]:
    """Create each index inside its own SAVEPOINT. A statement that fails is
    recorded with the EXACT Postgres error and rolled back to the savepoint;
    the outer transaction survives so the remaining statements still run.

    The connection is in autocommit mode, so the outer `conn.transaction()`
    issues a real BEGIN and each nested `conn.transaction()` a SAVEPOINT.

    A statement-level error (bad column, duplicate name — psycopg.Error minus
    OperationalError) is exactly what savepoints exist to isolate: recorded,
    rolled back, the loop continues. A connection-LEVEL failure
    (OperationalError — the link itself died) leaves every later statement on
    that connection doomed to fail for an unrelated reason; misattributing
    those as individual bad DDL would corrupt the failure feedback the LLM
    reads. So an OperationalError aborts the loop and propagates instead of
    being recorded as a per-statement failure.
    """
    failed: list[dict] = []
    with conn.transaction():  # outer BEGIN ... COMMIT
        for ddl in index_ddl:
            try:
                with conn.transaction():  # SAVEPOINT ... RELEASE
                    conn.execute(ddl)
            except psycopg.OperationalError:
                raise  # connection-level failure — do not misattribute to this statement
            except psycopg.Error as e:
                # savepoint already rolled back by the context manager
                failed.append({"ddl": ddl, "error": str(e).strip()})
    return failed


def explain_check(sql: str) -> str | None:
    """Validate a user-submitted query against the REAL schema without running
    it. Returns None if the query plans cleanly, or the EXACT Postgres error
    string if it does not (bad table, bad column, syntax error).

    Uses a plain EXPLAIN — no ANALYZE, so nothing is executed and no rows are
    read; the planner only has to resolve names and types. We let Postgres be
    the source of truth for "is this valid here" instead of hand-rolling a
    second SQL parser, exactly as timing trusts EXPLAIN ANALYZE over a
    Python stopwatch. This is a validity gate only — specs.validate_select()
    is the security allowlist and must be applied FIRST.
    """
    try:
        with psycopg.connect(AGENT_DSN, autocommit=True) as conn:
            conn.execute(f"EXPLAIN {sql}")
        return None
    except psycopg.Error as e:
        return str(e).strip()


def _execution_time_ms(conn: psycopg.Connection, sql: str) -> float | None:
    """Run one query under EXPLAIN and return Postgres's own "Execution Time".

    Returns None if the statement hit statement_timeout (QueryCanceled) — the
    caller records the timeout and applies the penalty score. Any OTHER
    database error is a genuine problem and is allowed to propagate loudly.
    """
    explain = f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {sql}"
    try:
        row = conn.execute(explain).fetchone()
    except pgerr.QueryCanceled:
        return None
    plan = row[0]
    if isinstance(plan, str):        # some setups return json as text
        plan = json.loads(plan)
    return float(plan[0]["Execution Time"])


def benchmark(index_ddl: list[str], workload: dict[str, str] | None = None) -> dict:
    """Apply `index_ddl`, then measure a workload.

    `workload` is a {qid: sql} dict; with the default (None) it measures the
    fixed 22-query TPC-H WORKLOAD that baselines/run_search/variance all use. A
    custom workload (from /custom) is the same shape and measured identically —
    the oracle does not care where the queries came from.

    Returns:
        {
          "p50_total_ms": float,          # sum of the per-query medians
          "per_query_ms": {qid: float},   # median execution time per query
          "storage_mb": float,            # managed index storage in MiB
          "failed_ddl": [{"ddl", "error"}],
          "timed_out": [qid],             # queries that hit statement_timeout
        }

    Deterministic given the same DDL and workload: it resets indexes, applies
    the DDL, ANALYZEs (fresh stats — without this the planner ignores new
    indexes), runs one discarded warm-up pass, then takes the median of
    TIMED_PASSES.

    A query that hits statement_timeout is scored at the TIMEOUT_MS penalty
    and short-circuited: we do not re-run a timeout across the timed passes,
    because a query that cannot finish in the limit while warm will not finish
    on an identical rerun — paying that time again buys no new information.
    """
    wl = WORKLOAD if workload is None else workload
    reset_indexes()

    # --- apply DDL + refresh statistics, on an autocommit connection ---
    with psycopg.connect(AGENT_DSN, autocommit=True) as conn:
        conn.execute(f"SET statement_timeout = '{TIMEOUT_MS}ms'")
        failed_ddl = _apply_indexes(conn, index_ddl)
        conn.execute("ANALYZE")  # ALWAYS — the #1 reason "my index did nothing"
        storage = _storage_mb(conn)
        per_index_mb = _index_sizes(conn)

    # --- timing, on a fresh session with the planner pinned down ---
    samples: dict[str, list[float]] = {qid: [] for qid in wl}
    timed_out: set[str] = set()
    with psycopg.connect(AGENT_DSN, autocommit=True) as conn:
        conn.execute("SET max_parallel_workers_per_gather = 0")
        conn.execute("SET jit = off")
        conn.execute(f"SET statement_timeout = '{TIMEOUT_MS}ms'")

        # One warm-up pass: primes shared_buffers / OS cache (results discarded)
        # and discovers which queries time out so we can short-circuit them.
        for qid, sql in wl.items():
            if _execution_time_ms(conn, sql) is None:
                timed_out.add(qid)

        for _ in range(TIMED_PASSES):
            for qid, sql in wl.items():
                if qid in timed_out:
                    samples[qid].append(float(TIMEOUT_MS))  # known timeout: don't rerun
                    continue
                t = _execution_time_ms(conn, sql)
                if t is None:
                    timed_out.add(qid)
                    samples[qid].append(float(TIMEOUT_MS))
                else:
                    samples[qid].append(t)

    per_query_ms = {qid: round(statistics.median(s), 3) for qid, s in samples.items()}
    p50_total_ms = round(sum(per_query_ms.values()), 3)

    return {
        "p50_total_ms": p50_total_ms,
        "per_query_ms": per_query_ms,
        "storage_mb": round(storage, 3),
        "per_index_mb": per_index_mb,
        "failed_ddl": failed_ddl,
        "timed_out": sorted(timed_out),
    }


if __name__ == "__main__":
    # Smoke run: no indexes. Prints the shape of a result.
    result = benchmark([])
    print(f"p50_total_ms : {result['p50_total_ms']}")
    print(f"storage_mb   : {result['storage_mb']}")
    print(f"timed_out    : {result['timed_out']}")
    print(f"failed_ddl   : {result['failed_ddl']}")
    print("per_query_ms :")
    for qid, ms in result["per_query_ms"].items():
        print(f"  {qid:<4} {ms:>10.3f}")

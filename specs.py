"""Stage 3: the index vocabulary and the security gate.

Two jobs:
  1. IndexSpec — a frozen, hashable description of one index, with a
     deterministic DDL string and a canonical form used for dedup.
  2. validate() — Control 1 of the two-control security model: a strict
     allowlist that every statement must pass BEFORE it reaches Postgres.
     (Control 2 is the least-privilege queryforge_agent role from Stage 1.)

Plus candidate_columns(): the pruned set of columns worth indexing, parsed
from the 22 queries and cross-checked against the live catalog.
"""

import hashlib
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import psycopg
from dotenv import load_dotenv

load_dotenv()

AGENT_DSN = os.environ.get(
    "PG_AGENT_DSN", "postgresql://queryforge_agent:agentpw@localhost:5432/queryforge"
)
WORKLOAD_DIR = Path(__file__).parent / "workload"


# --------------------------------------------------------------------------
# IndexSpec
# --------------------------------------------------------------------------
# Every field ends up interpolated directly into to_ddl()'s f-string. A bare
# identifier is the only shape that can't smuggle extra DDL syntax (an
# undeclared INCLUDE clause, an extra column) into the single CREATE INDEX
# statement to_ddl() is meant to build — enforced once here rather than
# trusted at every call site.
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _check_identifier(value: str, what: str) -> None:
    if not _IDENTIFIER.match(value):
        raise ValueError(f"not a bare SQL identifier for {what}: {value!r}")


@dataclass(frozen=True)
class IndexSpec:
    """One index. Frozen so it is hashable and usable inside a frozenset;
    a set of IndexSpecs is a Configuration (see config_hash)."""

    table: str
    columns: tuple[str, ...]
    include: tuple[str, ...] = ()
    method: str = "btree"

    def __post_init__(self) -> None:
        _check_identifier(self.table, "table")
        for c in self.columns:
            _check_identifier(c, "column")
        for c in self.include:
            _check_identifier(c, "include column")
        _check_identifier(self.method, "method")

    def canonical(self) -> str:
        """Stable identity string: "table|cols|include|method".

        Two specs that would build the same index share a canonical form,
        so it is the key for exact dedup (config_hash) and the input to the
        near-duplicate check in archive.py."""
        cols = ",".join(self.columns)
        incl = ",".join(self.include)
        return f"{self.table}|{cols}|{incl}|{self.method}"

    def index_name(self) -> str:
        """Deterministic, collision-resistant, and inside Postgres's 63-char
        identifier limit. Same index => same name => reset_indexes cleans it."""
        digest = hashlib.sha1(self.canonical().encode()).hexdigest()[:12]
        return f"qf_{digest}"

    def to_ddl(self) -> str:
        cols = ", ".join(self.columns)
        ddl = f"CREATE INDEX {self.index_name()} ON {self.table} USING {self.method} ({cols})"
        if self.include:
            ddl += f" INCLUDE ({', '.join(self.include)})"
        return ddl


# Configuration identity (SHA1 over sorted DDL) and near-duplicate detection
# live in archive.py, which operates on the deterministic DDL strings a
# configuration is stored as. One identity model, not two.


# --------------------------------------------------------------------------
# Control 1 — statement allowlist
# --------------------------------------------------------------------------
# The LLM generates SQL we execute against a live database. This runs on every
# statement before it is sent. It is deliberately strict and has no bypass:
# if a statement is not plainly a single CREATE INDEX, it does not run.
ALLOWED = re.compile(r"^\s*CREATE\s+INDEX\s+", re.I)
_DENY = re.compile(
    r"\b(DROP|ALTER|DELETE|UPDATE|INSERT|COPY|GRANT|REVOKE|CREATE\s+(?!INDEX))\b",
    re.I,
)


def validate(stmt: str) -> bool:
    """True only if `stmt` is a single, bare CREATE INDEX statement.

    Rejects: anything not starting with CREATE INDEX; statement stacking
    (a second statement hidden after a semicolon); and any dangerous keyword
    (DROP/ALTER/DELETE/UPDATE/INSERT/COPY/GRANT/REVOKE, or CREATE of anything
    other than an INDEX)."""
    if not ALLOWED.match(stmt):
        return False
    # The stacking/deny checks run on the statement with string literals and
    # comments removed (_strip_noise, also used below for column parsing), so
    # a legitimate value like a WHERE ... LIKE '%delete%' predicate can't
    # false-positive: content inside '...' or a comment is inert to Postgres,
    # never parsed as SQL syntax, so it is safe to ignore for this check.
    stripped = _strip_noise(stmt)
    if ";" in stripped.rstrip().rstrip(";"):  # no statement stacking
        return False
    if _DENY.search(stripped):
        return False
    return True


# --------------------------------------------------------------------------
# Control 1, read path — allowlist for user-submitted SELECT queries
# --------------------------------------------------------------------------
# The /custom endpoint executes ARBITRARY user SQL against the sandbox. Unlike
# the CREATE INDEX path (where the least-privilege role is a second independent
# control), here the role OWNS the TPC-H tables and so CAN drop/alter them — so
# for reads this allowlist is the ONLY thing standing between user input and
# table destruction. It is therefore built to be at least as strict:
#   - the statement must be a single read (starts with SELECT or WITH — WITH so
#     CTEs, which Q15 already uses, are allowed);
#   - no statement stacking;
#   - none of the DDL/DML deny-keywords;
#   - and additionally NO `INTO`, which blocks `SELECT ... INTO new_table` —
#     a way to create a table through a SELECT that the CREATE-INDEX deny-list
#     was never designed to catch.
_SELECT_ALLOWED = re.compile(r"^\s*(SELECT|WITH)\s", re.I)
_SELECT_DENY = re.compile(
    r"\b(DROP|ALTER|DELETE|UPDATE|INSERT|COPY|GRANT|REVOKE|TRUNCATE|CREATE|INTO)\b",
    re.I,
)


def validate_select(stmt: str) -> bool:
    """True only if `stmt` is a single read-only SELECT/WITH query.

    Rejects: anything not starting with SELECT or WITH; statement stacking; and
    any write/DDL keyword (DROP/ALTER/DELETE/UPDATE/INSERT/COPY/GRANT/REVOKE/
    TRUNCATE/CREATE) or INTO. As with validate(), the stacking and deny checks
    run on the statement with string literals and comments stripped, so a value
    inside quotes (e.g. a LIKE '%into%' predicate) can't false-positive."""
    if not _SELECT_ALLOWED.match(stmt):
        return False
    stripped = _strip_noise(stmt)
    if ";" in stripped.rstrip().rstrip(";"):  # no statement stacking
        return False
    if _SELECT_DENY.search(stripped):
        return False
    return True


# --------------------------------------------------------------------------
# Candidate column pruning
# --------------------------------------------------------------------------
def _strip_noise(sql: str) -> str:
    """Remove line/block comments and string literals so column-name matching
    can't be fooled by text inside quotes or comments."""
    sql = re.sub(r"--[^\n]*", " ", sql)
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.S)
    sql = re.sub(r"'(?:[^']|'')*'", " ", sql)  # single-quoted literals
    return sql


def _columns_in_query(sql: str, known: set[str]) -> set[str]:
    """Real columns referenced OUTSIDE a SELECT projection list, at any
    nesting level. A column used only to project output (SELECT ...) is not an
    access-path candidate; a column in WHERE / a comma-join predicate /
    GROUP BY / ORDER BY / HAVING is. TPC-H uses comma-style joins, so join
    predicates live in WHERE and are caught here too.

    A tiny clause state machine walks the tokens; parentheses push/pop the
    surrounding clause so a subquery's own SELECT list is excluded but its
    WHERE is not."""
    tokens = re.findall(r"\w+|\(|\)", _strip_noise(sql).lower())
    found: set[str] = set()
    state = "start"
    stack: list[str] = []
    for tok in tokens:
        if tok == "(":
            stack.append(state)
        elif tok == ")":
            state = stack.pop() if stack else state
        elif tok == "select":
            state = "select"
        elif tok == "from":
            state = "from"
        elif tok in ("where", "having", "group", "order"):
            state = "predicate"
        elif state != "select" and tok in known:
            found.add(tok)
    return found


def _fetch_columns(conn: psycopg.Connection) -> dict[str, str]:
    """Map every real column name to its table. TPC-H column names are globally
    unique (each carries a table prefix like l_ / o_ / ps_), which we assert."""
    rows = conn.execute(
        """
        SELECT table_name, column_name
        FROM information_schema.columns
        WHERE table_schema = 'public'
        """
    ).fetchall()
    col_to_table: dict[str, str] = {}
    for table, col in rows:
        assert col not in col_to_table, f"column name {col!r} is not unique"
        col_to_table[col] = table
    return col_to_table


def _load_default_workload() -> dict[str, str]:
    """The fixed 22-query TPC-H workload, read from workload/q*.sql. This is
    the default the whole experiment (baselines, run_search, variance) runs on;
    a user-submitted workload is a plain {qid: sql} dict in the same shape."""
    return {
        f"q{i}": (WORKLOAD_DIR / f"q{i}.sql").read_text(encoding="utf-8")
        for i in range(1, 23)
    }


def _compute_candidate_columns(workload: dict[str, str]) -> dict[str, list[str]]:
    with psycopg.connect(AGENT_DSN) as conn:
        col_to_table = _fetch_columns(conn)
    known = set(col_to_table)

    per_table: dict[str, set[str]] = defaultdict(set)
    for sql in workload.values():
        for col in _columns_in_query(sql, known):
            per_table[col_to_table[col]].add(col)
    return {t: sorted(cols) for t, cols in sorted(per_table.items())}


@lru_cache(maxsize=1)
def _default_candidate_columns() -> dict[str, list[str]]:
    return _compute_candidate_columns(_load_default_workload())


def candidate_columns(workload: dict[str, str] | None = None) -> dict[str, list[str]]:
    """{table: [candidate columns]} — the pruned index-candidate space, the
    union across a workload's queries, restricted to columns that actually
    exist in the catalog. With no argument, returns the cached result for the
    fixed 22-query TPC-H workload (what baselines/run_search/variance use); a
    workload dict computes the pruned set for a custom query set instead."""
    if workload is None:
        return _default_candidate_columns()
    return _compute_candidate_columns(workload)


def _compute_query_fingerprints(workload: dict[str, str]) -> dict[str, dict[str, list[str]]]:
    with psycopg.connect(AGENT_DSN) as conn:
        col_to_table = _fetch_columns(conn)
    known = set(col_to_table)

    out: dict[str, dict[str, list[str]]] = {}
    for qid, sql in workload.items():
        per: dict[str, list[str]] = defaultdict(list)
        for col in sorted(_columns_in_query(sql, known)):
            per[col_to_table[col]].append(col)
        out[qid] = dict(per)
    return out


@lru_cache(maxsize=1)
def _default_query_fingerprints() -> dict[str, dict[str, list[str]]]:
    return _compute_query_fingerprints(_load_default_workload())


def query_fingerprints(
    workload: dict[str, str] | None = None,
) -> dict[str, dict[str, list[str]]]:
    """{qid: {table: [columns]}} — a compact, per-query view of the columns
    each query accesses in WHERE / comma-join / GROUP BY / ORDER BY, grouped
    by table. This is what the LLM sees instead of raw SQL: it preserves the
    column CO-OCCURRENCE signal (which columns appear together in one query,
    the basis for a good composite index) at a fraction of the token cost, and
    it is exactly the access information a physical-design advisor reasons over.

    With no argument, returns the cached result for the fixed 22-query TPC-H
    workload; a workload dict computes fingerprints for a custom query set.
    """
    if workload is None:
        return _default_query_fingerprints()
    return _compute_query_fingerprints(workload)


def full_schema() -> dict[str, list[tuple[str, str]]]:
    """{table: [(column, data_type), ...]} — every column and its Postgres data
    type, for the human-facing insights panel. Unlike candidate_columns(), this
    is NOT pruned to any query set: it is the whole schema, so a user unfamiliar
    with TPC-H can see what they can query."""
    with psycopg.connect(AGENT_DSN) as conn:
        rows = conn.execute(
            """
            SELECT table_name, column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = 'public'
            ORDER BY table_name, ordinal_position
            """
        ).fetchall()
    out: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for table, col, dtype in rows:
        out[table].append((col, dtype))
    return dict(out)


def row_counts() -> dict[str, int]:
    """{table: row count} for every public table — measured live, so the
    insights panel shows real sizes (lineitem ~600k) rather than a guess."""
    with psycopg.connect(AGENT_DSN) as conn:
        tables = [
            t
            for (t,) in conn.execute(
                """
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public' ORDER BY table_name
                """
            ).fetchall()
        ]
        return {t: conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in tables}

# Note on the storage budget: there is deliberately NO size estimator here.
# True btree size depends on deduplication and fill factor and cannot be known
# without building the index, so the budget is enforced on the oracle's MEASURED
# storage_mb (see graph.archive_node), never on a guess. Nothing in this project
# reports a number it did not measure.


# --------------------------------------------------------------------------
# Gate output: validator self-tests + pruned candidate list
# --------------------------------------------------------------------------
def _validator_selftests() -> None:
    cases = [
        ("CREATE INDEX x ON t (a)", True),
        ("  create   index  x on t using btree (a) include (b)", True),
        ("CREATE INDEX x ON lineitem (l_partkey, l_shipdate)", True),
        ("CREATE INDEX x ON t (a) WHERE a > 0", True),          # partial index, allowed
        ("DROP TABLE t", False),
        ("CREATE TABLE t (a int)", False),
        ("UPDATE t SET a = 1", False),
        ("CREATE INDEX x ON t (a); DROP TABLE t", False),       # statement stacking
        ("CREATE INDEX x ON t (a); ", True),                    # lone trailing ; is fine
        ("SELECT 1", False),
        ("ALTER TABLE t ADD COLUMN a int", False),
    ]
    print("validate() self-tests:")
    all_ok = True
    for stmt, expected in cases:
        got = validate(stmt)
        ok = got == expected
        all_ok &= ok
        mark = "ok " if ok else "FAIL"
        print(f"  [{mark}] {got!s:<5} (want {expected!s:<5})  {stmt[:52]}")
    print(f"  => {'ALL PASS' if all_ok else 'FAILURES ABOVE'}\n")


def _select_validator_selftests() -> None:
    cases = [
        ("SELECT 1", True),
        ("select l_partkey from lineitem where l_quantity > 5", True),
        ("WITH r AS (SELECT 1 AS x) SELECT x FROM r", True),         # read-only CTE
        # Postgres allows data-modifying CTEs (WITH ... AS (DELETE ... RETURNING)):
        # the deny-list keyword catches them even though the statement is a WITH.
        ("WITH x AS (DELETE FROM orders RETURNING *) SELECT * FROM x", False),
        ("  SELECT count(*) FROM orders WHERE o_comment LIKE '%into%'", True),  # 'into' in a literal is inert
        ("SELECT * INTO backup FROM lineitem", False),               # SELECT ... INTO makes a table
        ("SELECT 1; DROP TABLE lineitem", False),                    # statement stacking
        ("DROP TABLE lineitem", False),
        ("DELETE FROM orders", False),
        ("UPDATE orders SET o_totalprice = 0", False),
        ("INSERT INTO orders VALUES (1)", False),
        ("CREATE TABLE t (a int)", False),
        ("TRUNCATE lineitem", False),
        ("SELECT 1;", True),                                          # lone trailing ; is fine
    ]
    print("validate_select() self-tests:")
    all_ok = True
    for stmt, expected in cases:
        got = validate_select(stmt)
        ok = got == expected
        all_ok &= ok
        mark = "ok " if ok else "FAIL"
        print(f"  [{mark}] {got!s:<5} (want {expected!s:<5})  {stmt[:52]}")
    print(f"  => {'ALL PASS' if all_ok else 'FAILURES ABOVE'}\n")


if __name__ == "__main__":
    _validator_selftests()
    _select_validator_selftests()

    spec = IndexSpec("lineitem", ("l_partkey", "l_shipdate"), include=("l_quantity",))
    print("IndexSpec example:")
    print(f"  canonical : {spec.canonical()}")
    print(f"  index_name: {spec.index_name()}")
    print(f"  to_ddl    : {spec.to_ddl()}")
    print(f"  validate  : {validate(spec.to_ddl())}\n")

    print("Pruned candidate columns (per table):")
    cols = candidate_columns()
    total = 0
    for table, cs in cols.items():
        total += len(cs)
        print(f"  {table:<10} ({len(cs):>2})  {', '.join(cs)}")
    print(f"\n  {total} candidate columns across {len(cols)} tables.")

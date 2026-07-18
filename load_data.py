"""Stage 1: generate TPC-H SF=0.1 with DuckDB, load it into Postgres,
extract the 22 benchmark queries, smoke-test them, print row counts.

Run once after `docker compose up -d`:  python load_data.py

Idempotent: drops and recreates the TPC-H tables on every run.
Everything in Postgres is done as `queryforge_agent` (the least-privilege
role from db/init.sql) — if this script can load the data as that role,
the oracle can benchmark as that role.
"""

import os
import sys
from pathlib import Path

import duckdb
import psycopg
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
WORKLOAD_DIR = ROOT / "workload"

AGENT_DSN = os.environ.get(
    "PG_AGENT_DSN", "postgresql://queryforge_agent:agentpw@localhost:5432/queryforge"
)

TABLES = [  # load order respects foreign-key-style dependencies (informational only)
    "region", "nation", "supplier", "customer", "part", "partsupp", "orders", "lineitem",
]

# Standard TPC-H spec column types (dbgen 2.18). PKs as per spec section 1.4.2.
SCHEMA_DDL = """
CREATE TABLE region (
    r_regionkey  integer        NOT NULL,
    r_name       char(25)       NOT NULL,
    r_comment    varchar(152),
    PRIMARY KEY (r_regionkey)
);
CREATE TABLE nation (
    n_nationkey  integer        NOT NULL,
    n_name       char(25)       NOT NULL,
    n_regionkey  integer        NOT NULL,
    n_comment    varchar(152),
    PRIMARY KEY (n_nationkey)
);
CREATE TABLE supplier (
    s_suppkey    integer        NOT NULL,
    s_name       char(25)       NOT NULL,
    s_address    varchar(40)    NOT NULL,
    s_nationkey  integer        NOT NULL,
    s_phone      char(15)       NOT NULL,
    s_acctbal    numeric(15,2)  NOT NULL,
    s_comment    varchar(101)   NOT NULL,
    PRIMARY KEY (s_suppkey)
);
CREATE TABLE customer (
    c_custkey    integer        NOT NULL,
    c_name       varchar(25)    NOT NULL,
    c_address    varchar(40)    NOT NULL,
    c_nationkey  integer        NOT NULL,
    c_phone      char(15)       NOT NULL,
    c_acctbal    numeric(15,2)  NOT NULL,
    c_mktsegment char(10)       NOT NULL,
    c_comment    varchar(117)   NOT NULL,
    PRIMARY KEY (c_custkey)
);
CREATE TABLE part (
    p_partkey     integer        NOT NULL,
    p_name        varchar(55)    NOT NULL,
    p_mfgr        char(25)       NOT NULL,
    p_brand       char(10)       NOT NULL,
    p_type        varchar(25)    NOT NULL,
    p_size        integer        NOT NULL,
    p_container   char(10)       NOT NULL,
    p_retailprice numeric(15,2)  NOT NULL,
    p_comment     varchar(23)    NOT NULL,
    PRIMARY KEY (p_partkey)
);
CREATE TABLE partsupp (
    ps_partkey    integer        NOT NULL,
    ps_suppkey    integer        NOT NULL,
    ps_availqty   integer        NOT NULL,
    ps_supplycost numeric(15,2)  NOT NULL,
    ps_comment    varchar(199)   NOT NULL,
    PRIMARY KEY (ps_partkey, ps_suppkey)
);
CREATE TABLE orders (
    o_orderkey      integer        NOT NULL,
    o_custkey       integer        NOT NULL,
    o_orderstatus   char(1)        NOT NULL,
    o_totalprice    numeric(15,2)  NOT NULL,
    o_orderdate     date           NOT NULL,
    o_orderpriority char(15)       NOT NULL,
    o_clerk         char(15)       NOT NULL,
    o_shippriority  integer        NOT NULL,
    o_comment       varchar(79)    NOT NULL,
    PRIMARY KEY (o_orderkey)
);
CREATE TABLE lineitem (
    l_orderkey      integer        NOT NULL,
    l_partkey       integer        NOT NULL,
    l_suppkey       integer        NOT NULL,
    l_linenumber    integer        NOT NULL,
    l_quantity      numeric(15,2)  NOT NULL,
    l_extendedprice numeric(15,2)  NOT NULL,
    l_discount      numeric(15,2)  NOT NULL,
    l_tax           numeric(15,2)  NOT NULL,
    l_returnflag    char(1)        NOT NULL,
    l_linestatus    char(1)        NOT NULL,
    l_shipdate      date           NOT NULL,
    l_commitdate    date           NOT NULL,
    l_receiptdate   date           NOT NULL,
    l_shipinstruct  char(25)       NOT NULL,
    l_shipmode      char(10)       NOT NULL,
    l_comment       varchar(44)    NOT NULL,
    PRIMARY KEY (l_orderkey, l_linenumber)
);
"""


def generate_csvs() -> None:
    """DuckDB's tpch extension replaces compiling dbgen. Deterministic output."""
    DATA_DIR.mkdir(exist_ok=True)
    con = duckdb.connect()  # in-memory
    con.execute("INSTALL tpch; LOAD tpch;")
    con.execute("CALL dbgen(sf=0.1);")
    for t in TABLES:
        out = (DATA_DIR / f"{t}.csv").as_posix()
        con.execute(f"COPY {t} TO '{out}' (FORMAT CSV, HEADER false)")
        print(f"  generated {t}.csv")
    con.close()


def extract_workload(con: duckdb.DuckDBPyConnection | None = None) -> None:
    """Write the 22 official TPC-H queries (DuckDB ships them; q15 is already
    in CTE form there, which is what Postgres needs — no CREATE VIEW)."""
    WORKLOAD_DIR.mkdir(exist_ok=True)
    con = duckdb.connect()
    con.execute("INSTALL tpch; LOAD tpch;")
    rows = con.execute("SELECT query_nr, query FROM tpch_queries()").fetchall()
    con.close()
    assert len(rows) == 22, f"expected 22 queries, got {len(rows)}"
    for nr, text in rows:
        (WORKLOAD_DIR / f"q{nr}.sql").write_text(text.strip(), encoding="utf-8")
    print(f"  wrote {len(rows)} queries to workload/")


def load_postgres() -> dict[str, int]:
    counts: dict[str, int] = {}
    with psycopg.connect(AGENT_DSN) as conn:
        with conn.cursor() as cur:
            for t in reversed(TABLES):  # drop children before parents
                cur.execute(f"DROP TABLE IF EXISTS {t} CASCADE")
            cur.execute(SCHEMA_DDL)
            for t in TABLES:
                path = DATA_DIR / f"{t}.csv"
                with cur.copy(f"COPY {t} FROM STDIN (FORMAT CSV)") as copy, open(path, "rb") as f:
                    while data := f.read(1 << 20):
                        copy.write(data)
                cur.execute(f"SELECT count(*) FROM {t}")
                counts[t] = cur.fetchone()[0]
            cur.execute("ANALYZE")
        conn.commit()
    return counts


def smoke_test_queries() -> None:
    """Every query must run on Postgres once, now — not fail later inside the
    oracle. Any error here is fatal and printed verbatim."""
    failures = []
    with psycopg.connect(AGENT_DSN) as conn:
        conn.execute("SET statement_timeout = '120s'")
        for i in range(1, 23):
            sql = (WORKLOAD_DIR / f"q{i}.sql").read_text(encoding="utf-8").rstrip().rstrip(";")
            try:
                conn.execute(sql)
                print(f"  q{i}: ok")
            except psycopg.Error as e:
                conn.rollback()
                failures.append((i, str(e).strip()))
                print(f"  q{i}: FAILED — {e}")
    if failures:
        print(f"\n{len(failures)} queries failed — fix before Stage 2:")
        for i, err in failures:
            print(f"  q{i}: {err}")
        sys.exit(1)


def main() -> None:
    print("[1/4] Generating TPC-H SF=0.1 via DuckDB...")
    generate_csvs()
    print("[2/4] Extracting 22-query workload...")
    extract_workload()
    print("[3/4] Loading into Postgres as queryforge_agent...")
    counts = load_postgres()
    print("[4/4] Smoke-testing all 22 queries on Postgres...")
    smoke_test_queries()
    print("\nRow counts:")
    for t in TABLES:
        print(f"  {t:<10} {counts[t]:>10,}")


if __name__ == "__main__":
    main()

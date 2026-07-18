"""Stage 7: an MCP server exposing the oracle as tools.

This is a thin wrapper — the LangGraph search imports oracle.py directly and
does NOT depend on this server, so nothing here can affect the search. It
exists so any MCP client (Claude Desktop, etc.) can drive the same measured
benchmark the agent uses.

The benchmark tool re-applies Control 1 (the specs.validate allowlist) before
touching the database: an MCP client is untrusted input just like the LLM is.

Run:  python mcp_server.py     (stdio transport)
"""

from mcp.server.fastmcp import FastMCP

import oracle
import specs

mcp = FastMCP("queryforge-oracle")


# The keys benchmark() ALWAYS returns, on both the success and the
# allowlist-rejection path, so a caller can unconditionally read
# result["p50_total_ms"] etc. without a KeyError — a rejection sets them to
# None and adds "error"/"rejected", it never removes them.
_BENCHMARK_KEYS = (
    "p50_total_ms", "per_query_ms", "storage_mb", "per_index_mb",
    "failed_ddl", "timed_out",
)


@mcp.tool()
def benchmark(index_ddl: list[str]) -> dict:
    """Apply CREATE INDEX statements and measure the 22-query TPC-H workload.

    Always returns a dict with these keys: p50_total_ms, per_query_ms,
    storage_mb, per_index_mb, failed_ddl (with exact Postgres errors), and
    timed_out. Every statement is checked against the CREATE INDEX allowlist
    first; if any fails, those keys are None and "error"/"rejected" are set
    instead — the call is refused rather than partially run, but the return
    shape never drops the documented keys."""
    rejected = [s for s in index_ddl if not specs.validate(s)]
    if rejected:
        return {
            **dict.fromkeys(_BENCHMARK_KEYS),
            "error": "rejected by allowlist (only bare CREATE INDEX allowed)",
            "rejected": rejected,
        }
    return oracle.benchmark(index_ddl)


@mcp.tool()
def reset_indexes() -> str:
    """Drop all managed (non-PK, non-unique) indexes in schema public."""
    oracle.reset_indexes()
    return "dropped all managed indexes"


@mcp.tool()
def storage_mb() -> float:
    """Total MiB of managed index storage currently in the database."""
    return oracle.storage_mb()


if __name__ == "__main__":
    mcp.run()

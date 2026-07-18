# QueryForge

LLM-guided search over Postgres index configurations for the 22-query TPC-H
workload (SF=0.1). The LLM proposes index sets; a real Postgres benchmark
("the oracle", oracle.py) measures them. **Measurement is ground truth. The
LLM never grades itself.** Formally: the Index Selection Problem (NP-hard) —
cost model replaced by a stopwatch, greedy enumeration replaced by an LLM
that reads its own failures.

Built by a final-year CS student who must explain every line in an interview.
Optimize for defensibility, not cleverness.

## Hard rules — override convenience every time

1. Build in stages. STOP at each gate and wait for explicit user approval.
2. NO mocked data, simulated benchmarks, or placeholder numbers. Unmeasured
   things do not get numbers. Estimates (e.g., pre-benchmark size gating)
   must be labeled estimates and never reported as metrics.
3. NO fallback that silently swallows errors. Failures are the most valuable
   signal — surface them, log them, feed them back to the LLM verbatim.
4. Boring code. No abstractions with one implementation. No config framework.
   No plugin system.
5. Unsure about intent? Ask. Do not guess and build.
6. Never write agent/graph code before oracle.py passes the variance gate
   (evals/variance_check.py, variance ≤ 5%).

## Stack — pinned, do not substitute

- Python 3.11, venv at ./venv
- PostgreSQL 16 in Docker (docker compose up -d), driver: psycopg v3 (NOT psycopg2)
- LangGraph: StateGraph + MemorySaver only. NO LangChain chains/agents/LLM wrappers.
- Groq SDK direct: llama-3.1-8b-instant for propose/mutate;
  llama-3.3-70b-versatile ONLY for the single final analyze call.
  (Free tier binds on tokens/day: 70B=100K TPD, 8B=500K TPD.)
- Structured output: Groq JSON mode + Pydantic model_validate_json.
  ValidationError = rejected proposal, fed back to model. Never retried silently.
- Dedup: SHA1 hash set (exact) + Jaccard over canonical index strings (near-dup),
  pure Python in archive.py. ChromaDB was deliberately cut — do not add it.
  Top-k-by-fitness is a sort, not a vector search.
- Langfuse via @observe, env-gated: no keys ⇒ no-op with ONE loud startup log.
- FastAPI on port 7860 (HF Spaces requirement).
- MCP: mcp.server.fastmcp.FastMCP — thin wrapper over oracle.py only;
  the graph imports oracle.py directly.

## Security — two independent controls, both mandatory

1. specs.py validate(): every statement must match ^\s*CREATE\s+INDEX,
   no statement stacking, deny-keywords (DROP/ALTER/DELETE/UPDATE/INSERT/
   COPY/GRANT/REVOKE/CREATE-non-INDEX). No skip flag, no dev bypass, ever.
2. queryforge_agent role: NOSUPERUSER NOCREATEDB NOCREATEROLE; owns only the
   TPC-H tables in the dedicated `queryforge` database.
   NOTE (interview point): PG16 requires table OWNERSHIP for CREATE INDEX and
   ANALYZE (MAINTAIN privilege is PG17), so the agent owns its sandbox tables;
   blast radius = the sandbox DB, nothing else. The regex allowlist is the
   primary control.

## Oracle invariants (oracle.py)

- Timing = EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) "Execution Time".
  NEVER Python wall-clock.
- Session: max_parallel_workers_per_gather=0, jit=off, statement_timeout='30s'.
- ALWAYS ANALYZE after DDL. 1 warm-up pass discarded, 3 timed passes,
  per-query median.
- Each DDL in a savepoint; psycopg.Error caught, exact error string recorded
  in failed_ddl, run continues. Timeout ⇒ qid in timed_out, scored 30000ms.
- reset_indexes() drops only non-primary, non-unique indexes in public.

## Commands

- Start DB:        docker compose up -d
- Load data:       python load_data.py           (prints row counts)
- Variance gate:   python evals/variance_check.py
- Full search:     python evals/run_search.py    (fixed seed, 20 generations)
- Baselines:       python baselines.py
- Serve:           uvicorn app:app --host 0.0.0.0 --port 7860

## Env (.env, never committed)

GROQ_API_KEY (required from Stage 4) · LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY /
LANGFUSE_HOST (optional) · PG_DSN / PG_AGENT_DSN

## Do not

- Do not invent, estimate, or extrapolate any metric.
- Do not add ChromaDB or any vector store.
- Do not add LangChain chains/agents.
- Do not add retries that hide DDL errors.
- Do not weaken the DDL allowlist or the least-privilege role.
- Do not tune the experiment until the LLM beats random search; if it does
  not beat random at equal benchmark budget, report that plainly.
- If regressed_queries reads 0 for 20 straight generations, the oracle is
  probably broken — say so loudly rather than celebrating.

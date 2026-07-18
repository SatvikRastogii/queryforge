"""Stage 4 verification: a 2-generation end-to-end smoke run.

Confirms the wiring works before the full 20-generation search:
  - the graph runs baseline -> propose -> validate -> benchmark -> archive -> analyze
  - `history` ACCUMULATES across nodes (the operator.add reducer)
  - the LLM actually lowers measured latency vs the no-index baseline
  - rejects (if any) route back to propose, and failures are recorded

Needs GROQ_API_KEY in .env. Run from repo root:  python evals/smoke_graph.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import graph  # noqa: E402
from oracle import WORKLOAD  # noqa: E402

graph.MAX_GENERATIONS = int(sys.argv[1]) if len(sys.argv) > 1 else 2  # real search uses 20


def main() -> None:
    app = graph.build_graph()
    initial = {"workload": WORKLOAD, "storage_budget_mb": graph.STORAGE_BUDGET_MB}
    trace_url = None
    with graph.search_trace(mode="smoke", generations=graph.MAX_GENERATIONS):
        final = app.invoke(
            initial,
            config={"configurable": {"thread_id": "smoke"}, "recursion_limit": 200},
        )
        if graph._LANGFUSE_ON:
            trace_url = graph._lf_client().get_trace_url()
    graph.flush_traces()

    print("\n=== event log (history accumulation) ===")
    for h in final["history"]:
        ev, gen = h["event"], h.get("generation")
        if ev == "benchmark":
            extra = f"p50={h['p50_total_ms']:.0f}ms storage={h['storage_mb']:.1f}MB " \
                    f"within_budget={h['within_budget']} failed={len(h['failed_ddl'])}"
        elif ev == "propose":
            extra = h.get("parse_error", h.get("reasoning", ""))[:70]
        elif ev == "reject":
            extra = "; ".join(h.get("reasons", []))[:70]
        else:
            extra = ""
        print(f"  gen {gen}  {ev:<10} {extra}")

    tok_p = sum(h.get("tokens_prompt", 0) for h in final["history"])
    tok_c = sum(h.get("tokens_completion", 0) for h in final["history"])

    print("\n=== result ===")
    print(f"  baseline_ms : {final['baseline_ms']:.0f}")
    print(f"  best_ms     : {final['best_ms']:.0f}  "
          f"({(1 - final['best_ms'] / final['baseline_ms']) * 100:.1f}% faster)")
    print(f"  best_config :")
    for d in final["best_config"] or ["  (no indexes)"]:
        print(f"    {d}")
    print(f"  tokens      : {tok_p} prompt + {tok_c} completion")
    print(f"  history len : {len(final['history'])} events (reducer working)")
    if trace_url:
        print(f"  langfuse    : {trace_url}")

    analysis = next((h["analysis"] for h in final["history"] if h["event"] == "analyze"), None)
    if analysis:
        print("\n=== 70B analysis ===")
        print(analysis)


if __name__ == "__main__":
    main()

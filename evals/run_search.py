"""Stage 6: the full search — 20 generations, real measurements, honest output.

Runs the LangGraph search to completion, records one row per generation to
evals/results.csv, and plots progress against the three baselines
(evals/baselines.json, from baselines.py) in evals/fitness.png.

The credibility column is `regressed_queries`: how many queries are SLOWER under
the current best than under the no-index baseline. Every real winning config
loses somewhere. If this reads 0 for every generation, the oracle is almost
certainly broken and the result is fiction — we say so rather than celebrate.

Run from repo root (after baselines.py):  python evals/run_search.py
"""

import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import graph  # noqa: E402
from oracle import WORKLOAD  # noqa: E402

EVALS = Path(__file__).resolve().parent
RESULTS_CSV = EVALS / "results.csv"
HISTORY_JSON = EVALS / "run_history.json"
FITNESS_PNG = EVALS / "fitness.png"
BASELINES_JSON = EVALS / "baselines.json"

COLUMNS = [
    "generation", "p50_total_ms", "best_ms", "storage_mb", "n_indexes",
    "failed_ddl_count", "rejected_count", "regressed_queries",
    "tokens_prompt", "tokens_completion", "wall_s",
]

# Okabe-Ito colorblind-safe palette (fixed-order categorical assignment).
C_BEST = "#0072B2"      # best-so-far (the search's progress)
C_VALID = "#009E73"     # a within-budget attempt
C_OVER = "#999999"      # an over-budget attempt (tried but not eligible)
C_NONE = "#444444"      # baseline: no indexes
C_NAIVE = "#E69F00"     # baseline: naive
C_RANDOM = "#D55E00"    # baseline: random best


def _regressed(base_pq: dict, best_pq: dict) -> int:
    """Queries slower under `best_pq` than under the no-index baseline."""
    return sum(1 for q in base_pq if best_pq.get(q, base_pq[q]) > base_pq[q])


def _collect_rows(final_gen_states: list[tuple[dict, float]]) -> list[dict]:
    """Turn the per-generation state snapshots into results rows.

    Row 0 is always the measured no-index baseline (generation 0) — a real
    history entry, not invented — so a reader of results.csv (including
    app.py's /replay) never has to guess which row is the reference line."""
    rows = []
    if final_gen_states:
        base = next(
            h for h in final_gen_states[0][0]["history"] if h.get("event") == "baseline"
        )
        rows.append({
            "generation": 0,
            "p50_total_ms": base["p50_total_ms"],
            "best_ms": base["p50_total_ms"],
            "storage_mb": base["storage_mb"],
            "n_indexes": 0,
            "failed_ddl_count": 0,
            "rejected_count": 0,
            "regressed_queries": 0,
            "tokens_prompt": 0,
            "tokens_completion": 0,
            "wall_s": 0.0,
        })
    for state, wall in final_gen_states:
        gen = state["generation"]
        hist = state["history"]
        bench = next(h for h in hist if h.get("event") == "benchmark" and h.get("generation") == gen)
        base = next(h for h in hist if h.get("event") == "baseline")
        best_entry = next(
            (h for h in hist if h.get("ddls") == state["best_config"] and "per_query_ms" in h),
            None,
        )
        base_pq = base["per_query_ms"]
        best_pq = best_entry["per_query_ms"] if best_entry else base_pq
        rows.append({
            "generation": gen,
            "p50_total_ms": bench["p50_total_ms"],
            "best_ms": state["best_ms"],
            "storage_mb": bench["storage_mb"],
            "n_indexes": len(bench["ddls"]),
            "failed_ddl_count": len(bench["failed_ddl"]),
            "rejected_count": sum(
                1 for h in hist if h.get("event") == "reject" and h.get("generation") == gen
            ),
            "regressed_queries": _regressed(base_pq, best_pq),
            "tokens_prompt": sum(
                h.get("tokens_prompt", 0) for h in hist
                if h.get("event") == "propose" and h.get("generation") == gen
            ),
            "tokens_completion": sum(
                h.get("tokens_completion", 0) for h in hist
                if h.get("event") == "propose" and h.get("generation") == gen
            ),
            "wall_s": round(wall, 1),
        })
    return rows


def run() -> tuple[list[dict], dict]:
    app = graph.build_graph()
    initial = {"workload": WORKLOAD, "storage_budget_mb": graph.STORAGE_BUDGET_MB}
    config = {"configurable": {"thread_id": "run"}, "recursion_limit": 400}

    gen_snapshots: list[tuple[dict, float]] = []
    last_gen, t_prev = 0, time.time()
    final_state = None
    with graph.search_trace(generations=graph.MAX_GENERATIONS, budget_mb=graph.STORAGE_BUDGET_MB):
        for state in app.stream(initial, config, stream_mode="values"):
            final_state = state
            gen = state.get("generation", 0)
            if gen > last_gen:  # a generation's benchmark just landed
                now = time.time()
                gen_snapshots.append((dict(state), now - t_prev))
                t_prev, last_gen = now, gen
                b = gen_snapshots[-1][0]
                print(f"  gen {gen:2d}: p50={b['last_result']['p50_total_ms']:8.0f}ms  "
                      f"best={b['best_ms']:8.0f}ms  storage={b['last_result']['storage_mb']:5.1f}MB")
    graph.flush_traces()

    return _collect_rows(gen_snapshots), final_state


def _plot(rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    baselines = json.loads(BASELINES_JSON.read_text()) if BASELINES_JSON.exists() else None
    gens = [r["generation"] for r in rows]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    # best-so-far: the headline progress line
    ax.plot(gens, [r["best_ms"] for r in rows], color=C_BEST, lw=2,
            marker="o", ms=5, label="LLM best-so-far", zorder=5)

    # each generation's attempt, split by whether it fit the budget
    valid_g = [r["generation"] for r in rows if r["storage_mb"] <= graph.STORAGE_BUDGET_MB]
    valid_p = [r["p50_total_ms"] for r in rows if r["storage_mb"] <= graph.STORAGE_BUDGET_MB]
    over_g = [r["generation"] for r in rows if r["storage_mb"] > graph.STORAGE_BUDGET_MB]
    over_p = [r["p50_total_ms"] for r in rows if r["storage_mb"] > graph.STORAGE_BUDGET_MB]
    ax.scatter(valid_g, valid_p, color=C_VALID, s=42, label="attempt (within budget)", zorder=4)
    ax.scatter(over_g, over_p, facecolors="none", edgecolors=C_OVER, s=42,
               label="attempt (over budget)", zorder=4)

    # baselines as recessive horizontal reference lines
    if baselines:
        ax.axhline(baselines["none"]["p50_total_ms"], color=C_NONE, ls="--", lw=1.3,
                   label=f"baseline none ({baselines['none']['p50_total_ms']:.0f} ms)")
        ax.axhline(baselines["naive"]["p50_total_ms"], color=C_NAIVE, ls="--", lw=1.3,
                   label=f"baseline naive ({baselines['naive']['p50_total_ms']:.0f} ms)")
        rb = baselines["random"]["best_within_budget"]
        if rb:
            ax.axhline(rb["p50_total_ms"], color=C_RANDOM, ls="--", lw=1.3,
                       label=f"baseline random-best ({rb['p50_total_ms']:.0f} ms)")

    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_yscale("log")
    ax.set_xlabel("generation")
    ax.set_ylabel("workload p50 total (ms, log scale)")
    ax.set_title("QueryForge — LLM index search vs. baselines (all measured)")
    ax.grid(True, which="both", axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, framealpha=0.9, loc="upper right")
    fig.tight_layout()
    fig.savefig(FITNESS_PNG, dpi=130, bbox_inches="tight")
    print(f"  wrote {FITNESS_PNG}")


def main() -> None:
    print(f"Full search: {graph.MAX_GENERATIONS} generations, budget "
          f"{graph.STORAGE_BUDGET_MB:.0f} MB\n")
    rows, final = run()

    RESULTS_CSV.parent.mkdir(exist_ok=True)
    with RESULTS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    HISTORY_JSON.write_text(json.dumps(final["history"], indent=2), encoding="utf-8")
    _plot(rows)

    # --- honest summary ---
    print("\n=== SEARCH RESULT ===")
    print(f"  baseline (none) : {final['baseline_ms']:.0f} ms")
    print(f"  best            : {final['best_ms']:.0f} ms  "
          f"({(1 - final['best_ms'] / final['baseline_ms']) * 100:.1f}% faster)")
    print(f"  best config     : {len(final['best_config'])} indexes")
    for d in final["best_config"]:
        print(f"      {d}")

    max_regressed = max((r["regressed_queries"] for r in rows), default=0)
    if not final["best_config"]:
        print("\n  NOTE: no configuration ever came in under budget — best is still")
        print("  the no-index baseline. That is a budget/proposal issue, NOT a broken")
        print("  oracle (real over-budget configs were far faster; see run_history.json).")
    elif max_regressed == 0:
        print("\n  WARNING: regressed_queries == 0 for EVERY generation, yet a real")
        print("  config won. Every genuine win regresses some query — this strongly")
        print("  suggests the oracle is broken. Investigate before trusting results.")
    else:
        print(f"\n  regressed_queries peaks at {max_regressed} — expected; the win "
              "is a net trade-off, not a free lunch.")

    analysis = next((h["analysis"] for h in final["history"] if h["event"] == "analyze"), None)
    if analysis:
        print("\n=== 70B analysis ===")
        print(analysis)


if __name__ == "__main__":
    main()

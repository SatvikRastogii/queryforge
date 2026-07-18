"""Stage 5: baselines — the honest yardsticks, all measured by the SAME oracle.

Three references the LLM search must be judged against:

  1. none   — no indexes. The reference line every speedup is relative to.
  2. naive  — a single-column btree on EVERY pruned candidate column. The
              "index everything" strawman: usually fast but far over budget.
  3. random — 20 randomly sampled valid configs under the storage budget, at
              the SAME benchmark budget the LLM gets (20 evaluations). This is
              the ablation that can kill the project: if the LLM's careful
              selection does not beat the best random config, the LLM added
              nothing. We report that either way and do not tune to hide it.

Writes evals/baselines.json for the Stage 6 plot and the report. Fixed seed,
so it is reproducible.
"""

import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import oracle  # noqa: E402
import specs  # noqa: E402
from graph import STORAGE_BUDGET_MB  # noqa: E402  (single source of truth)
from specs import IndexSpec, candidate_columns  # noqa: E402

RANDOM_SEED = 42
RANDOM_N = 20
RANDOM_MAX_INDEXES = 6  # each random config is 1..6 single-column indexes
OUT_PATH = Path(__file__).resolve().parent / "evals" / "baselines.json"


def _benchmark(ddls: list[str]) -> dict:
    """oracle.benchmark(), but every statement passes Control 1 (the same
    allowlist graph.py's validate_node and mcp_server.py enforce) first — no
    caller of oracle.benchmark() is exempt from the allowlist, even one that
    only ever builds DDL from trusted, catalog-derived column names. A
    rejection here means OUR OWN DDL generation is broken, so it raises loudly
    rather than silently dropping the statement."""
    rejected = [d for d in ddls if not specs.validate(d)]
    if rejected:
        raise ValueError(f"self-generated DDL failed the allowlist: {rejected}")
    return oracle.benchmark(ddls)


def _summarize(ddls: list[str], result: dict) -> dict:
    return {
        "ddls": ddls,
        "n_indexes": len(ddls),
        "p50_total_ms": result["p50_total_ms"],
        "storage_mb": result["storage_mb"],
        "within_budget": result["storage_mb"] <= STORAGE_BUDGET_MB,
        "timed_out": result["timed_out"],
        "failed_ddl": result["failed_ddl"],
    }


def run_none() -> dict:
    return _summarize([], _benchmark([]))


def run_naive() -> dict:
    ddls = sorted(
        IndexSpec(table, (col,)).to_ddl()
        for table, cols in candidate_columns().items()
        for col in cols
    )
    return _summarize(ddls, _benchmark(ddls))


def run_random() -> dict:
    """20 random single-column configs, fixed seed. Reports the best config
    that came in UNDER budget (the fair comparison to the LLM's best), plus
    every trial for the record."""
    rng = random.Random(RANDOM_SEED)
    pairs = [(t, c) for t, cols in candidate_columns().items() for c in cols]

    trials = []
    for i in range(RANDOM_N):
        k = rng.randint(1, RANDOM_MAX_INDEXES)
        picked = rng.sample(pairs, k)
        ddls = sorted({IndexSpec(t, (c,)).to_ddl() for t, c in picked})
        trials.append(_summarize(ddls, _benchmark(ddls)))
        print(f"  random {i + 1:2d}/{RANDOM_N}: "
              f"p50={trials[-1]['p50_total_ms']:8.0f}ms  "
              f"storage={trials[-1]['storage_mb']:5.1f}MB  "
              f"within_budget={trials[-1]['within_budget']}")

    within = [t for t in trials if t["within_budget"]]
    best = min(within, key=lambda t: t["p50_total_ms"]) if within else None
    return {
        "seed": RANDOM_SEED,
        "n": RANDOM_N,
        "n_within_budget": len(within),
        "best_within_budget": best,
        "trials": trials,
    }


def main() -> None:
    t0 = time.time()
    print(f"Storage budget: {STORAGE_BUDGET_MB:.1f} MB\n")

    print("[1/3] none (no indexes)...")
    none = run_none()
    print(f"  p50={none['p50_total_ms']:.0f}ms  storage={none['storage_mb']:.1f}MB\n")

    print("[2/3] naive (single-column index on every candidate)...")
    naive = run_naive()
    print(f"  {naive['n_indexes']} indexes  p50={naive['p50_total_ms']:.0f}ms  "
          f"storage={naive['storage_mb']:.1f}MB  within_budget={naive['within_budget']}\n")

    print(f"[3/3] random ({RANDOM_N} configs, seed {RANDOM_SEED})...")
    rnd = run_random()

    out = {
        "storage_budget_mb": STORAGE_BUDGET_MB,
        "none": none,
        "naive": naive,
        "random": rnd,
        "wall_s": round(time.time() - t0, 1),
    }
    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\n=== BASELINES (all measured by the same oracle) ===")
    print(f"  none                 p50={none['p50_total_ms']:9.0f} ms   "
          f"storage={none['storage_mb']:6.1f} MB")
    print(f"  naive ({naive['n_indexes']:2d} idx)        p50={naive['p50_total_ms']:9.0f} ms   "
          f"storage={naive['storage_mb']:6.1f} MB   within_budget={naive['within_budget']}")
    best = rnd["best_within_budget"]
    if best:
        print(f"  random best (<=budget) p50={best['p50_total_ms']:9.0f} ms   "
              f"storage={best['storage_mb']:6.1f} MB   ({best['n_indexes']} idx, "
              f"{rnd['n_within_budget']}/{RANDOM_N} trials fit budget)")
    else:
        print(f"  random: NONE of {RANDOM_N} trials fit the budget")
    print(f"\n  wrote {OUT_PATH}  (wall {out['wall_s'
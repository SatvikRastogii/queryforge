"""Stage 2 gate: is the oracle's timing stable enough to trust?

Runs benchmark([]) — the no-index baseline — five times and reports the
spread of p50_total_ms. If (max - min) / mean exceeds 5%, the oracle is too
noisy to draw conclusions from and we do NOT proceed: the whole project rests
on these numbers being reproducible.

Run from the repo root:  python evals/variance_check.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from oracle import benchmark  # noqa: E402

RUNS = 5
GATE_PCT = 5.0


def main() -> int:
    print(f"Running benchmark([]) x{RUNS} (no indexes)...\n")
    p50s: list[float] = []
    for i in range(RUNS):
        result = benchmark([])
        p50 = result["p50_total_ms"]
        p50s.append(p50)
        print(f"  run {i + 1}: p50_total_ms = {p50:10.3f}")

    mean = sum(p50s) / len(p50s)
    spread_pct = (max(p50s) - min(p50s)) / mean * 100.0

    print()
    print(f"  mean          : {mean:10.3f} ms")
    print(f"  min / max     : {min(p50s):10.3f} / {max(p50s):.3f} ms")
    print(f"  (max-min)/mean: {spread_pct:10.2f} %   (gate: <= {GATE_PCT:.0f}%)")
    print()

    if spread_pct <= GATE_PCT:
        print("PASS — variance within gate. Oracle is trustworthy.")
        return 0
    print("FAIL — variance exceeds gate. Do NOT proceed to the agent.")
    print("Investigate: background load, more TIMED_PASSES, or drop noisiest queries.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

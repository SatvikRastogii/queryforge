"""Self-checks for the four guardrails added around /live and /custom:
concurrency lock, run cooldown, Groq daily token budget, /custom body size cap.

Plain asserts, no framework — matches specs.py's own __main__ self-test style.
Exercises each guardrail in isolation: no Postgres connection or Groq call is
ever made (the size-cap check rejects before either would be touched).

Run directly:  python test_guardrails.py
"""

from datetime import date

from fastapi.testclient import TestClient

import app as qf_app
import graph


def _reset_run_state() -> None:
    qf_app._last_run_finished_at = 0.0
    if qf_app._BENCH_LOCK.locked():
        qf_app._BENCH_LOCK.release()


def test_concurrency_lock() -> None:
    _reset_run_state()
    with qf_app._run_slot():
        try:
            with qf_app._run_slot():
                raise AssertionError("second concurrent acquire should have raised BenchmarkBusy")
        except qf_app.BenchmarkBusy:
            pass
    print("ok: concurrency lock rejects a second concurrent run")


def test_cooldown() -> None:
    _reset_run_state()
    with qf_app._run_slot():
        pass
    try:
        with qf_app._run_slot():
            raise AssertionError("immediate re-run should have raised BenchmarkBusy (cooldown)")
    except qf_app.BenchmarkBusy:
        pass
    qf_app._last_run_finished_at -= qf_app.RUN_COOLDOWN_S + 1  # fast-forward past cooldown
    with qf_app._run_slot():
        pass
    print("ok: cooldown blocks an immediate re-run, then clears after RUN_COOLDOWN_S")


def test_token_budget() -> None:
    model = "test-model-guardrail-check"
    key = (model, date.today().isoformat())
    graph.TOKEN_BUDGET_PER_DAY[model] = 100
    try:
        graph._check_token_budget(model)  # under cap: must not raise
        graph._record_token_usage(model, 100)
        try:
            graph._check_token_budget(model)
            raise AssertionError("expected TokenBudgetExceeded once at the daily cap")
        except graph.TokenBudgetExceeded:
            pass
    finally:
        del graph.TOKEN_BUDGET_PER_DAY[model]
        graph._tokens_used_today.pop(key, None)
    print("ok: token budget guard raises once the daily cap is reached")


def test_body_size_cap() -> None:
    client = TestClient(qf_app.app)
    oversized = "SELECT 1;" * (qf_app.MAX_CUSTOM_BODY_BYTES // 5)
    resp = client.post("/custom", data={"queries": oversized})
    assert resp.status_code == 413, f"expected 413, got {resp.status_code}"
    print("ok: oversized /custom body rejected with 413 before it reaches custom_run")


if __name__ == "__main__":
    test_concurrency_lock()
    test_cooldown()
    test_token_budget()
    test_body_size_cap()
    print("\nALL GUARDRAIL CHECKS PASS")

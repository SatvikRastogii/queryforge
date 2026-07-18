"""Stage 4 support: the archive — dedup and top-k feedback for the search.

The archive is not a separate store; it is a set of pure functions over the
run's `history` list (the append-only log in ForgeState). Keeping it in state
means it checkpoints with everything else and there is no hidden global to
reason about.

Two kinds of duplicate detection:
  - EXACT: SHA1 over the sorted DDL statements. Two configurations with the
    same set of CREATE INDEX statements have the same signature.
  - NEAR: Jaccard overlap of the two DDL-statement sets. Configurations that
    differ by essentially nothing (share almost every index) are near-dups
    and not worth spending a benchmark on.

This is the ONE job ChromaDB was going to do. It is a set intersection over a
few dozen items, so it is a handful of lines of exact, deterministic Python —
no embeddings, no vector index. Top-k by fitness is a plain sort, not a search.
"""

import hashlib

NEAR_DUP_THRESHOLD = 0.85  # Jaccard >= this => "basically the same configuration"


def config_signature(ddls: list[str]) -> str:
    """Order-independent identity of a configuration: SHA1 over its sorted
    DDL statements. Each statement is deterministic (the index name embeds the
    canonical hash), so identical indexes collapse to the same signature."""
    return hashlib.sha1("\n".join(sorted(ddls)).encode()).hexdigest()


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def benchmarked(history: list[dict]) -> list[dict]:
    """The history entries that recorded a real measurement."""
    return [h for h in history if h.get("event") == "benchmark"]


def seen_exact(history: list[dict], ddls: list[str]) -> bool:
    sig = config_signature(ddls)
    return any(config_signature(h["ddls"]) == sig for h in benchmarked(history))


def near_duplicate(
    history: list[dict], ddls: list[str], threshold: float = NEAR_DUP_THRESHOLD
) -> bool:
    """True if some already-benchmarked config overlaps `ddls` by >= threshold."""
    candidate = set(ddls)
    return any(
        _jaccard(candidate, set(h["ddls"])) >= threshold for h in benchmarked(history)
    )


def top_k(history: list[dict], k: int) -> list[dict]:
    """The k fastest WITHIN-BUDGET configurations measured so far (ascending
    p50). Over-budget configs are still tracked for dedup, but they are not a
    legitimate frontier, so the LLM is only ever shown valid ones to build on."""
    valid = [h for h in benchmarked(history) if h.get("within_budget", True)]
    return sorted(valid, key=lambda h: h["p50_total_ms"])[:k]

"""Metrics of evaluation episodes (EVAL_PLAN §6-§7), computed only from the episode rows a run wrote.

Greedy success is the share of tasks solved; sampled single-rollout success is the mean over tasks of each task's mean
over its rollouts; pass^4 per task is C(c, 4) / C(n, 4) from c successes in n rollouts (never best-of-k). Episode ends
give the error taxonomy. Resume@prefix: success by condition and depth, and the off-path gap (on-path minus off-path).
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from typing import Any


def pass_k(successes: int, rollouts: int, k: int) -> float | None:
    """Probability that k rollouts drawn without replacement from the n all succeed; None when n < k."""
    return math.comb(successes, k) / math.comb(rollouts, k) if rollouts >= k else None


def _mean(xs: Sequence[float]) -> float | None:
    return math.fsum(xs) / len(xs) if xs else None


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"standard": {}, "resume_prefix": None}
    standard = [r for r in rows if r["condition"] == "standard"]
    for name in sorted({r["decoding"] for r in standard}):
        rs = [r for r in standard if r["decoding"] == name]
        by_task: dict[str, list[dict[str, Any]]] = {}
        for r in rs:
            by_task.setdefault(r["task_id"], []).append(r)
        rollouts = {len(v) for v in by_task.values()}
        if len(rollouts) != 1:
            raise ValueError(f"{name}: tasks have different numbers of rollouts {sorted(rollouts)}")
        n = rollouts.pop()
        per_task = {t: [bool(r["success"]) for r in v] for t, v in sorted(by_task.items())}
        out["standard"][name] = {
            "tasks": len(per_task),
            "rollouts_per_task": n,
            "success": _mean([sum(s) / n for s in per_task.values()]),
            "pass^4": _mean([p for s in per_task.values() if (p := pass_k(sum(s), n, 4)) is not None])
            if n >= 4
            else None,
            "ends": dict(sorted(Counter(r["end"] for r in rs).items())),
            "calls_per_episode": _mean([r["calls"] for r in rs]),
        }
    resume = [r for r in rows if r["condition"] != "standard"]
    if resume:
        cells: dict[tuple[str, int], list[bool]] = {}
        for r in resume:
            cells.setdefault((r["condition"], r["depth"]), []).append(bool(r["success"]))
        by_condition = {}
        for cond in ("empty", "on_path", "off_path"):
            xs = [x for (c, _), v in cells.items() if c == cond for x in v]
            by_condition[cond] = {"episodes": len(xs), "success": _mean([float(x) for x in xs])}
        on, off = by_condition["on_path"]["success"], by_condition["off_path"]["success"]
        out["resume_prefix"] = {
            "by_condition": by_condition,
            "by_condition_and_depth": {
                f"{c}@{d}": {"episodes": len(v), "success": _mean([float(x) for x in v])}
                for (c, d), v in sorted(cells.items())
            },  # fmt: skip
            "off_path_gap": None if on is None or off is None else on - off,
        }
    return out

"""Statistics of EVAL_PLAN §8: paired bootstrap intervals clustered by task, and McNemar's test on paired outcomes.

A task is the unit that is resampled: its seeds, rollouts and Resume@prefix starts are averaged into one value per
task first, so they never count as independent observations. Differences between configurations are paired by task
(the same test tasks, starts and decoding seeds for every configuration). Everything is deterministic given the
seed.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from typing import Any


def task_means(values: Mapping[str, Sequence[float]]) -> dict[str, float]:
    """One value per task: the mean of its units (seeds, rollouts, starts)."""
    out = {}
    for task, xs in values.items():
        if not xs:
            raise ValueError(f"task {task!r} has no values")
        out[task] = math.fsum(xs) / len(xs)
    return out


def cluster_bootstrap(per_task: Mapping[str, float], *, resamples: int = 10_000, seed: int = 0,
                      level: float = 0.95) -> dict[str, Any]:  # fmt: skip
    """The mean over tasks with a percentile interval from resampling tasks with replacement."""
    tasks = sorted(per_task)
    if not tasks:
        raise ValueError("no tasks")
    xs = [per_task[t] for t in tasks]
    n = len(xs)
    rng = random.Random(f"qet-bootstrap\x1f{seed}")
    stats = sorted(math.fsum(xs[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    lo_i = int(math.floor((1 - level) / 2 * resamples))
    hi_i = min(resamples - 1, int(math.ceil((1 + level) / 2 * resamples)) - 1)
    return {"estimate": math.fsum(xs) / n, "low": stats[lo_i], "high": stats[hi_i], "tasks": n,
            "resamples": resamples, "level": level, "seed": seed}  # fmt: skip


def paired_difference(a: Mapping[str, Sequence[float]], b: Mapping[str, Sequence[float]], **kw: Any) -> dict[str, Any]:
    """a - b over the tasks both have, each side averaged within task first, with the clustered bootstrap interval;
    "significant" when the interval excludes 0 (the proposed EVAL_PLAN §9 terms)."""
    if set(a) != set(b):
        raise ValueError(f"the two sides cover different tasks ({len(set(a) ^ set(b))} differ)")
    ma, mb = task_means(a), task_means(b)
    res = cluster_bootstrap({t: ma[t] - mb[t] for t in ma}, **kw)
    res["significant"] = res["low"] > 0 or res["high"] < 0
    return res


def mcnemar(a: Mapping[str, bool], b: Mapping[str, bool]) -> dict[str, Any]:
    """Exact McNemar test (two-sided binomial test on the discordant pairs) for paired binary outcomes per task."""
    if set(a) != set(b):
        raise ValueError("the two sides cover different tasks")
    n10 = sum(bool(a[t]) and not bool(b[t]) for t in a)
    n01 = sum(bool(b[t]) and not bool(a[t]) for t in a)
    n, k = n10 + n01, min(n10, n01)
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)
    return {"a_only": n10, "b_only": n01, "tasks": len(a), "p_value": p}

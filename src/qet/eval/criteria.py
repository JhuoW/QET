"""The M1 falsification criteria (EVAL_PLAN §9 as pre-registered at Gate A, D-030e), computed from saved predictions.

Common terms: FuncBenchGen test graphs; every run at its checkpoint at 100% of B; success averaged over the three seeds
of each task; differences with paired-bootstrap 95% intervals (10,000 resamples, clustered by task); "significant"
means the interval excludes 0; McNemar per seed alongside. Inputs are the evaluation runs' folders in results/raw/
(eval-only runs naming their checkpoint); nothing is re-run.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qet.eval.stats import mcnemar, paired_difference


@dataclass
class EvalRun:
    """One evaluation run: what it evaluated and its episodes."""

    run_id: str
    configuration: str
    backbone: str
    seed: int
    split: str
    label: str | None  # the checkpoint label (frac-1, ...); None for a base model
    greedy: dict[str, bool] = field(default_factory=dict)  # task -> greedy success
    samples: dict[str, list[bool]] = field(default_factory=dict)  # task -> sampled successes
    resume: list[dict[str, Any]] = field(default_factory=list)  # Resume@prefix rows
    variant: str = ""  # "" for the main matrix; n1250, n2500, fi, stateful (D-033); "base" for a base model
    episodes: dict[str, int] = field(default_factory=dict)  # prediction file -> rows


def variant_of(training_run_id: str | None) -> str:
    """M1-<tag>-<backbone>[-<variant>]-s<seed>-<stamp> -> the variant ("" for the main matrix; k2/k4/k8 for a4)."""
    if training_run_id is None:
        return "base"
    parts = training_run_id.split("-")
    seed_at = max(i for i, p in enumerate(parts) if p.startswith("s") and p[1:].isdigit())
    variant = "-".join(parts[3:seed_at])
    if parts[1].startswith("a4k"):  # a4 with K = 2, 4, 8 share configuration a4 and seed 0: K tells them apart
        variant = "-".join(x for x in (parts[1][2:], variant) if x)
    return variant


def load_runs(paths: Iterable[Path]) -> list[EvalRun]:
    runs = []
    for folder in paths:
        m = json.loads((folder / "manifest.json").read_text())
        if m.get("status") != "completed":
            continue
        data = json.loads((folder / "data" / "data_manifest.json").read_text())
        spec = data["spec"]
        split = (spec.get("eval") or {}).get("split") or spec.get("env")
        ck = m.get("evaluated_checkpoint")
        run = EvalRun(m["run_id"], m["configuration"]["id"], m["backbone"]["id"], m["seed"], split,
                      ck["label"] if ck else None, variant=variant_of(ck["run_id"] if ck else None))  # fmt: skip
        for f in sorted((folder / "predictions").glob("*.jsonl")):
            lines = f.read_text(encoding="utf-8").splitlines()
            run.episodes[f.stem] = len(lines)
            for line in lines:
                r = json.loads(line)
                if r["condition"] != "standard":
                    run.resume.append(r)
                elif r["decoding"] == "greedy":
                    run.greedy[r["task_id"]] = bool(r["success"])
                else:
                    run.samples.setdefault(r["task_id"], []).append(bool(r["success"]))
        runs.append(run)
    return runs


def by_seed(runs: Iterable[EvalRun], configuration: str, backbone: str, split: str = "test",
            label: str | None = "frac-1", variant: str = "") -> dict[int, EvalRun]:  # fmt: skip
    out = {}
    for r in runs:
        if (r.configuration, r.backbone, r.split, r.label, r.variant) == (configuration, backbone, split, label,
                                                                          variant):  # fmt: skip
            if r.seed in out:
                raise ValueError(f"two evaluation runs of configuration {configuration}, seed {r.seed}: {r.run_id}")
            out[r.seed] = r
    return out


def seed_means(runs: Mapping[int, EvalRun], tasks: Iterable[str]) -> dict[str, list[float]]:
    """Per task, the greedy success of each seed (the bootstrap averages them within the task)."""
    return {t: [float(r.greedy[t]) for _, r in sorted(runs.items())] for t in tasks}


def difference(runs: list[EvalRun], a: str, b: str, backbone: str, tasks: set[str], **kw: Any) -> dict[str, Any]:
    ra, rb = by_seed(runs, a, backbone), by_seed(runs, b, backbone)
    if sorted(ra) != sorted(rb) or not ra:
        raise ValueError(f"configurations {a} and {b} need the same seeds (have {sorted(ra)} and {sorted(rb)})")
    shared = sorted(t for t in tasks if all(t in r.greedy for r in (*ra.values(), *rb.values())))
    res = paired_difference(seed_means(ra, shared), seed_means(rb, shared), **kw)
    res["mcnemar_per_seed"] = {s: mcnemar({t: ra[s].greedy[t] for t in shared}, {t: rb[s].greedy[t] for t in shared})
                               for s in sorted(ra)}  # fmt: skip
    res["comparison"] = f"{a} - {b}"
    return res


def criterion_1(runs: list[EvalRun], widths: Mapping[str, int], backbone: str, **kw: Any) -> dict[str, Any]:
    """Greedy success on graphs of measured width >= 3: if the interval of (6 - 3) or of (6 - 4) contains 0 or lies
    below it, the objective claim fails."""
    pool = {t for t, w in widths.items() if w >= 3}
    diffs = [difference(runs, "6", c, backbone, pool, **kw) for c in ("3", "4")]
    fails = any(d["low"] <= 0 for d in diffs)
    return {"differences": diffs, "pool": len(pool), "claim": "fails" if fails else "holds"}


def criterion_2(runs: list[EvalRun], tasks: set[str], backbone: str, **kw: Any) -> dict[str, Any]:
    """Greedy success on all test graphs: if (2 - 1) is significant and positive but the interval of (6 - 2) contains
    0 or lies below it, the gain belongs to the interface, not the objective."""
    d21, d62 = difference(runs, "2", "1", backbone, tasks, **kw), difference(runs, "6", "2", backbone, tasks, **kw)
    met = d21["low"] > 0 and d62["low"] <= 0
    return {
        "differences": [d21, d62],
        "condition_met": met,
        "reading": "the gain belongs to the interface, not the objective" if met else "the condition is not met",
    }


def gaps(run: EvalRun) -> dict[str, float]:
    """Per task with off-path starts: on-path minus off-path success (each averaged over the task's starts)."""
    on: dict[str, list[float]] = {}
    off: dict[str, list[float]] = {}
    for r in run.resume:
        if r.get("history", "full") != "full":
            continue
        target = on if r["condition"] == "on_path" else off if r["condition"] == "off_path" else None
        if target is not None:
            target.setdefault(r["task_id"], []).append(float(r["success"]))
    return {t: sum(on[t]) / len(on[t]) - sum(off[t]) / len(off[t]) for t in off if t in on}


def criterion_3(runs: list[EvalRun], backbone: str, **kw: Any) -> dict[str, Any]:
    """Resume@prefix greedy success on the test tasks that have off-path starts, with the off-path gap = on-path minus
    off-path success at the frozen starts: if the interval of (gap 3 - gap 6) or of (gap 5 - gap 6) contains 0 or lies
    below it, claim C4 fails."""
    per = {c: {s: gaps(r) for s, r in by_seed(runs, c, backbone).items()} for c in ("3", "5", "6")}
    out = []
    for c in ("3", "5"):
        if sorted(per[c]) != sorted(per["6"]) or not per[c]:
            raise ValueError(f"configurations {c} and 6 need the same seeds for Resume@prefix")
        seeds = sorted(per[c])
        tasks = sorted(set.intersection(*(set(per[x][s]) for x in (c, "6") for s in seeds)))
        a = {t: [per[c][s][t] for s in seeds] for t in tasks}
        b = {t: [per["6"][s][t] for s in seeds] for t in tasks}
        res = paired_difference(a, b, **kw)
        res["comparison"] = f"gap {c} - gap 6"
        out.append(res)
    fails = any(d["low"] <= 0 for d in out)
    return {"differences": out, "claim": "fails" if fails else "holds"}

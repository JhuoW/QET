"""The M1 training matrix's billed GPU-hours, projected from the runs themselves (D-030a guard, D-031c):

    python scripts/m1_projection.py

For every M1.6 training run (configs/m1/train-*.yaml) the latest run folder gives the measured step-time rate (scored +
gradient tokens per logged step-second) and the progress (the training log, or a newer budget-fraction checkpoint:
the first two runs buffer their logs); PBS gives the billed walltime so far (`qstat -xf`, resources_used.walltime).
A run's projection is its billed time so far plus (B - tokens used) / rate plus 60 s to save and close; a finished run
counts its billed time. The M1.5 warm-up runs (the batch approval covers them) and their failed attempts are added at
their billed time, the warm-up length measurement included. The total is compared with the Gate A projection read
from docs/reports/gate-A.md (the 125M row) and with the 110% guard. Engineering projection, not a result. Writes
nothing.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "results" / "raw"


def billed_hours(job: str) -> float | None:
    out = subprocess.run(["qstat", "-xf", job], capture_output=True, text=True).stdout
    m = re.search(r"resources_used\.walltime = (\d+):(\d+):(\d+)", out)
    return None if m is None else int(m[1]) + int(m[2]) / 60 + int(m[3]) / 3600


def latest_tokens(run: Path, log_rows: list[dict]) -> int:
    used = max((r["tokens_used"] for r in log_rows if r["phase"] == "train"), default=0)
    ckpts = REPO / "checkpoints" / run.name
    for meta in ckpts.glob("*/qet_checkpoint.json") if ckpts.exists() else []:
        used = max(used, json.loads(meta.read_text()).get("tokens_used") or 0)
    return used


def main() -> None:
    gate = (REPO / "docs" / "reports" / "gate-A.md").read_text(encoding="utf-8")
    projected_gate_a = float(re.search(r"^\| 125,000,000 \|.*\| ([\d,]+) \|$", gate, re.M)[1].replace(",", ""))
    rows, total = [], 0.0
    for cfg in sorted((REPO / "configs" / "m1").glob("train-*.yaml")):
        name = cfg.stem.removeprefix("train-")  # e.g. c6-qwen7b-s0 -> runs M1-c6-qwen7b-s0-<stamp>
        runs = sorted(RAW.glob(f"M1-{name}-2026*"))
        if not runs:
            rows.append((name, "not started", None))
            continue
        run = runs[-1]
        man = json.loads((run / "manifest.json").read_text())
        budget = man["budgets"]["train"]["token_budget"]
        log = [json.loads(x) for x in (run / "train_log.jsonl").read_text().splitlines() if x.strip()]
        train = [r for r in log if r["phase"] == "train"]
        tok = sum(r["scored"]["tokens"] + r["gradient"]["tokens"] for r in train)
        sec = sum(sum(r["seconds"].values()) for r in train)
        billed = billed_hours(man["hardware"]["scheduler_job"]) or 0.0
        if man["status"] == "completed":
            hours = billed
        elif sec:
            hours = billed + (budget - latest_tokens(run, log)) / (tok / sec) / 3600 + 60 / 3600
        else:
            rows.append((name, "no logged step yet", None))
            continue
        total += hours
        rows.append((name, f"{man['status']}, {tok / sec if sec else 0:,.0f} tokens/s", hours))
    warm = 0.0
    for man_path in sorted(RAW.glob("M1-warmup-*/manifest.json")):  # the length measurement and failed attempts too
        man = json.loads(man_path.read_text())
        warm += billed_hours(man["hardware"]["scheduler_job"]) or man.get("gpu_hours") or 0.0
    for name, state, hours in rows:
        print(f"{name:18s} {state:32s} {'' if hours is None else f'{hours:5.2f} h'}")
    missing = sum(h is None for _, _, h in rows)
    grand = total + warm
    share = grand / projected_gate_a
    print(f"\ntraining runs {total:.1f} h ({len(rows) - missing} projected, {missing} without a rate) + warm-ups "
          f"{warm:.2f} h = {grand:.1f} billed GPU-hours; Gate A projection {projected_gate_a:.0f}: {share:.1%} "
          f"(guard 110% = {1.1 * projected_gate_a:.1f})")  # fmt: skip


if __name__ == "__main__":
    sys.path.insert(0, str(REPO / "src"))
    main()

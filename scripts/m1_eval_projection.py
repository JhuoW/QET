"""The M1.7 evaluation's billed GPU-hours, projected from the batches themselves (D-032 guard):

    python scripts/m1_eval_projection.py

Reads the latest chains file (logs/pbs/chains-*.json) for the evaluation batches and their PBS jobs. A finished batch
counts its billed walltime (`qstat -xf`, resources_used.walltime). Every other evaluation is projected from the
completed evaluation runs (results/raw/M1-eval-*/manifest.json, gpu_hours): the mean of its kind (full, frac0.25,
frac0.5, frac0.75, renamed, heldout, toolhop, nohist), Qwen and Llama alike until a Llama run has completed (the
means are printed); a kind with no completed run yet counts the full mean, which overstates nohist. The stateful
evaluations (no Resume@prefix, 120 test tasks against 2,240) count the full mean scaled by 120 / 2,240, which
overstates them. A running batch counts its completed runs' hours, the mean for the rest, and
its billed time in place of that sum when larger. Each finished batch's billed time above the sum of its runs is the
batch overhead (engine loads), added at its mean to every batch not yet finished.

The D-032 guard covers (a)-(b): the core, ablations and transfer sets, not ToolHop or the D-033 extras (variants
n1250, n224, fi, stateful), which are reported in a separate line. Guard: 110% of the likely 145 GPU-hours.
Engineering projection, not a result. Writes nothing.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "results" / "raw"
LIKELY = 145.0
EXTRAS = ("-n1250-", "-n224-", "-fi-", "-stateful-")
STATEFUL_SCALE = 120 / 2240
KINDS = r"full|frac0\.25|frac0\.5|frac0\.75|renamed|heldout|toolhop|nohist"
NAME = re.compile(rf"eval-(?P<run>.+)-(?P<kind>{KINDS})-s(?P<seed>\d+)")


def qstat(job: str) -> tuple[str | None, float | None]:
    out = subprocess.run(["qstat", "-xf", job], capture_output=True, text=True).stdout
    st = re.search(r"job_state = (\w)", out)
    m = re.search(r"resources_used\.walltime = (\d+):(\d+):(\d+)", out)
    return (st[1] if st else None), (None if m is None else int(m[1]) + int(m[2]) / 60 + int(m[3]) / 3600)


def main() -> None:
    chains = json.loads(sorted((REPO / "logs" / "pbs").glob("chains-*.json"))[-1].read_text())
    done: dict[str, float] = {}  # config name -> gpu_hours of its latest completed run
    for man_path in sorted(RAW.glob("M1-eval-*/manifest.json")):
        man = json.loads(man_path.read_text())
        if man["status"] == "completed":
            done[Path(man["config"]["path"]).stem] = man["gpu_hours"]
    by_kind = defaultdict(list)
    for name, hours in done.items():
        m = NAME.fullmatch(name)
        if m and not any(e in name for e in EXTRAS):
            by_kind[m["kind"]].append(hours)
    mean = {k: sum(v) / len(v) for k, v in sorted(by_kind.items())}

    def estimate(name: str) -> float:
        kind = NAME.fullmatch(name)["kind"]
        return mean["full"] * STATEFUL_SCALE if "-stateful-" in name else mean.get(kind, mean["full"])

    batches, overheads = [], []
    for job in chains["jobs"]:
        names = [Path(x).stem for x in (REPO / job["path"]).read_text().split() if x.strip()]
        state, billed = qstat(job["job"])
        ran = sum(done[n] for n in names if n in done)
        rest = sum(estimate(n) for n in names if n not in done)
        if state == "F":
            overheads.append((billed or 0.0) - ran)
        batches.append((job, names, state, billed, ran, rest))
    over = sum(overheads) / len(overheads) if overheads else 0.0
    totals = {"core": 0.0, "extras": 0.0}
    for job, names, state, billed, ran, rest in batches:
        hours = billed if state == "F" else max(billed or 0.0, ran + rest) + over
        part = "extras" if any(e in job["path"] for e in EXTRAS) else "core"
        core_share = sum(1 for n in names if NAME.fullmatch(n)["kind"] != "toolhop")
        if part == "core" and core_share < len(names):  # a batch's ToolHop run goes to the extras line
            th = sum(done.get(n, estimate(n)) for n in names if NAME.fullmatch(n)["kind"] == "toolhop")
            totals["extras"] += th
            hours -= th
        totals[part] += hours
        print(f"{job['job']:11s} {Path(job['path']).stem:24s} {state or '?'} {len(names)} runs {hours:6.2f} h")
    means = ", ".join(f"{k} {v:.2f} ({len(by_kind[k])})" for k, v in mean.items())
    print(f"\nmeans of completed runs (GPU-h): {means}")
    print(f"batch overhead {over:.2f} h ({len(overheads)} finished batches)")
    print(f"(a)-(b) core {totals['core']:.1f} GPU-h against likely {LIKELY:.0f}: {totals['core'] / LIKELY:.1%} "
          f"(guard 110% = {1.1 * LIKELY:.1f}); ToolHop and D-033 extras {totals['extras']:.1f} GPU-h; "
          f"all {sum(totals.values()):.1f}")  # fmt: skip


if __name__ == "__main__":
    sys.path.insert(0, str(REPO / "src"))
    main()

"""The M1.7 evaluation budget, from a measured vLLM check on a trained checkpoint (D-030f) and the frozen manifests:

    python scripts/eval_budget.py [--check results/raw/M1-vllm-check-<stamp>]
                                  [--compile-run results/raw/M1-compile-<stamp>]

The check gives the vLLM time per generation step on dev tasks spread over the grid (greedy + 8 samples). An episode's
steps are bounded three ways: the check's own mean (its checkpoint ends many episodes early with invalid actions, a
lower bound), the demonstration's calls plus finish (an agent that solves the task), and the step budget (2 x calls,
plus finish). Resume@prefix episodes start after the prefix, so their steps are counted from the frozen starts.
Prompts grow along an episode, so longer episodes cost more per step than the check's mean: the high column is still
not an upper bound. Engine startup (about 90 s per evaluation run) is added once per run. Writes nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STARTUP_S = 90.0


def main(argv: list[str] | None = None) -> None:
    from qet.core.compile import read_records
    from qet.eval.resume_prefix import starts

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--check", type=Path, help="default: the latest completed vLLM check with a training checkpoint"
    )
    parser.add_argument("--compile-run", type=Path, default=Path("results/raw/M1-compile-20261018T023058Z"))
    args = parser.parse_args(argv)
    if args.check is None:
        cands = []
        for m in sorted((REPO / "results" / "raw").glob("M1-vllm-check-*/manifest.json")):
            man = json.loads(m.read_text())
            if man.get("status") == "completed" and "frac-" in str((man.get("adapter") or {}).get("path")):
                cands.append(m.parent)
        args.check = cands[-1]
    check = args.check if args.check.is_absolute() else REPO / args.check
    tp = json.loads((check / "summary.json").read_text())["throughput"]
    per_step = tp["seconds"] / tp["steps"]
    mean_steps = tp["steps"] / tp["episodes"]
    run = args.compile_run if args.compile_run.is_absolute() else REPO / args.compile_run
    tests = read_records(run / "test" / "records.jsonl")
    held = read_records(run / "heldout_larger" / "records.jsonl")

    def episode_steps(recs):  # (low, likely, high) steps per episode from the empty state
        n = [r.n for r in recs]
        return mean_steps, sum(x + 1 for x in n) / len(n), sum(2 * x + 1 for x in n) / len(n)

    def resume_steps(recs, mode):  # total (likely, high) steps of the starts after their prefixes, per task
        likely = high = 0
        for r in recs:
            for s in starts(r, 0, per_task=mode):
                left = r.n - s.depth
                likely, high = likely + left + 1, high + 2 * left + 1
        return likely / len(recs), high / len(recs)

    def hours(episodes, steps):
        return [episodes * x * per_step / 3600 for x in steps]

    t_low, t_mid, t_high = episode_steps(tests)
    h_low, h_mid, h_high = episode_steps(held)
    nt, nh = len(tests), len(held)
    rows = []
    full = hours(nt * 9, (t_low, t_mid, t_high))  # greedy + 8 samples (EVAL_PLAN §7)
    rows.append(("A", "test, greedy + 8 samples, the 100% checkpoint of 33 runs and 2 base models", 35, full))
    greedy = hours(nt, (t_low, t_mid, t_high))
    rows.append(("B", "curves: test, greedy, the 25/50/75% checkpoints of 33 runs", 99, greedy))
    for mode in ("one_depth", "all_depths"):
        likely, high = resume_steps(tests, mode)
        low = likely * t_low / t_mid
        rows.append(("C", f"Resume@prefix ({mode}), greedy, configurations 1, 2, 3, 5, 6 x 3 seeds (Qwen)", 15,
                     hours(nt, (low, likely, high))))  # fmt: skip
    renamed, heldout = hours(nt, (t_low, t_mid, t_high)), hours(nh, (h_low, h_mid, h_high))
    ren_held = [a + b for a, b in zip(renamed, heldout, strict=True)]
    rows.append(("D", "renamed + held-out larger, greedy, configurations 1-6 x 3 seeds (Qwen)", 18, ren_held))
    print(f"vLLM check {check.name}: {per_step:.4f} s per step, {mean_steps:.2f} steps per episode "
          f"(checkpoint {json.loads((check / 'manifest.json').read_text())['adapter']['path']})")  # fmt: skip
    print(f"test: {nt} tasks, demonstration {t_mid:.2f} steps, step budget {t_high:.2f}; held-out larger: {nh} tasks, "
          f"{h_mid:.2f} / {h_high:.2f} steps\n")  # fmt: skip
    print("| part | evaluation | runs | GPU-hours per run (low / likely / high) | total (low / likely / high) |")
    print("| --- | --- | --- | --- | --- |")
    for part, what, n, (lo, mid, hi) in rows:
        s = STARTUP_S / 3600
        print(f"| {part} | {what} | {n} | {lo + s:.2f} / {mid + s:.2f} / {hi + s:.2f} | "
              f"{n * (lo + s):.0f} / {n * (mid + s):.0f} / {n * (hi + s):.0f} |")  # fmt: skip


if __name__ == "__main__":
    sys.path.insert(0, str(REPO / "src"))
    main()

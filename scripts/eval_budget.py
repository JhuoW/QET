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
    rows_tp = [json.loads(x) for x in (check / "throughput.jsonl").read_text().splitlines() if x.strip()]
    mixed_steps = tp["steps"] / tp["episodes"]  # greedy + 8 samples, as the check ran
    greedy_rows = [r for r in rows_tp if r["decoding"] == "greedy"]
    greedy_steps = sum(len(r["steps"]) for r in greedy_rows) / len(greedy_rows)
    run = args.compile_run if args.compile_run.is_absolute() else REPO / args.compile_run
    tests = read_records(run / "test" / "records.jsonl")
    held = read_records(run / "heldout_larger" / "records.jsonl")
    demo_mean = sum(r.n + 1 for r in tests) / len(tests)

    def episode_steps(recs, measured):  # (the check's mean, demonstration, step budget) steps per episode
        n = [r.n for r in recs]
        demo = sum(x + 1 for x in n) / len(n)
        return measured * demo / demo_mean, demo, sum(2 * x + 1 for x in n) / len(n)  # measured, scaled to the set

    def resume_steps(recs, mode):  # per task: the steps left after each start's prefix (the agent's own budget)
        demo = cap = 0
        for r in recs:
            for s in starts(r, 0, per_task=mode):
                demo += r.n - s.depth + 1  # the remaining calls plus finish
                cap += 2 * r.n - s.depth + 1  # step_budget() - prefix calls, plus finish (agent.py)
        demo, cap = demo / len(recs), cap / len(recs)
        return demo * greedy_steps / demo_mean, demo, cap

    def hours(episodes, steps):
        return [episodes * x * per_step / 3600 for x in steps]

    nt, nh = len(tests), len(held)
    rows = []
    rows.append(("A", "test, greedy + 8 samples, the 100% checkpoint of 33 runs and 2 base models", 35,
                 hours(nt * 9, episode_steps(tests, mixed_steps))))  # fmt: skip
    rows.append(("B", "curves: test, greedy only (a choice EVAL_PLAN §8 leaves open), the 25/50/75% checkpoints of "
                 "33 runs", 99, hours(nt, episode_steps(tests, greedy_steps))))  # fmt: skip
    for mode in ("one_depth", "all_depths"):
        rows.append(("C", f"Resume@prefix ({mode}), greedy, all 2,240 test tasks, configurations 1, 2, 3, 5, 6 x 3 "
                     "seeds (Qwen)", 15, hours(nt, resume_steps(tests, mode))))  # fmt: skip
    rows.append(("C'", "Resume@prefix (all_depths, as chosen: D-032b), no history (EVAL_PLAN §6.5), configuration 1 x "
                 "3 seeds", 3, hours(nt, resume_steps(tests, "all_depths"))))  # fmt: skip
    ren = hours(nt, episode_steps(tests, greedy_steps))
    hel = hours(nh, episode_steps(held, greedy_steps))
    rows.append(("D", "renamed + held-out larger, greedy, configurations 1-6 x 3 seeds (Qwen)", 18,
                 [a + b for a, b in zip(ren, hel, strict=True)]))  # fmt: skip
    adapter = json.loads((check / "manifest.json").read_text())["adapter"]["path"]
    print(f"vLLM check {check.name} (checkpoint {adapter}): {per_step:.4f} s per step; {mixed_steps:.2f} steps per "
          f"episode over greedy + 8 samples, {greedy_steps:.2f} over greedy episodes")  # fmt: skip
    print(f"test: {nt} tasks, demonstration {demo_mean:.2f} steps; held-out larger: {nh} tasks\n")
    print("| part | evaluation | runs | GPU-hours per run (check's mean / demonstration / step budget) | total |")
    print("| --- | --- | --- | --- | --- |")
    s = STARTUP_S / 3600
    for part, what, n, cols in rows:
        per = " / ".join(f"{c + s:.2f}" for c in cols)
        tot = " / ".join(f"{n * (c + s):.0f}" for c in cols)
        print(f"| {part} | {what} | {n} | {per} | {tot} |")
    print("""
The step-budget column is not an upper bound: the time per step was measured at the check's mean prompt length, and
prompts grow along an episode (Resume@prefix episodes start with long prompts). Not measured: the configuration-1 and
a1 renderers, Llama, the base models without an adapter. Not included at all: ToolHop (adapter and data), the injected
false-independence ablation, the stateful extension and the data curves (each needs training runs), and any dev
evaluation (none is planned: no checkpoint is selected).""")


if __name__ == "__main__":
    sys.path.insert(0, str(REPO / "src"))
    main()

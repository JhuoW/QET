"""Gate reports from results/raw/: python scripts/gate_report.py --gate A [--compile-run DIR] [--stats-run DIR]
[--pilot-run DIR] [--out docs/reports/gate-A.md]

Gate A (MILESTONES M1): compile coverage and fallback, width distribution, incomparable-pair share, states per graph,
true-DAG agreement, invariance results, tokens per graph, throughput and the GPU-hour projection, the power analysis
behind a proposed test size per cell, and proposed wording for EVAL_PLAN §9. Every number is read from the named run
folders (their sha256 values are listed at the end); proposals are marked as such. The report ends at a STOP: the human
approves the budget, the test manifest and the pre-registration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "results" / "raw"


def latest(prefix: str) -> Path:
    runs = sorted(p for p in RAW.glob(f"{prefix}*") if (p / "manifest.json").exists() and "limit" not in p.name)
    done = [p for p in runs if json.loads((p / "manifest.json").read_text()).get("status") == "completed"]
    if not done:
        raise SystemExit(f"no completed run named {prefix}* in results/raw")
    return done[-1]


class Sources:
    def __init__(self) -> None:
        self.files: list[tuple[str, str]] = []

    def json(self, path: Path):
        self.files.append((str(path.relative_to(REPO)), hashlib.sha256(path.read_bytes()).hexdigest()))
        return json.loads(path.read_text())

    def lines(self, path: Path):
        self.files.append((str(path.relative_to(REPO)), hashlib.sha256(path.read_bytes()).hexdigest()))
        return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def pct(x: float | None, digits: int = 1) -> str:
    return "–" if x is None else f"{100 * x:.{digits}f}%"


def num(x: float | None, digits: int = 3) -> str:
    if x is None:
        return "–"
    return f"{x:,.{digits}f}" if isinstance(x, float) else f"{x:,}"


def mcnemar_pairs(delta: float, psi: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Paired tasks needed to detect a success-rate difference delta with discordant share psi (Connor 1987)."""
    z_a, z_b = NormalDist().inv_cdf(1 - alpha / 2), NormalDist().inv_cdf(power)
    return math.ceil((z_a * math.sqrt(psi) + z_b * math.sqrt(psi - delta**2)) ** 2 / delta**2)


def gate_a(compile_run: Path, stats_run: Path, pilot_run: Path | None, src: Sources) -> str:
    cm = src.json(compile_run / "manifest.json")
    reports = {s: src.json(compile_run / s / "report.json") for s in cm["splits"]}
    stats_manifest = src.json(stats_run / "manifest.json")
    stats = src.json(stats_run / "stats.json")
    out = ["# Gate A report (M1: FuncBenchGen)", ""]
    out += [f"Sources: compile run `{compile_run.name}` (commit {cm['git']['commit'][:7]}"
            f"{', dirty' if cm['git']['dirty'] else ''}), pilot statistics `{stats_run.name}` (first "
            f"{stats['graphs']} training graphs, tokenizer {stats_manifest['tokenizer']['backbone']})"
            + (f", pilot run `{pilot_run.name}`." if pilot_run else "; no pilot GPU run yet."), ""]  # fmt: skip

    out += ["## 1. Compile coverage and fallback", "",
            "| split | tasks | demos verified | chain fallbacks | Compile missed a true constraint | caught by the "
            "replay audit | unsafe orders left |", "| --- | --- | --- | --- | --- | --- | --- |"]  # fmt: skip
    for s, r in reports.items():
        a = r["audit"]
        out.append(f"| {s} | {num(r['compile']['tasks_compiled'])} | {num(r['compile']['tasks_compiled'])} | "
                   f"{num(a['replay_fallbacks'])} ({pct(r['compile']['fallback_share'])}) | "
                   f"{num(a['raw_missed_a_true_constraint'])} | {num(a['missed_and_caught_by_replay'])} | "
                   f"{num(a['final_with_unsafe_orders'])} |")  # fmt: skip
    out += ["", "Every demonstration passed the environment's checker before Compile (otherwise the compile run "
            "stops). Compile received no task constraints (D-020a). The replay audit ran "
            f"{cm['settings']['audit_orders']} random orders plus one hoisted order per node (D-029).", ""]  # fmt: skip

    out += ["## 2. Agreement with the generator's true DAGs", "",
            "| split | precision (raw → final) | recall (raw → final) | recall by reachability (raw → final) | "
            "unsafe-order rate (raw → final) |", "| --- | --- | --- | --- | --- |"]  # fmt: skip
    for s, r in reports.items():
        raw, fin = r["audit"]["raw"], r["audit"]["final"]
        cells = [f"{num(raw[k])} → {num(fin[k])}" for k in ("precision", "recall", "recall_by_reachability",
                                                             "unsafe_order_rate")]  # fmt: skip
        out.append(f"| {s} | " + " | ".join(cells) + " |")
    out += ["", "\"Final\" is after the chain fallback; FuncBenchGen precision below 1 comes from the fallbacks, the "
            "stateful extension's from decision edges around payments that truly commute.", ""]  # fmt: skip

    out += ["## 3. Structure: width, incomparable pairs, states", ""]
    for s in ("train", "test"):
        if s in reports:
            c = reports[s]["compile"]
            out.append(f"- {s}: width distribution {c['width']}; incomparable-pair share "
                       f"{pct(c['incomparable_pair_share'])}; states per graph {c['states_per_graph']}")  # fmt: skip
    out += [f"- pilot sample ({stats['graphs']} training graphs): width bins {stats['width_bins']}; "
            f"{stats['states']['above_s_max']} graphs above S_max = {stats['s_max']} (trained on a sub-lattice of "
            f"K = {stats['k']} orders), mean exact lattice size {num(stats['states']['mean_exact'], 1)}",
            ""]  # fmt: skip

    inv = stats["invariance"]
    out += ["## 4. Invariance", "",
            f"- On the pilot sample: {num(inv['prefixes_checked'])} prefixes of {inv['orders_per_graph']} random legal "
            f"orders per graph rendered against the demo-order history of the same set: {num(inv['failures'])} "
            "differences (SPEC §5.6 a).",
            "- Unit tests (reported as tests, not results): tests/core/test_reduce_render.py (300 random graphs × 20 "
            "orders × every prefix), tests/infer/test_agent.py (the agent following legal orders sees the training "
            "prompt of every state), tests/core/test_fuzz.py.", ""]  # fmt: skip

    out += ["## 5. Tokens per graph (one epoch, scored + gradient)", "",
            "| configuration | scored | gradient | largest graph |", "| --- | --- | --- | --- |"]  # fmt: skip
    for c, t in stats["tokens_per_graph"].items():
        out.append(f"| {c} | {num(t['scored'], 0)} | {num(t['gradient'], 0)} | {num(t['max'])} |")
    out += ["", "By width bin (configurations 2 and 6): "
            + "; ".join(f"{b}: {num(v['2'], 0)} / {num(v['6'], 0)}" for b, v in
                        stats["tokens_per_graph_by_width_bin"].items()), ""]  # fmt: skip

    out += ["## 6. Throughput and GPU-hour projection", ""]
    epoch6 = stats["tokens_per_graph"]["6"]["scored"] + stats["tokens_per_graph"]["6"]["gradient"]
    if pilot_run is not None:
        pm = src.json(pilot_run / "manifest.json")
        rows = [r for r in src.lines(pilot_run / "train_log.jsonl") if r["phase"] == "train"]
        tok = sum(r["scored"]["tokens"] + r["gradient"]["tokens"] for r in rows)
        sec = sum(sum(r["seconds"].values()) for r in rows)
        tps = tok / sec if sec else None
        out += [f"- Pilot `{pilot_run.name}`: configuration {pm['configuration']['id']} on {pm['backbone']['id']}, "
                f"{len(rows)} training steps, {num(tok)} tokens in {num(sec, 0)} s of step time: {num(tps, 0)} "
                f"tokens/s; the run used {num(pm.get('gpu_hours'), 3)} GPU-hours "
                f"({pm.get('gpu_hours_by_phase')}).", ""]  # fmt: skip
        if tps:
            out += ["Projection (proposal; the token budget per run is the human's decision): every configuration gets "
                    "the same budget B (EVAL_PLAN §8). With B = E epochs of configuration 6 over 5,000 training graphs "
                    f"(about {num(epoch6 * 5000, 0)} tokens per epoch from the pilot sample's mean):", "",
                    "| E | B (tokens) | GPU-hours per run | M1 run matrix (≈ 33 runs) |",
                    "| --- | --- | --- | --- |"]  # fmt: skip
            for e in (0.5, 1, 2):
                b = e * epoch6 * 5000
                per = b / tps / 3600
                out.append(f"| {e:g} | {num(b, 0)} | {num(per, 1)} | {num(33 * per, 0)} |")
            out += ["", "Training only, assuming the Llama-3.1-8B runs match the measured Qwen2.5-7B throughput; "
                    "evaluation comes on top (decoding is far slower per token than training; a vLLM path is "
                    "planned for it, D-026b).", ""]  # fmt: skip
    else:
        out += ["- No pilot GPU run yet: the projection needs its measured throughput.", ""]

    out += ["## 7. Power analysis and proposed test size per cell", "",
            "Paired tasks needed to detect a success-rate difference Δ between two configurations with McNemar's test "
            "(two-sided α = 0.05, power 0.8), for a share ψ of discordant tasks:", "",
            "| Δ \\ ψ | 0.10 | 0.20 | 0.30 |", "| --- | --- | --- | --- |"]  # fmt: skip
    for d in (0.05, 0.10, 0.15):
        out.append(f"| {d:.2f} | " + " | ".join(num(mcnemar_pairs(d, p)) for p in (0.10, 0.20, 0.30)) + " |")
    test = reports.get("test", {}).get("compile", {})
    wide = sum(v for k, v in test.get("width", {}).items() if int(k) >= 3)
    share = wide / test["tasks_compiled"] if test else None
    out += ["", f"Criterion 1 pools graphs of width ≥ 3: {pct(share)} of the provisional test graphs "
            "(224 cells × 20). Proposal: keep 20 graphs per cell (4,480 test tasks) if the human accepts detecting "
            "Δ ≈ 0.10 at ψ ≈ 0.2 on the width ≥ 3 pool; the table gives the trade-off.", ""]  # fmt: skip

    out += ["## 8. Proposed wording for EVAL_PLAN §9 (for the human to accept, change or reject)", "",
            "1. On FuncBenchGen test graphs of measured width ≥ 3, at matched compute and the 100% budget checkpoint "
            "selected on dev, if the greedy success of configuration 3 or 4 is within the paired-bootstrap 95% "
            "interval of configuration 6's (clustered by task), the objective claim fails.",
            "2. If configuration 2 beats 1 significantly (McNemar, α = 0.05) but configuration 6 does not beat 2, the "
            "gain belongs to the interface, not the objective.",
            "3. If configuration 6's off-path gap (Resume@prefix on-path minus off-path success, frozen starts) is not "
            "smaller than both configuration 3's and 5's, claim C4 fails.",
            "4-5. Unchanged (M2: StableToolBench and AppWorld).", ""]  # fmt: skip

    out += ["## 9. Decisions for the human", "",
            "- The GPU budget and the token budget per run (section 6).",
            "- The test manifest: size per cell (section 7) and the held-out sets (D-028), then frozen.",
            "- The wording of EVAL_PLAN §9 (section 8).",
            "- Open questions that touch M1: Q17 (a4 on every graph), D-028c (the stateful extension), Q16, Q18.", "",
            "**STOP.** The agent does not start M1.5 or any further GPU run until the human decides.", ""]  # fmt: skip
    out += ["## Sources (sha256)", ""] + [f"- `{p}` {d}" for p, d in src.files]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gate", choices=["A"], required=True)
    parser.add_argument("--compile-run", type=Path)
    parser.add_argument("--stats-run", type=Path)
    parser.add_argument("--pilot-run", type=Path)
    parser.add_argument("--out", type=Path, default=REPO / "docs" / "reports" / "gate-A.md")
    args = parser.parse_args(argv)
    compile_run = args.compile_run or latest("M1-compile-")
    stats_run = args.stats_run or latest("M1-pilot-stats-")
    pilot = args.pilot_run
    if pilot is None:
        try:
            pilot = latest("M1-pilot-qet-")
        except SystemExit:
            pilot = None
    text = gate_a(*(REPO / p if not Path(p).is_absolute() else Path(p) for p in (compile_run, stats_run)),
                  pilot if pilot is None or Path(pilot).is_absolute() else REPO / pilot, Sources())  # fmt: skip
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()

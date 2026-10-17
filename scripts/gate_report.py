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
import sys
from collections import Counter
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
    """Every file the report reads, listed once with its sha256."""

    def __init__(self) -> None:
        self.files: list[tuple[str, str]] = []

    def file(self, path: Path) -> str:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if (str(path.relative_to(REPO)), digest) not in self.files:
            self.files.append((str(path.relative_to(REPO)), digest))
        return digest

    def json(self, path: Path):
        self.file(path)
        return json.loads(path.read_text())

    def lines(self, path: Path):
        self.file(path)
        return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def pct(x: float | None, digits: int = 1) -> str:
    return "–" if x is None else f"{100 * x:.{digits}f}%"


def num(x: float | None, digits: int = 3) -> str:
    if x is None:
        return "–"
    return f"{x:,.{digits}f}" if isinstance(x, float) else f"{x:,}"


WIDTH_BINS = {"1": (1, 1), "2": (2, 2), "3-4": (3, 4), "5+": (5, 10**9)}  # EVAL_PLAN §3.1


def binned(hist: dict[str, int], bins: list[tuple[int, int]]) -> list[int]:
    """Counts of a {value: count} histogram (string keys) in inclusive [lo, hi] bins."""
    return [sum(v for k, v in hist.items() if lo <= int(k) <= hi) for lo, hi in bins]


def mcnemar_pairs(delta: float, psi: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Paired tasks needed to detect a success-rate difference delta with discordant share psi (Connor 1987). The
    difference p10 - p01 cannot exceed the discordant share p10 + p01."""
    if not 0 < delta <= psi <= 1:
        raise ValueError(f"need 0 < delta <= psi <= 1, got delta={delta}, psi={psi}")
    z_a, z_b = NormalDist().inv_cdf(1 - alpha / 2), NormalDist().inv_cdf(power)
    return math.ceil((z_a * math.sqrt(psi) + z_b * math.sqrt(psi - delta**2)) ** 2 / delta**2)


def smallest_delta(pairs: int, psi: float, alpha: float = 0.05, power: float = 0.8) -> float | None:
    """The smallest success-rate difference (to 0.001, at most psi) that `pairs` paired tasks detect."""
    for i in range(1, round(1000 * psi) + 1):
        if mcnemar_pairs(i / 1000, psi, alpha, power) <= pairs:
            return i / 1000
    return None


def gold_action_tokens(records: list, fmt) -> dict[str, int]:
    """Each task's longest gold action in the backbone's tokens, end-of-turn token included: the target the decoder
    has to emit in one step (SPEC §6.7, D-002)."""
    from qet.core.canonical import action_string

    memo: dict[str, int] = {}
    longest = {}
    for rec in records:
        actions = [action_string(nd.tool, nd.args) for nd in rec.nodes]
        actions.append(action_string(rec.final["tool"], rec.final["args"]))
        for a in actions:
            if a not in memo:
                memo[a] = len(fmt.target_ids(a))
        longest[rec.task_id] = max(memo[a] for a in actions)
    return longest


def gate_a(compile_run: Path, stats_run: Path, pilot_run: Path | None, src: Sources) -> str:
    from qet.core.compile import read_records, width
    from qet.train.chat import ChatFormat

    cm = src.json(compile_run / "manifest.json")
    reports = {s: src.json(compile_run / s / "report.json") for s in cm["splits"]}
    stats_manifest = src.json(stats_run / "manifest.json")
    stats = src.json(stats_run / "stats.json")
    pm = src.json(pilot_run / "manifest.json") if pilot_run is not None else None

    def commit(m: dict) -> str:
        return f"commit {m['git']['commit'][:7]}, {'dirty' if m['git']['dirty'] else 'clean'}"

    out = ["# Gate A report (M1: FuncBenchGen)", ""]
    out += [f"Sources: compile run `{compile_run.name}` ({commit(cm)}); pilot statistics `{stats_run.name}` "
            f"({commit(stats_manifest)}; the first {stats['graphs']} training graphs, tokenizer "
            f"{stats_manifest['tokenizer']['backbone']})"
            + (f"; pilot run `{pilot_run.name}` ({commit(pm)}; job {pm['hardware']['scheduler_job']})." if pm
               else "; no pilot GPU run yet."), ""]  # fmt: skip

    out += ["## 1. Compile coverage and fallback", "",
            "| split | tasks | chain fallbacks | Compile missed a true constraint | caught by the replay audit | "
            "unsafe orders left |", "| --- | --- | --- | --- | --- | --- |"]  # fmt: skip
    for s, r in reports.items():
        a = r["audit"]
        out.append(f"| {s} | {num(r['compile']['tasks_compiled'])} | "
                   f"{num(a['replay_fallbacks'])} ({pct(r['compile']['fallback_share'])}) | "
                   f"{num(a['raw_missed_a_true_constraint'])} | {num(a['missed_and_caught_by_replay'])} | "
                   f"{num(a['final_with_unsafe_orders'])} |")  # fmt: skip
    out += ["", "Every task has a demonstration that passed the environment's checker before Compile: a failed check "
            "stops the compile run (`compile_trace` in src/qet/datasets.py), and this run completed. Compile received "
            f"no task constraints (D-020a). The replay audit ran {cm['settings']['audit_orders']} random orders plus "
            "one hoisted order per node (D-029).", ""]  # fmt: skip

    out += ["## 2. Agreement with the generator's true DAGs", "",
            "Means over graphs. The unsafe-order rate of a graph is the share of 200 sampled legal orders of its "
            "compiled graph that break a true constraint.", "",
            "| split | precision (raw → final) | recall (raw → final) | recall by reachability (raw → final) | "
            "unsafe-order rate (raw → final) |", "| --- | --- | --- | --- | --- |"]  # fmt: skip
    for s, r in reports.items():
        raw, fin = r["audit"]["raw"], r["audit"]["final"]
        cells = [f"{num(raw[k])} → {num(fin[k])}" for k in ("precision", "recall", "recall_by_reachability",
                                                             "unsafe_order_rate")]  # fmt: skip
        out.append(f"| {s} | " + " | ".join(cells) + " |")
    out += ["", "\"Final\" is after the chain fallback. On FuncBenchGen, raw precision below 1 comes from Compile's "
            "conservative extra edges (for example, an argument value that several earlier calls produced gets an "
            "edge from each, SPEC §4 rule 1), and the drop from raw to final from the chain fallbacks. The stateful "
            "extension's precision comes from decision edges around payments that truly commute.", ""]  # fmt: skip

    out += ["## 3. Structure: width, incomparable pairs, states", "",
            "Graphs as trained, after the chain fallback (a fallback graph has width 1 and n + 1 states).", "",
            "| split | width 1 | 2 | 3-4 | 5+ | incomparable pairs | states ≤ 16 | 17-64 | 65-256 | > S_max |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]  # fmt: skip
    for s, r in reports.items():
        c = r["compile"]
        w = binned(c["width"], list(WIDTH_BINS.values()))
        st = binned({k: v for k, v in c["states_per_graph"].items() if k.isdigit()}, [(0, 16), (17, 64), (65, 256)])
        over = sum(v for k, v in c["states_per_graph"].items() if not k.isdigit())
        total = c["tasks_compiled"]
        out.append(f"| {s} | " + " | ".join(pct(x / total) for x in w) + f" | {pct(c['incomparable_pair_share'])} | "
                   + " | ".join(pct(x / total) for x in st) + f" | {pct(over / total)} |")  # fmt: skip
    bins = ", ".join(f"{b}: {n}" for b, n in stats["width_bins"].items())
    out += ["", f"- Pilot sample ({stats['graphs']} training graphs): graphs per width bin {bins}; "
            f"{stats['states']['above_s_max']} graphs above S_max = {stats['s_max']} (trained on a sub-lattice of "
            f"K = {stats['k']} orders); mean exact lattice size {num(stats['states']['mean_exact'], 1)}.",
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
            "| configuration | scored, mean | gradient, mean | largest graph (scored + gradient) |",
            "| --- | --- | --- | --- |"]  # fmt: skip
    for c, t in stats["tokens_per_graph"].items():
        out.append(f"| {c} | {num(t['scored'], 0)} | {num(t['gradient'], 0)} | {num(t['max'])} |")
    out += ["", "By width bin (configurations 2 and 6): "
            + "; ".join(f"{b}: {num(v['2'], 0)} / {num(v['6'], 0)}" for b, v in
                        stats["tokens_per_graph_by_width_bin"].items()), ""]  # fmt: skip

    out += ["## 6. Throughput, memory and the GPU-hour projection", ""]
    episode_hours = one_eval = None  # GPU-hours of one evaluation episode / checkpoint on the current decoding path
    projection: dict[float, dict[str, float]] = {}  # per budget B: epochs and GPU-hours (section 10 quotes them)
    runs, grid = 0, {}
    if pm is not None:
        log = src.lines(pilot_run / "train_log.jsonl")
        rows = [r for r in log if r["phase"] == "train"]
        warm = [r for r in log if r["phase"] == "warmup"]
        gpus = len(pm["hardware"]["gpus"])  # GPU-hours count every GPU of the job
        s1, s2 = (sum(r[k]["tokens"] for r in rows) for k in ("scored", "gradient"))
        t1, t2 = (sum(r["seconds"][k] for r in rows) for k in ("pass1", "pass2"))
        rest = sum(r["seconds"][k] for r in rows for k in ("weights", "update", "diagnostic"))
        tps1, tps2 = s1 / t1, s2 / t2
        overhead = rest / (s1 + s2)  # seconds per budget token outside the two passes
        peak = max((r["max_memory_gib"] for r in log if "max_memory_gib" in r), default=None)
        hours = pm["gpu_hours_by_phase"]
        outside = hours["train"] * 3600 / gpus - sum(sum(r["seconds"].values()) for r in log)
        seen = {g for r in log for g in r["graphs"]}
        stop = pm["budgets"]["train"]["max_minutes"]
        out += [f"Pilot `{pilot_run.name}`: configuration {pm['configuration']['id']} on {pm['backbone']['id']}, "
                f"{stats['graphs']} training graphs in its data; the {stop:g}-minute training stop came after "
                f"{len(rows)} steps that reached {len(seen)} distinct graphs (the warm-up's among them). Throughput "
                "only, no claims. It ran in the live working tree (the pinned checkout of D-025b is still open); its "
                "manifest records the same commit at start and end and no code change during the run.", "",
                "| measurement | value |", "| --- | --- |",
                f"| steps | {len(warm)} warm-up (demo-order SFT), then {len(rows)} training |",
                f"| pass 1 (scoring, no gradients) | {num(s1)} tokens in {num(t1, 0)} s: {num(tps1, 0)} tokens/s |",
                f"| pass 2 (weighted SFT, forward and backward) | {num(s2)} tokens in {num(t2, 0)} s: {num(tps2, 0)} "
                "tokens/s |",
                f"| weights, optimizer update, diagnostic | {num(rest, 2)} s ({pct(rest / (t1 + t2 + rest), 2)} of "
                "step time) |",
                f"| train phase outside the logged steps (model setup, pack building) | {num(outside, 1)} s "
                f"({pct(outside / (hours['train'] * 3600 / gpus))} of the phase) |",
                f"| peak GPU memory allocated (`torch.cuda.max_memory_allocated`, max over steps) | {num(peak, 1)} GiB "
                "of 140.4 GiB (H200, 143,771 MiB) |",
                f"| GPU-hours | load {num(hours['load'], 3)}, train {num(hours['train'], 3)}, eval "
                f"{num(hours['eval'], 3)}; total {num(pm['gpu_hours'], 3)} (the other "
                f"{num(pm['gpu_hours'] - sum(hours.values()), 3)} is the run's bookkeeping outside these phases, such "
                "as writing its data folder) |", ""]  # fmt: skip

        # seconds per budget token: pass-1 configurations spend a share of the budget on scoring (cheaper per token)
        per_graph = stats["tokens_per_graph"]
        share = {c: t["scored"] / (t["scored"] + t["gradient"]) for c, t in per_graph.items()}
        share["1"] = share["a1"] = share["2"]  # no pass 1, like configuration 2
        spt = {c: s / tps1 + (1 - s) / tps2 + overhead for c, s in share.items()}
        matrix = [(c, 3) for c in ("1", "2", "3", "4", "5", "6")] + [(c, 3) for c in ("2", "3", "6")]
        matrix += [(c, 1) for c in ("a1", "a2", "a3", "a4", "a4", "a4")]  # a4 with K = 2, 4, 8: same share
        runs = sum(n for _, n in matrix)
        matrix_spt = sum(n * spt[c] for c, n in matrix)
        warm_late = warm[1:]  # the first step also initializes the GPU
        tps2_warm = sum(r["gradient"]["tokens"] for r in warm_late) / sum(r["seconds"]["pass2"] for r in warm_late)
        alt = sum(n * (spt[c] if share[c] else 1 / tps2_warm + overhead) for c, n in matrix)
        manifest = src.json(REPO / "configs" / "m1" / "manifests" / "train.json")
        cell = {t["task_id"]: (t["core"], t["depth"], t["irrelevant"], t["form"]) for t in manifest["tasks"]}
        graphs = src.lines(stats_run / "graphs.jsonl")
        full = {k: v / len(cell) for k, v in Counter(cell.values()).items()}

        def per_graph_full(c: str) -> float:  # the sample's mean per grid cell, weighted by the full split's cells
            by: dict[tuple, list[int]] = {}
            for g in graphs:
                t = g["tokens"][c]
                by.setdefault(cell[g["task_id"]], []).append(t["scored"] + t["gradient"])
            if set(by) != set(full):
                raise SystemExit("the pilot sample does not cover every training cell")
            return sum(full[k] * sum(v) / len(v) for k, v in by.items())

        epoch = {c: len(cell) * per_graph_full(c) for c in ("2", "6")}
        out += ["Training projection (a proposal: the token budget B per run is the human's decision). Matched compute "
                "gives every run the same B of scored + gradient tokens (EVAL_PLAN §8); a run's time follows its "
                "share of scoring tokens at the measured rate of each pass, plus the pilot's other step time per "
                f"token. Epochs are over the {len(cell):,} training graphs, from the pilot sample's mean tokens per "
                "graph in each of the 224 grid cells weighted by the full split's graphs per cell: about "
                f"{num(epoch['2'] / 1e6, 1)} million tokens per epoch for configuration 2 and "
                f"{num(epoch['6'] / 1e6, 0)} million for configuration 6.", "",
                "| B (tokens per run) | epochs, configuration 2 | epochs, configuration 6 | GPU-hours per run "
                f"(configuration 2 / 6) | M1 matrix, {runs} runs |", "| --- | --- | --- | --- | --- |"]  # fmt: skip
        core = ["1", "2", "3", "4", "5", "6", "2", "3", "6"]  # the core configurations on Qwen, then on Llama
        for b in (125e6, 250e6, 500e6, 1e9):
            projection[b] = {"epochs2": b / epoch["2"], "epochs6": b / epoch["6"], "matrix": b * matrix_spt / 3600}
            grid[b] = sum(2 * spt[c] * b / 3600 for c in core)  # two more learning rates per core configuration
            out.append(f"| {num(int(b))} | {num(b / epoch['2'], 2)} | {num(b / epoch['6'], 2)} | "
                       f"{num(b * spt['2'] / 3600, 1)} / {num(b * spt['6'] / 3600, 1)} | "
                       f"{num(b * matrix_spt / 3600, 0)} |")  # fmt: skip
        out += ["", "Assumptions and what is left out:",
                "- The matrix is EVAL_PLAN §4's: configurations 1-6 × 3 seeds on Qwen2.5-7B-Instruct; 2, 3 and 6 × 3 "
                "seeds on Llama-3.1-8B-Instruct, assumed to train at the measured Qwen throughput (not measured); "
                "a1, a2, a3 and a4 with K = 2, 4, 8 on one seed.",
                "- Every configuration is assumed to run each pass at configuration 6's measured rate. The pilot's "
                f"own warm-up steps 2-{len(warm)} (demo-order SFT, shaped like configuration 2) ran pass 2 at "
                f"{num(tps2_warm, 0)} tokens/s; at that rate for the configurations without pass 1, the matrix "
                f"takes {pct(alt / matrix_spt - 1)} longer.",
                "- Not included: dev tuning trials (EVAL_PLAN §8; section 10 prices a learning-rate grid); the format "
                "warm-up (D-010: its tokens count inside B, and configurations 2-6 share one warm-up per backbone "
                "and seed); pack building on the CPU (in the pilot, part of the train-phase time outside the logged "
                "steps above, for 500 graphs); evaluation (below).", ""]  # fmt: skip

        preds = src.lines(pilot_run / "predictions" / "greedy.jsonl")
        steps = [s for p in preds for s in p["steps"]]
        gen = sum(s["output_tokens"] for s in steps)
        eval_s = hours["eval"] * 3600 / gpus
        per_step = eval_s / max(1, len(steps))
        test_tasks = src.json(REPO / "configs" / "m1" / "manifests" / "test.json")["tasks"]
        dev_calls = {t["task_id"]: t["calls"] for t in src.json(REPO / "configs/m1/manifests/dev.json")["tasks"]}
        n_test = len(test_tasks)
        mean_steps = sum(t["calls"] + 1 for t in test_tasks) / n_test  # the demo path's calls plus finish
        pilot_calls = [dev_calls[p["task_id"]] for p in preds]
        episode_hours = mean_steps * per_step / 3600
        one_eval = n_test * 9 * episode_hours  # greedy + 8 samples per task (EVAL_PLAN §7)
        out += ["Evaluation with the current decoding path (Hugging Face, one sequence at a time, D-026b): the pilot "
                f"ran {len(preds)} greedy dev episodes, {len(steps)} generation steps and {num(gen)} generated tokens "
                f"in {num(eval_s, 0)} s ({num(per_step, 2)} s per step). Assuming an episode takes as many steps as "
                f"its demonstration (calls plus finish; {num(mean_steps, 1)} on the provisional test set; the pilot's "
                f"episodes took {len(steps)} steps where this gives {sum(c + 1 for c in pilot_calls)}), one "
                f"evaluation of one checkpoint on the {n_test:,} test tasks with greedy plus 8 samples would take "
                f"about {num(one_eval, 0)} GPU-hours, so {runs} runs need at least {num(runs * one_eval, 0)} before "
                "Resume@prefix, dev evaluations, the checkpoint curves, ToolHop and the held-out sets. This is "
                f"likely an underestimate: the pilot's tasks have {num(sum(pilot_calls) / len(pilot_calls), 1)} calls "
                f"on average, the test tasks {num(mean_steps - 1, 1)}, with longer prompts and actions. This path "
                "does not scale: batched decoding (vLLM, or a batched loop with the same stopping and sampling rules, "
                "D-026b) has to land and be measured before M1.7.", ""]  # fmt: skip
    else:
        out += ["- No pilot GPU run yet: the projection needs its measured throughput.", ""]

    out += ["## 7. Decoding limit", "",
            "The longest gold action of each task in tokens, end-of-turn token included, from the compiled "
            "demonstrations of each evaluation split. An output that reaches max_new_tokens without the end-of-turn "
            "token is an invalid action (D-026b), so a task whose gold action is longer than the limit cannot be "
            "solved.", "",
            "| split | backbone | longest gold action | tasks with one > 64 | > 128 | > 256 |",
            "| --- | --- | --- | --- | --- | --- |"]  # fmt: skip
    fmts = {b: ChatFormat.load(b) for b in ("Qwen/Qwen2.5-7B-Instruct", "meta-llama/Llama-3.1-8B-Instruct")}
    longest_all, over64_test, split_records = 0, None, {}
    for s in ("dev", "test", "heldout_larger", "renamed", "stateful_test"):
        if s not in cm["splits"]:
            continue
        path = compile_run / s / "records.jsonl"
        digest = src.file(path)
        if digest != cm["splits"][s]["files"]["records.jsonl"]:
            raise SystemExit(f"{path} does not match its compile manifest")
        records = split_records[s] = read_records(path)
        for b, fmt in fmts.items():
            longest = gold_action_tokens(records, fmt)
            top = max(longest.values())
            longest_all = max(longest_all, top)
            over = [sum(v > cap for v in longest.values()) / len(longest) for cap in (64, 128, 256)]
            if s == "test" and b.startswith("Qwen"):
                over64_test = over[0]
            out.append(f"| {s} | {b.split('/')[1]} | {num(top)} | " + " | ".join(pct(x) for x in over) + " |")
    proposal_cap = 64 * math.ceil(2 * longest_all / 64)
    out += ["", f"Proposal: max_new_tokens = {proposal_cap} for every system and backbone (twice the longest gold "
            "action, rounded up to a multiple of 64). The pilot's 64 would make "
            f"{pct(over64_test)} of the test tasks unsolvable on Qwen by construction. A higher limit costs only "
            "when an output runs on without the end-of-turn token. It is frozen with the benchmark manifest "
            "(EVAL_PLAN §10).", ""]  # fmt: skip

    out += ["## 8. Power analysis and proposed test size per cell", "",
            "Paired tasks needed to detect a success-rate difference Δ between two configurations with McNemar's test "
            "(two-sided α = 0.05, power 0.8, Connor 1987), for a share ψ of discordant tasks. Δ cannot exceed ψ "
            "(\"–\"); at Δ = ψ the normal approximation is at its weakest.", "",
            "| Δ \\ ψ | 0.10 | 0.20 | 0.30 |", "| --- | --- | --- | --- |"]  # fmt: skip
    for d in (0.05, 0.10, 0.15):
        out.append(f"| {d:.2f} | " + " | ".join(num(mcnemar_pairs(d, p)) if d <= p else "–" for p in (0.10, 0.20, 0.30))
                   + " |")  # fmt: skip
    test_widths = {r.task_id: width(r.n, r.preds()) for r in split_records["test"]}
    test_manifest = src.json(REPO / "configs" / "m1" / "manifests" / "test.json")["tasks"]
    by_cell: dict[tuple, list[str]] = {}
    for t in test_manifest:  # manifest order: the first k of each cell are the first k trials
        by_cell.setdefault((t["core"], t["depth"], t["irrelevant"], t["form"]), []).append(t["task_id"])
    shares = {
        b: sum(lo <= w <= hi for w in test_widths.values()) / len(test_widths) for b, (lo, hi) in WIDTH_BINS.items()
    }
    narrow = min(shares, key=shares.get)
    out += ["", f"Criterion 1 pools test graphs of width ≥ 3 ({pct(shares['3-4'] + shares['5+'])} of the provisional "
            f"test graphs); the smallest width bin (EVAL_PLAN §3.1) is width {narrow} ({pct(shares[narrow])}). A "
            "smaller test set takes the first k graphs of each cell of the provisional manifest; its pools are "
            "counted from the compiled test graphs. \"Smallest Δ\" is the least difference a pool detects with the "
            "power above, per comparison on one seed's greedy outcomes (three seeds per configuration add "
            "precision). Criterion 1 needs both (6 − 3) and (6 − 4) to be significant, so its joint power lies "
            "between 0.64 (independent comparisons) and 0.8.", "",
            "| graphs per cell | test tasks | width ≥ 3 pool | smallest Δ (ψ = 0.1 / 0.2 / 0.3) | smallest bin: "
            "tasks, Δ at ψ = 0.2 | evaluation GPU-hours per checkpoint, current decoding |",
            "| --- | --- | --- | --- | --- | --- |"]  # fmt: skip
    sizes = {}
    for k in (5, 10, 20):
        chosen = [tid for ids in by_cell.values() for tid in ids[:k]]
        pool = sum(test_widths[t] >= 3 for t in chosen)
        lo, hi = WIDTH_BINS[narrow]
        small = sum(lo <= test_widths[t] <= hi for t in chosen)
        sizes[k] = {"tasks": len(chosen), "delta": smallest_delta(pool, 0.20), "bin": smallest_delta(small, 0.20)}
        deltas = " / ".join(num(smallest_delta(pool, p), 3) for p in (0.10, 0.20, 0.30))
        cost = num(len(chosen) * 9 * episode_hours, 0) if episode_hours is not None else "–"
        out.append(f"| {k} | {num(len(chosen))} | {num(pool)} | {deltas} | {num(small)}, {num(sizes[k]['bin'], 3)} | "
                   f"{cost} |")  # fmt: skip
    out += ["", f"Proposal: 10 graphs per cell ({num(sizes[10]['tasks'])} test tasks): at ψ = 0.2 its width ≥ 3 pool "
            f"detects Δ = {num(sizes[10]['delta'], 3)} and its smallest bin {num(sizes[10]['bin'], 3)}, against "
            f"{num(sizes[20]['delta'], 3)} and {num(sizes[20]['bin'], 3)} for 20 per cell at twice the evaluation "
            "cost. The renamed set follows the test graphs it renames; held-out larger stays as built (D-028b). The "
            "human chooses; then the test manifests are frozen.", ""]  # fmt: skip

    out += ["## 9. Proposed wording for EVAL_PLAN §9 (for the human to accept, change or reject)", "",
            "Common terms (proposal): FuncBenchGen test graphs; every run evaluated at its checkpoint at 100% of B, so "
            "the compared runs have spent the same compute and no checkpoint is selected; success averaged over the "
            "three seeds of each task; differences with paired-bootstrap 95% intervals (10,000 resamples, clustered "
            "by task); \"significant\" means the interval excludes 0. McNemar per seed is reported alongside. "
            "Choosing each run's checkpoint by dev success among its 25/50/75/100% checkpoints would compare runs at "
            "different compute: a change to \"at matched compute\" that the human would have to adopt "
            "explicitly.", "",
            "1. Greedy success on graphs of measured width ≥ 3: if the interval of (6 − 3) or of (6 − 4) contains 0 "
            "or lies below it, the objective claim fails.",
            "2. Greedy success on all test graphs: if (2 − 1) is significant and positive but the interval of (6 − 2) "
            "contains 0 or lies below it, the gain belongs to the interface, not the objective.",
            "3. Resume@prefix greedy success on the test tasks that have off-path starts, with the off-path gap = "
            "on-path minus off-path success at the frozen starts: if the interval of (gap 3 − gap 6) or of "
            "(gap 5 − gap 6) contains 0 or lies below it, claim C4 fails.",
            "4-5. Unchanged (M2: StableToolBench and AppWorld).", ""]  # fmt: skip

    out += ["## 10. Decisions for the human", ""]
    if projection:
        p250, p500 = projection[250e6], projection[500e6]
        out += [f"1. **Training budget.** Proposal: B = 250,000,000 tokens per run, about {num(p250['matrix'], 0)} "
                f"GPU-hours of training for the {runs}-run matrix at the measured throughput (500,000,000: about "
                f"{num(p500['matrix'], 0)}). At 250M, configuration 2 gets {num(p250['epochs2'], 2)} epochs of the "
                f"5,000 graphs and configuration 6 {num(p250['epochs6'], 2)}: QET's lattice costs about "
                f"{num(epoch['6'] / epoch['2'], 1)} times the tokens per graph, so at matched compute it sees a "
                "fraction of the graphs. That is the comparison EVAL_PLAN §8 defines; the 25-100% curves show the "
                "trend within B, not what more compute would do. Each run is above the 1 GPU-hour per-job cap "
                "(D-021), so the budget is recorded as a batch approval in DECISIONS.md.",
                "2. **Tuning.** EVAL_PLAN §8 asks for the same number of dev trials per configuration; D-009 proposes "
                "a small learning-rate grid and defers it to Gate A. Proposal: no grid in M1, with D-009's defaults "
                "(AdamW, lr 1e-4, 3% warm-up, cosine decay) for every configuration, which is equal tuning with zero "
                "trials each. A 3-value grid on one seed at full B for the 9 core configurations (two extra rates "
                f"each; the default-rate run doubles as seed 0) would add about {num(grid[250e6], 0)} GPU-hours of "
                "training at 250M, plus dev evaluations.",
                f"3. **Evaluation.** The current decoder would need about {num(one_eval, 0)} GPU-hours per checkpoint "
                "evaluation at 20 graphs per cell (likely more; section 6). Proposal: the agent builds batched "
                "decoding (no GPU spending beyond short validation jobs within the per-job cap), measures it, and the "
                "evaluation budget is set from that measurement.",
                f"4. **Decoding limit** (section 7): max_new_tokens = {proposal_cap} for every system.",
                "5. **Test size per cell** (section 8) and the held-out sets (D-028); then the test manifests are "
                "frozen.",
                "6. **EVAL_PLAN §9 wording** (section 9), including the checkpoint rule: EVAL_PLAN.md is human-only.",
                "7. **Open items that touch M1:** Q17 (a4 on every graph), D-028c (the stateful extension), Q16, Q18; "
                "the warm-up length (D-010 fixes it once, from the pilot, as the steps until format errors on dev fall "
                "below 1%; the pilot ran a fixed 5 steps without measuring format errors, so it is still open; "
                "proposal: measure it at the start of M1.5); whether to measure Llama-3.1-8B throughput with its own "
                "short pilot (assumed equal to Qwen's above).", ""]  # fmt: skip
    else:
        out += ["- The GPU budget and the token budget per run (section 6, once the pilot has run).",
                "- Tuning (EVAL_PLAN §8, D-009), the decoding limit (section 7), the test size per cell (section 8) "
                "and the held-out sets (D-028), then frozen.",
                "- The wording of EVAL_PLAN §9 (section 9).",
                "- Open questions that touch M1: Q17 (a4 on every graph), D-028c (the stateful extension), Q16, Q18.",
                ""]  # fmt: skip
    out += ["**STOP.** The agent does not start M1.5 or any further GPU run until the human decides.", ""]
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
    sys.path.insert(0, str(REPO / "src"))
    main()

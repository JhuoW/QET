"""M1.4 pilot statistics: lattice sizes, width bins and tokens per graph on the first N training graphs of an M1.3
compile run (MILESTONES M1.4; EVAL_PLAN §3.1 width bins 1, 2, 3-4, 5+):

    python scripts/m1_pilot_stats.py --compile-run results/raw/M1-compile-<stamp> [--graphs 500] [--k 4]
                                     [--backbone Qwen/Qwen2.5-7B-Instruct]

Tokens per graph are the scored (pass 1) and gradient (pass 2) tokens one epoch spends on the graph under each
configuration, counted from the real prompts and targets with the backbone's tokenizer; graphs above S_max count the
sub-lattice of K orders (epoch 0). Writes results/raw/M1-pilot-stats-<stamp>/ (stats.json, graphs.jsonl, manifest.json).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CONFIGS = ["2", "3", "4", "5", "6", "a2", "a3", "a4"]


def width_bin(w: int) -> str:
    return {1: "1", 2: "2", 3: "3-4", 4: "3-4"}.get(w, "5+")


def main(argv: list[str] | None = None) -> Path:
    from qet.core.compile import incomparable_pairs, read_records, width
    from qet.core.lattice import S_MAX, enumerate_states, node_masks
    from qet.manifest import git_state, sha256_file, utc_now
    from qet.train.chat import ChatFormat
    from qet.train.methods import method
    from qet.train.trainer import PackBuilder, pass1_edges

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--compile-run", type=Path, required=True)
    parser.add_argument("--graphs", type=int, default=500)
    parser.add_argument("--k", type=int, default=4, help="sub-lattice orders above S_max (and a4)")
    parser.add_argument("--backbone", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--orders", type=int, default=5, help="random legal orders per graph for the invariance check")
    parser.add_argument("--out", type=Path, default=REPO / "results" / "raw")
    args = parser.parse_args(argv)
    run = args.compile_run if args.compile_run.is_absolute() else REPO / args.compile_run
    compiled = json.loads((run / "manifest.json").read_text())
    records_path = run / "train" / "records.jsonl"
    if sha256_file(records_path) != compiled["splits"]["train"]["files"]["records.jsonl"]:
        raise SystemExit(f"{records_path} does not match its compile manifest")
    records = read_records(records_path)[: args.graphs]
    fmt = ChatFormat.load(args.backbone)
    contracts = {t["name"]: {"reads": [], "writes": []} for r in records for t in r.tools if t["name"] != "finish"}
    builders = {c: PackBuilder(fmt, contracts, method(c), run_seed=0, k=args.k) for c in CONFIGS}
    import random

    from qet.core.lattice import random_topological_order
    from qet.core.render import render

    checks = failures = 0
    rows = []
    for rec in records:
        # SPEC §5.6 (a): every prefix of random legal orders renders like the demo-order history of the same set
        steps, task = rec.steps(), {"request": rec.request, "tools": rec.tools}
        rng, masks = random.Random(f"invariance\x1f{rec.task_id}"), node_masks(rec.n, rec.preds())
        for _ in range(args.orders):
            order = random_topological_order(masks, rng)
            for k in range(rec.n + 1):
                done = sorted(order[:k])
                legal = render(task, [steps[v] for v in order[:k]], contracts)
                checks += 1
                failures += legal != render(task, [steps[v] for v in done], contracts)
        preds = rec.preds()
        states = enumerate_states(node_masks(rec.n, preds), cap=S_MAX)
        row = {"task_id": rec.task_id, "n": rec.n, "width": width(rec.n, preds),
               "incomparable_pairs": incomparable_pairs(rec.n, preds), "pairs": rec.n * (rec.n - 1) // 2,
               "states": len(states) if states is not None else None, "chain_fallback": rec.compile.fallback_to_chain,
               "tokens": {}}  # fmt: skip
        for c, b in builders.items():
            pack = b.build(rec, 0)
            edges = pass1_edges(pack, b.method)  # the edges pass 2 can weight, too
            gradient = sum(len(pack.pair(e)) for e in edges)
            row["tokens"][c] = {"scored": gradient if b.method.pass1 else 0, "gradient": gradient,
                                "states": len(pack.lat.states), "exact": pack.lat.exact}  # fmt: skip
        rows.append(row)

    def mean(xs):
        xs = list(xs)
        return sum(xs) / len(xs) if xs else None

    bins = {}
    for r in rows:
        bins.setdefault(width_bin(r["width"]), []).append(r)
    stats = {
        "graphs": len(rows),
        "width_bins": {b: len(v) for b, v in sorted(bins.items())},
        "chain_fallbacks": sum(r["chain_fallback"] for r in rows),
        "incomparable_pair_share": sum(r["incomparable_pairs"] for r in rows) / max(1, sum(r["pairs"] for r in rows)),
        "states": {
            "exact_graphs": sum(r["states"] is not None for r in rows),
            "above_s_max": sum(r["states"] is None for r in rows),
            "mean_exact": mean(r["states"] for r in rows if r["states"] is not None),
        },  # fmt: skip
        "tokens_per_graph": {
            c: {
                "scored": mean(r["tokens"][c]["scored"] for r in rows),
                "gradient": mean(r["tokens"][c]["gradient"] for r in rows),
                "max": max(r["tokens"][c]["scored"] + r["tokens"][c]["gradient"] for r in rows),
            }
            for c in CONFIGS
        },  # fmt: skip
        "tokens_per_graph_by_width_bin": {
            b: {c: mean(r["tokens"][c]["scored"] + r["tokens"][c]["gradient"] for r in v) for c in ("2", "6")}
            for b, v in sorted(bins.items())
        },  # fmt: skip
        "s_max": S_MAX,
        "k": args.k,
        "invariance": {"prefixes_checked": checks, "failures": failures, "orders_per_graph": args.orders},
    }
    created = utc_now()
    out = args.out / f"M1-pilot-stats-{created:%Y%m%dT%H%M%SZ}"
    out.mkdir(parents=True)
    (out / "stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    with open(out / "graphs.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    manifest = {"run_id": out.name, "created_at": created.isoformat(), "git": git_state(),
                "script": "scripts/m1_pilot_stats.py", "compile_run": str(run.relative_to(REPO))
                if run.is_relative_to(REPO) else str(run), "records_sha256": sha256_file(records_path),
                "graphs": len(rows), "k": args.k, "tokenizer": fmt.fingerprint(),
                "files": {p.name: sha256_file(p) for p in sorted(out.iterdir())}, "status": "completed"}  # fmt: skip
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(out)
    return out


if __name__ == "__main__":
    sys.path.insert(0, str(REPO / "src"))
    main()

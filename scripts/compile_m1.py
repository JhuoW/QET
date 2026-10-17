"""M1.3: demonstrations, Compile and audits for the M1 manifests (D-007, D-029; SPEC §4; EVAL_PLAN §7):

    python scripts/compile_m1.py [--splits SPLIT ...] [--limit N] [--allow-dirty]

For every task of each manifest: a demonstration (a seeded random topological order of the generator's true DAG,
executed and verified by the environment's checker), Compile on the trace alone, the replay audit (random and hoisted
orders; a mismatch makes the graph a chain), and the audit against the true DAG. Writes results/raw/M1-compile-<stamp>/
(per split: records, Compile's raw records, per-task audits, the compile report) with a manifest, and a RUNLOG row.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SPLITS = ["train", "dev", "test", "heldout_larger", "renamed", "stateful_train", "stateful_test"]


def compile_split(manifest: dict, *, limit: int | None, demo_seed: int, audit_orders: int):
    from qet.datasets import audit_summary, compile_trace, m1_tasks
    from qet.envs.funcbenchgen import FuncBenchGenEnv
    from qet.envs.toy import ToyEnv

    if limit is not None:
        manifest = {**manifest, "tasks": manifest["tasks"][:limit]}
    stateful = manifest["source"] == "qet.envs.toy.stateful_task"
    make_env = ToyEnv if stateful else FuncBenchGenEnv
    compiled = [compile_trace(t, make_env, demo_seed=demo_seed, audit_orders=audit_orders) for t in m1_tasks(manifest)]
    source = "by construction (accounts and orders)" if stateful else FuncBenchGenEnv.contract_source
    return compiled, audit_summary(compiled), source


def main(argv: list[str] | None = None) -> Path:
    from qet import runlog
    from qet.core.compile import compile_report, write_records
    from qet.manifest import git_state, is_committed, sha256_file, utc_now

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifests", type=Path, default=REPO / "configs" / "m1" / "manifests")
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=SPLITS)
    parser.add_argument("--limit", type=int, help="the first N tasks of each split (checks)")
    parser.add_argument("--demo-seed", type=int, default=0)
    parser.add_argument("--audit-orders", type=int, default=8, help="random orders of the replay audit")
    parser.add_argument("--out", type=Path, default=REPO / "results" / "raw")
    parser.add_argument("--runlog", type=Path, default=REPO / "docs" / "RUNLOG.md")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args(argv)
    git = git_state()
    paths = {s: args.manifests / f"{s}.json" for s in args.splits}
    if not args.allow_dirty and (git["dirty"] or not all(is_committed(p) for p in paths.values())):
        raise SystemExit(f"compile needs committed manifests and a clean tree (dirty: {git['dirty_files']})")
    created = utc_now()
    run_id = f"M1-compile-{created:%Y%m%dT%H%M%SZ}" + (f"-limit{args.limit}" if args.limit else "")
    out = args.out / run_id
    out.mkdir(parents=True)
    rel = lambda p: str(p.relative_to(REPO)) if p.resolve().is_relative_to(REPO) else str(p)  # noqa: E731
    row = {"run_id": run_id, "submitted": created.isoformat(), "milestone": "M1", "config": "scripts/compile_m1.py",
           "method": "compile", "backbone": "-", "seed": "-", "status": "running", "GPU-h": "0", "path": rel(out),
           "commit": git["commit"][:7] + ("+dirty" if git["dirty"] else ""),
           "note": f"limit={args.limit}" if args.limit else ""}  # fmt: skip
    runlog.append_row(args.runlog, row)
    splits = {}
    try:
        for split, path in paths.items():
            t0 = time.perf_counter()
            manifest = json.loads(path.read_text())
            compiled, audits, source = compile_split(manifest, limit=args.limit, demo_seed=args.demo_seed,
                                                     audit_orders=args.audit_orders)  # fmt: skip
            folder = out / split
            folder.mkdir()
            write_records(folder / "records.jsonl", [c.record for c in compiled])
            write_records(folder / "raw_records.jsonl", [c.raw for c in compiled])
            with open(folder / "audits.jsonl", "w", encoding="utf-8") as f:
                for c in compiled:
                    row = {"task_id": c.record.task_id, "replayed": c.replayed, "mismatches": c.mismatches,
                           "raw": c.audit_raw, "final": c.audit_final}  # fmt: skip
                    f.write(json.dumps(row) + "\n")
            report = {"compile": compile_report([c.record for c in compiled], contract_source=source),
                      "compile_raw": compile_report([c.raw for c in compiled], contract_source=source),
                      "audit": audits}  # fmt: skip
            (folder / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            splits[split] = {"manifest": rel(path), "manifest_sha256": sha256_file(path), "tasks": len(compiled),
                             "seconds": time.perf_counter() - t0,
                             "files": {p.name: sha256_file(p) for p in sorted(folder.iterdir())}}  # fmt: skip
            print(f"{split}: {len(compiled)} tasks, {audits['replay_fallbacks']} chain fallbacks", flush=True)
        status = "completed"
    except BaseException as e:
        status = f"failed: {type(e).__name__}: {e}"
        raise
    finally:
        record = {"run_id": run_id, "created_at": created.isoformat(), "finished_at": utc_now().isoformat(),
                  "git": git, "script": "scripts/compile_m1.py", "settings": {"demo_seed": args.demo_seed,
                  "audit_orders": args.audit_orders, "hoisted_orders": True, "limit": args.limit},
                  "splits": splits, "status": status}  # fmt: skip
        (out / "manifest.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        runlog.update_row(args.runlog, run_id, status="completed" if status == "completed" else "failed")
    print(out)
    return out


if __name__ == "__main__":
    sys.path.insert(0, str(REPO / "src"))
    main()

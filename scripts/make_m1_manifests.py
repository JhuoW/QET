"""Build the M1 manifests (MILESTONES M1.2) from configs/m1/funcbenchgen.yaml:

    python scripts/make_m1_manifests.py [--spec configs/m1/funcbenchgen.yaml] [--out configs/m1/manifests]
                                        [--only SPLIT ...] [--check]

One JSON file per split lists every task (its cell, seed, trial, variant) with the sha256 of its content (public text,
gold calls, answer, true DAG), so a task list can be rebuilt and verified from code and the pinned repository. Seeds
come from disjoint ranges per split. With --check, the files are rebuilt in memory and compared with the ones on disk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[1]


def spec_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def funcbenchgen_tasks(spec: dict[str, Any], split: str) -> list[dict[str, Any]]:
    from qet.envs.funcbenchgen import build_task, cell_config, graph_payloads, grid_cells, task_digest

    grid = spec["grid"]
    cells = grid_cells(grid["core"], grid["depth"], grid["irrelevant"])
    s = spec["splits"][split]
    entries: list[tuple[dict[str, Any], int, bool]] = []
    if split == "train":
        entries = [(cells[i % len(cells)], s["seed_start"] + i, False) for i in range(s["graphs"])]
    elif split in ("dev", "test"):
        entries = [(c, s["seed_start"] + ci * 10_000 + j, False) for ci, c in enumerate(cells)
                   for j in range(s["graphs_per_cell"])]  # fmt: skip
    elif split == "heldout_larger":
        big = grid_cells(s["core"], s["depth"], s["irrelevant"])
        entries = [(c, s["seed_start"] + ci * 10_000 + j, False) for ci, c in enumerate(big)
                   for j in range(s["graphs_per_cell"])]  # fmt: skip
    elif split == "renamed":
        of = spec["splits"][s["of"]]
        entries = [(c, of["seed_start"] + ci * 10_000 + j, True) for ci, c in enumerate(cells)
                   for j in range(of["graphs_per_cell"])]  # fmt: skip
    else:
        raise ValueError(f"unknown split {split!r}")
    graph_payloads([(cell_config(**c), seed) for c, seed, _ in entries])  # every graph in one helper call
    out = []
    for cell, seed, renamed in entries:
        task = build_task(cell_config(**cell), seed, 0, renamed=renamed)
        out.append({"task_id": task.task_id, **cell, "graph_seed": seed, "trial": 0, "renamed": renamed,
                    "calls": task.min_calls, "sha256": task_digest(task)})  # fmt: skip
    return out


def stateful_tasks(spec: dict[str, Any], split: str) -> list[dict[str, Any]]:
    from qet.envs.toy import stateful_task

    st = spec["stateful"]
    cells, s = st["cells"], st[split]
    if split == "train":
        entries = [(cells[i % len(cells)], s["seed_start"] + i) for i in range(s["graphs"])]
    else:
        entries = [
            (c, s["seed_start"] + ci * 10_000 + j) for ci, c in enumerate(cells) for j in range(s["graphs_per_cell"])
        ]
    out = []
    for cell, seed in entries:
        task = stateful_task(seed, **cell)
        content = {"request": task.request, "calls": list(task.calls), "answer": task.answer,
                   "true_preds": [list(p) for p in task.true_preds]}  # fmt: skip
        digest = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
        out.append({"task_id": f"{task.task_id}-o{cell['n_orders']}a{cell['n_accounts']}", **cell, "seed": seed,
                    "calls": len(task.calls), "sha256": digest})  # fmt: skip
    return out


SPLITS = ["train", "dev", "test", "heldout_larger", "renamed", "stateful_train", "stateful_test"]


def build(spec_path: Path, split: str) -> dict[str, Any]:
    spec = yaml.safe_load(spec_path.read_text())
    if split.startswith("stateful_"):
        tasks, source = stateful_tasks(spec, split.split("_", 1)[1]), "qet.envs.toy.stateful_task"
    else:
        tasks, source = funcbenchgen_tasks(spec, split), "qet.envs.funcbenchgen.build_task"
    ids = [t["task_id"] for t in tasks]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{split}: duplicate task ids")
    return {"split": split, "spec": str(spec_path.relative_to(REPO)), "spec_sha256": spec_sha256(spec_path),
            "funcbenchgen_commit": spec["funcbenchgen_commit"], "source": source, "count": len(tasks),
            "provisional": split in ("test", "renamed", "heldout_larger"), "tasks": tasks}  # fmt: skip


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec", type=Path, default=REPO / "configs" / "m1" / "funcbenchgen.yaml")
    parser.add_argument("--out", type=Path, default=REPO / "configs" / "m1" / "manifests")
    parser.add_argument("--only", nargs="+", choices=SPLITS, default=SPLITS)
    parser.add_argument("--check", action="store_true", help="rebuild in memory and compare with the files")
    args = parser.parse_args(argv)
    spec = yaml.safe_load(args.spec.read_text())
    head = subprocess.run(["git", "-C", str(REPO / "third_party" / "FuncBenchGen"), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()  # fmt: skip
    if head != spec["funcbenchgen_commit"]:
        raise SystemExit(f"third_party/FuncBenchGen is at {head}, the spec pins {spec['funcbenchgen_commit']}")
    failed = False
    for split in args.only:
        text = json.dumps(build(args.spec.resolve(), split), indent=1) + "\n"
        path = args.out / f"{split}.json"
        if args.check:
            ok = path.exists() and path.read_text() == text
            failed |= not ok
            print(f"{split}: {'identical' if ok else 'DIFFERS'}")
        else:
            args.out.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            print(f"{split}: {path.relative_to(REPO)} sha256 {hashlib.sha256(text.encode()).hexdigest()}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

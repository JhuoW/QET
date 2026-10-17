"""The data of a run: training graph records (compiled from executed demonstrations; Compile never sees a true DAG),
evaluation tasks, the tool contracts, and a data manifest that pins all of it by sha256."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qet.config import DataConfig
from qet.core.compile import GraphRecord, compile_report, write_records
from qet.envs.base import Environment
from qet.envs.toy import ToyEnv, ToyTask, compile_toy, make_tasks
from qet.manifest import sha256_file, sha256_json


@dataclass
class Dataset:
    train_records: list[GraphRecord]
    test_tasks: list[ToyTask]
    test_records: list[GraphRecord]  # the test tasks' compiled demonstrations: Resume@prefix only, never trained on
    contracts: dict[str, Any]
    contract_source: str
    make_env: Callable[[ToyTask], Environment]
    spec: dict[str, Any]  # what generated the data


def _merged_contracts(tasks: list[ToyTask]) -> dict[str, Any]:
    """One contract per tool across the dataset; a tool name with two different contracts is an error."""
    merged: dict[str, Any] = {}
    for t in tasks:
        for tool, c in t.contracts.items():
            if merged.setdefault(tool, c) != c:
                raise ValueError(f"tool {tool} has two different contracts in the dataset")
    return dict(sorted(merged.items()))


def build_dataset(cfg: DataConfig, *, limit: int | None = None) -> Dataset:
    """The toy env's tasks from disjoint seed ranges; demonstrations are seeded random topological orders (D-007)."""
    if cfg.env != "toy":
        raise ValueError(f"unknown env {cfg.env!r}")
    n_train = cfg.train.count if limit is None else min(limit, cfg.train.count)
    n_test = cfg.test.count if limit is None else min(limit, cfg.test.count)
    train = make_tasks(cfg.kind, range(cfg.train.start, cfg.train.start + n_train), **cfg.params)
    test = make_tasks(cfg.kind, range(cfg.test.start, cfg.test.start + n_test), **cfg.params)
    train_records = [compile_toy(t, cfg.demo_seed)[0] for t in train]
    test_records = [compile_toy(t, cfg.demo_seed)[0] for t in test]
    spec = {"env": cfg.env, "kind": cfg.kind, "params": cfg.params, "demo_seed": cfg.demo_seed,
            "train": {"start": cfg.train.start, "count": n_train}, "test": {"start": cfg.test.start, "count": n_test},
            "truncation": cfg.truncation}  # fmt: skip
    env0 = ToyEnv(train[0] if train else test[0])
    return Dataset(
        train_records, test, test_records, _merged_contracts(train + test), env0.contract_source, ToyEnv, spec
    )


def write_dataset(ds: Dataset, out: Path) -> tuple[Path, str]:
    """Write the records, the evaluation task list, the compile report and the data manifest; return the manifest's
    path and sha256."""
    out.mkdir(parents=True, exist_ok=True)
    write_records(out / "train_records.jsonl", ds.train_records)
    write_records(out / "test_records.jsonl", ds.test_records)
    (out / "test_tasks.json").write_text(json.dumps([t.task_id for t in ds.test_tasks]) + "\n", encoding="utf-8")
    report = compile_report(ds.train_records, contract_source=ds.contract_source)
    (out / "compile_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "spec": ds.spec,
        "contracts_sha256": sha256_json(ds.contracts),
        "contract_source": ds.contract_source,
        "files": {
            name: sha256_file(out / name)
            for name in ("train_records.jsonl", "test_records.jsonl", "test_tasks.json", "compile_report.json")
        },  # fmt: skip
        "train_task_ids": [r.task_id for r in ds.train_records],
        "test_task_ids": [t.task_id for t in ds.test_tasks],
    }
    path = out / "data_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path, sha256_file(path)

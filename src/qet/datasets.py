"""The data of a run: training graph records (compiled from executed demonstrations; Compile never sees a true DAG),
evaluation tasks, the tool contracts, and a data manifest that pins all of it by sha256."""

from __future__ import annotations

import dataclasses
import json
import random
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qet.config import DataConfig
from qet.core.compile import GraphRecord, compile_report, write_records
from qet.envs.base import Environment, run_calls
from qet.envs.toy import ToyEnv, ToyTask, compile_toy, demonstration, make_tasks
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
    benchmark: dict[str, Any] = dataclasses.field(default_factory=dict)  # the manifest's benchmark / evaluator fields


def _merged_contracts(tasks: list[ToyTask]) -> dict[str, Any]:
    """One contract per tool across the dataset; a tool name with two different contracts is an error."""
    merged: dict[str, Any] = {}
    for t in tasks:
        for tool, c in t.contracts.items():
            if merged.setdefault(tool, c) != c:
                raise ValueError(f"tool {tool} has two different contracts in the dataset")
    return dict(sorted(merged.items()))


def build_dataset(cfg: DataConfig, *, limit: int | None = None) -> Dataset:
    """The toy env's tasks from disjoint seed ranges; demonstrations are seeded random topological orders (D-007).
    FuncBenchGen runs read the records of an M1.3 compile run instead (see `funcbenchgen_dataset`)."""
    if cfg.env == "funcbenchgen":
        return funcbenchgen_dataset(cfg, limit=limit)
    if cfg.env == "stateful":
        return stateful_dataset(cfg, limit=limit)
    if cfg.env == "toolhop":
        return toolhop_dataset(cfg, limit=limit)
    if cfg.env != "toy":
        raise ValueError(f"unknown env {cfg.env!r}")
    n_train = cfg.train.count if limit is None else min(limit, cfg.train.count)
    n_test = cfg.test.count if limit is None else min(limit, cfg.test.count)
    train = make_tasks(cfg.kind, range(cfg.train.start, cfg.train.start + n_train), **cfg.params)
    test = make_tasks(cfg.kind, range(cfg.test.start, cfg.test.start + n_test), **cfg.params)
    for t in train + test:  # demonstrations are verified by the environment's checker (SPEC §1)
        demo, _ = demonstration(t, cfg.demo_seed)
        env = ToyEnv(t)
        run_calls(env, demo[:-1])
        if not env.check(demo[-1]["args"]["answer"]):
            raise ValueError(f"the demonstration of {t.task_id} does not pass the checker")
    train_records = [compile_toy(t, cfg.demo_seed)[0] for t in train]
    test_records = [compile_toy(t, cfg.demo_seed)[0] for t in test]
    spec = {"env": cfg.env, "kind": cfg.kind, "params": cfg.params, "demo_seed": cfg.demo_seed,
            "train": {"start": cfg.train.start, "count": n_train}, "test": {"start": cfg.test.start, "count": n_test},
            "truncation": cfg.truncation}  # fmt: skip
    env0 = ToyEnv(train[0] if train else test[0])
    benchmark = {"name": "toy", "code": "src/qet/envs/toy.py", "evaluator": "ToyEnv.check",
                 "step_budget": "twice the minimum number of calls"}  # fmt: skip
    return Dataset(
        train_records,
        test,
        test_records,
        _merged_contracts(train + test),
        env0.contract_source,
        ToyEnv,
        spec,
        benchmark,
    )


def write_dataset(ds: Dataset, out: Path, resume_starts: list[dict[str, Any]] | None = None) -> tuple[Path, str]:
    """Write the records, the evaluation task list, the compile report, the Resume@prefix starts (frozen by the
    evaluation seed, the same for every training seed) and the data manifest; return the manifest's path and sha256."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "resume_starts.json").write_text(json.dumps(resume_starts or []) + "\n", encoding="utf-8")
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
            for name in (
                "train_records.jsonl",
                "test_records.jsonl",
                "test_tasks.json",
                "compile_report.json",
                "resume_starts.json",
            )
        },  # fmt: skip
        "train_task_ids": [r.task_id for r in ds.train_records],
        "test_task_ids": [t.task_id for t in ds.test_tasks],
    }
    path = out / "data_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path, sha256_file(path)


# ----------------------------------------------------------------------------- M1 (FuncBenchGen, stateful extension)
@dataclass
class CompiledTask:
    """One evaluation or training task after M1.3: the record trained on (after the replay audit) and the audits."""

    record: GraphRecord  # Compile's graph, or its chain fallback after a failed replay audit (SPEC §4)
    raw: GraphRecord  # Compile's own output
    true_preds: list[list[int]]  # the generator's DAG by demo position (audits only, never Compile's input)
    replayed: int
    mismatches: int
    audit_raw: dict[str, Any]
    audit_final: dict[str, Any]


def compile_trace(
    task: Any, make_env: Callable[[Any], Environment], *, demo_seed: int, audit_orders: int
) -> CompiledTask:
    """D-007: the demonstration is a seeded random topological order of the generator's true DAG, executed in the
    environment and verified by its checker; Compile sees the trace only (no task constraints, D-020a); the replay
    audit executes `audit_orders` random legal orders and every node's hoisted order (D-029), and falls back to a
    chain on any mismatch."""
    from qet.core.compile import compile_demo, fallback_to_chain
    from qet.core.lattice import node_masks, random_topological_order
    from qet.envs.base import edge_audit, replay_audit

    rng = random.Random(f"m1-demo\x1f{task.task_id}\x1f{demo_seed}")
    order = random_topological_order(node_masks(len(task.calls), task.true_preds), rng)
    env = make_env(task)
    steps = run_calls(env, [task.calls[i] for i in order])
    if not env.check(task.answer) or getattr(env, "failures", None):
        raise ValueError(f"the demonstration of {task.task_id} does not pass the checker")
    demo = [*steps, {"tool": "finish", "args": {"answer": task.answer}}]
    raw = compile_demo(task.public(), demo, env.contracts(), task_id=task.task_id)
    replayed, mismatches = replay_audit(lambda: make_env(task), raw, k=audit_orders, seed=demo_seed, hoisted=True)
    if mismatches:
        record = fallback_to_chain(raw, orders_replayed=replayed, mismatches=mismatches)
    else:
        record = dataclasses.replace(raw, compile=dataclasses.replace(raw.compile, orders_replayed=replayed))
    position = {call: i for i, call in enumerate(order)}
    true_preds = [sorted(position[u] for u in task.true_preds[call]) for call in order]
    return CompiledTask(record, raw, true_preds, replayed, mismatches, edge_audit(raw, true_preds),
                        edge_audit(record, true_preds))  # fmt: skip


def m1_tasks(manifest: dict[str, Any]) -> list[Any]:
    """The tasks a manifest lists, rebuilt and checked against its content hashes."""
    from qet.envs.funcbenchgen import build_task, cell_config, graph_payloads, task_digest
    from qet.envs.toy import stateful_task

    entries = manifest["tasks"]
    if manifest["source"] == "qet.envs.toy.stateful_task":
        tasks = []
        for e in entries:
            t = stateful_task(e["seed"], n_orders=e["n_orders"], n_accounts=e["n_accounts"], task_id=e["task_id"])
            tasks.append(t)
        return tasks
    cells = [cell_config(e["core"], e["depth"], e["irrelevant"], e["form"]) for e in entries]
    graph_payloads([(c, e["graph_seed"]) for c, e in zip(cells, entries, strict=True)])
    tasks = []
    for c, e in zip(cells, entries, strict=True):
        t = build_task(c, e["graph_seed"], e["trial"], renamed=e["renamed"])
        if (t.task_id, task_digest(t)) != (e["task_id"], e["sha256"]):
            raise ValueError(f"{e['task_id']} does not match its manifest entry")
        tasks.append(t)
    return tasks


def audit_summary(compiled: list[CompiledTask]) -> dict[str, Any]:
    """Compile against the true DAGs (EVAL_PLAN §7): edge precision and recall, recall by reachability, the
    unsafe-order rate, and how the replay audit handled the graphs whose Compile missed a true constraint."""

    def mean(xs: list[float]) -> float | None:
        return sum(xs) / len(xs) if xs else None

    missed = [c for c in compiled if c.audit_raw["recall_by_reachability"] < 1.0]
    return {
        "graphs": len(compiled),
        "task_constraints_passed": 0,  # D-020a: no task_order reaches Compile in M1
        "raw": {
            k: mean([c.audit_raw[k] for c in compiled])
            for k in ("precision", "recall", "recall_by_reachability", "unsafe_order_rate")
        },  # fmt: skip
        "final": {
            k: mean([c.audit_final[k] for c in compiled])
            for k in ("precision", "recall", "recall_by_reachability", "unsafe_order_rate")
        },  # fmt: skip
        "raw_missed_a_true_constraint": len(missed),
        "missed_and_caught_by_replay": sum(c.mismatches > 0 for c in missed),
        "final_with_unsafe_orders": sum(c.audit_final["unsafe_order_rate"] > 0 for c in compiled),
        "replay_fallbacks": sum(c.mismatches > 0 for c in compiled),
        "replay_orders_per_graph": compiled[0].replayed if compiled else None,
    }


def inject_false_independence(records: Sequence[GraphRecord], rate: float, seed: int) -> list[GraphRecord]:
    """The injected false-independence ablation (EVAL_PLAN §5 block C, D-033): in a share `rate` of the graphs that have
    a dependency, drop one edge of the transitive reduction (a true constraint with no other path between its ends),
    so the lattice holds orders that are illegal in truth. The draw is seeded by `seed` and the task id; the demo order
    stays legal."""
    from qet.core.compile import ancestor_masks

    out = []
    for rec in records:
        rng = random.Random(f"inject-fi\x1f{seed}\x1f{rec.task_id}")
        reduction = []
        for e in rec.edges:
            others = [frozenset(u for u in p if (u, v) != (e.u, e.v)) for v, p in enumerate(rec.preds())]
            if not ancestor_masks(rec.n, others)[e.v] >> e.u & 1:
                reduction.append(e)
        if reduction and rng.random() < rate:
            drop = rng.choice(sorted(reduction))
            rec = dataclasses.replace(rec, edges=tuple(e for e in rec.edges if e != drop))
        out.append(rec)
    return out


def toolhop_dataset(cfg: DataConfig, *, limit: int | None = None) -> Dataset:
    """ToolHop, evaluation only (EVAL_PLAN §3.2, D-035): the pinned queries in the mandatory tool-use setting."""
    from qet.envs.toolhop import DATA, DATA_SHA256, DATASET_COMMIT, ToolHopEnv, load_tasks

    n_test = cfg.test.count if limit is None else min(limit, cfg.test.count)
    tasks = load_tasks()[cfg.test.start : cfg.test.start + n_test]
    tools = sorted({t["name"] for task in tasks for t in task.tools} - {"finish"})
    spec = {"env": "toolhop", "data": str(DATA.relative_to(DATA.parents[3])), "sha256": DATA_SHA256,
            "test": {"start": cfg.test.start, "count": n_test}, "truncation": cfg.truncation}  # fmt: skip
    benchmark = {"name": "toolhop", "dataset": "bytedance-research/ToolHop", "commit": DATASET_COMMIT,
                 "code": "src/qet/envs/toolhop.py", "evaluator": "the official answer check (code/evaluation_open.py)",
                 "step_budget": "9 turns: 8 tool calls and the answer (the official max_turns)"}  # fmt: skip
    contracts = {name: {"reads": [], "writes": []} for name in tools}
    return Dataset([], tasks, [], contracts, ToolHopEnv.contract_source, ToolHopEnv, spec, benchmark)


def stateful_dataset(cfg: DataConfig, *, limit: int | None = None) -> Dataset:
    """The stateful extension (D-028c): the records of an M1.3 compile run's stateful splits (checked against the
    hashes in its manifest) and the toy tasks rebuilt from the manifests it compiled; ToyEnv with the tasks' own
    contracts (by construction)."""
    from qet.core.compile import read_records
    from qet.manifest import REPO, sha256_file

    src = cfg.source or {}
    run = REPO / src["compile_run"]
    compiled = json.loads((run / "manifest.json").read_text())
    if compiled.get("status") != "completed":
        raise ValueError(f"compile run {run} did not complete")
    for split in (src["train"], src["eval"]):
        for name, digest in compiled["splits"][split]["files"].items():
            if sha256_file(run / split / name) != digest:
                raise ValueError(f"{run / split / name} does not match its compile manifest")
    n_train = cfg.train.count if limit is None else min(limit, cfg.train.count)
    n_test = cfg.test.count if limit is None else min(limit, cfg.test.count)

    def split(name: str, start: int, n: int) -> tuple[list[GraphRecord], list[Any]]:
        records = read_records(run / name / "records.jsonl")[start : start + n]
        manifest = json.loads((REPO / compiled["splits"][name]["manifest"]).read_text())
        if manifest["source"] != "qet.envs.toy.stateful_task":
            raise ValueError(f"split {name} is not a stateful split")
        tasks = m1_tasks({**manifest, "tasks": manifest["tasks"][start : start + n]})
        if [t.task_id for t in tasks] != [r.task_id for r in records] or len(tasks) != n:
            raise ValueError(f"tasks and records of {name} do not line up")
        return records, tasks

    train, train_tasks = split(src["train"], cfg.train.start, n_train)
    test_records, tasks = split(src["eval"], cfg.test.start, n_test)
    spec = {"env": "stateful", "compile_run": src["compile_run"], "train": {"split": src["train"],
            "start": cfg.train.start, "count": n_train}, "eval": {"split": src["eval"], "start": cfg.test.start,
            "count": n_test}, "truncation": cfg.truncation}  # fmt: skip
    benchmark = {"name": "stateful-toy (D-028c)", "code": "src/qet/envs/toy.py", "evaluator": "ToyEnv.check",
                 "step_budget": "twice the minimum number of calls"}  # fmt: skip
    env0 = ToyEnv((train_tasks or tasks)[0])
    return Dataset(train, tasks, test_records, _merged_contracts(train_tasks + tasks), env0.contract_source, ToyEnv,
                   spec, benchmark)  # fmt: skip


def funcbenchgen_dataset(cfg: DataConfig, *, limit: int | None = None) -> Dataset:
    """Training records from an M1.3 compile run (checked against the hashes in its manifest) and evaluation tasks
    rebuilt from the manifest that run compiled; `train` and `test` slice the two splits."""
    from qet.core.compile import read_records
    from qet.envs.funcbenchgen import FuncBenchGenEnv
    from qet.manifest import REPO, sha256_file

    src = cfg.source or {}
    run = REPO / src["compile_run"]
    compiled = json.loads((run / "manifest.json").read_text())
    if compiled.get("status") != "completed":
        raise ValueError(f"compile run {run} did not complete")
    for split in (src["train"], src["eval"]):
        for name, digest in compiled["splits"][split]["files"].items():
            if sha256_file(run / split / name) != digest:
                raise ValueError(f"{run / split / name} does not match its compile manifest")
    n_train = cfg.train.count if limit is None else min(limit, cfg.train.count)
    n_test = cfg.test.count if limit is None else min(limit, cfg.test.count)
    train = read_records(run / src["train"] / "records.jsonl")[cfg.train.start : cfg.train.start + n_train]
    test_records = read_records(run / src["eval"] / "records.jsonl")[cfg.test.start : cfg.test.start + n_test]
    manifest = json.loads((REPO / compiled["splits"][src["eval"]]["manifest"]).read_text())
    entries = manifest["tasks"][cfg.test.start : cfg.test.start + n_test]
    tasks = m1_tasks({**manifest, "tasks": entries})
    if [t.task_id for t in tasks] != [r.task_id for r in test_records]:
        raise ValueError("evaluation tasks and records do not line up")
    if len(train) != n_train or len(tasks) != n_test:
        raise ValueError("the compile run has fewer records than the config asks for")
    tools = {t["name"] for r in train for t in r.tools} | {t["name"] for task in tasks for t in task.tools}
    contracts = {name: {"reads": [], "writes": []} for name in sorted(tools - {"finish"})}  # pure (SPEC §3)
    if cfg.inject:
        train = inject_false_independence(train, float(cfg.inject["false_independence"]), int(cfg.inject["seed"]))
    spec = {"env": "funcbenchgen", "compile_run": src["compile_run"], "train": {"split": src["train"],
            "start": cfg.train.start, "count": n_train}, "eval": {"split": src["eval"], "start": cfg.test.start,
            "count": n_test}, "truncation": cfg.truncation}  # fmt: skip
    if cfg.inject:
        spec["inject"] = dict(cfg.inject)
    commit = subprocess.run(["git", "-C", str(REPO / "third_party" / "FuncBenchGen"), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()  # fmt: skip
    benchmark = {"name": "funcbenchgen", "code": "src/qet/envs/funcbenchgen.py", "commit": commit,
                 "evaluator": "FuncBenchGenEnv.check (the repository's rule)",
                 "step_budget": "twice the minimum number of calls (the repository's cap)"}  # fmt: skip
    return Dataset(train, tasks, test_records, contracts, FuncBenchGenEnv.contract_source, FuncBenchGenEnv, spec,
                   benchmark)  # fmt: skip

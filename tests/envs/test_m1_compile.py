"""M1.3: demonstrations, Compile and the audits (D-007, D-029; SPEC §4; EVAL_PLAN §7) on small samples of the M1
manifests, and the compile script end to end into a temporary folder."""

import json
import sys
from pathlib import Path

import pytest

from qet.core.compile import read_records
from qet.datasets import audit_summary, compile_trace, m1_tasks
from qet.envs.base import hoisted_orders
from qet.envs.funcbenchgen import FuncBenchGenEnv
from qet.envs.toy import ToyEnv

REPO = Path(__file__).resolve().parents[2]
MANIFESTS = REPO / "configs" / "m1" / "manifests"


def sample(split, n):
    manifest = json.loads((MANIFESTS / f"{split}.json").read_text())
    return {**manifest, "tasks": manifest["tasks"][:n]}


@pytest.fixture(scope="module")
def compiled():
    return [compile_trace(t, FuncBenchGenEnv, demo_seed=0, audit_orders=8) for t in m1_tasks(sample("train", 120))]


@pytest.mark.slow  # starts interpreters or compiles a 120-task sample
def test_every_true_constraint_survives_the_audit(compiled):
    for c in compiled:
        assert c.audit_final["recall_by_reachability"] == 1.0 and c.audit_final["unsafe_order_rate"] == 0.0, (
            c.raw.task_id
        )
        missed = c.audit_raw["recall_by_reachability"] < 1.0
        assert (c.mismatches > 0) == missed  # pure functions: the audit fails exactly when Compile missed a dependency
        assert c.record.compile.fallback_to_chain == missed and c.replayed == 8 + c.raw.n


@pytest.mark.slow  # starts interpreters or compiles a 120-task sample
def test_compile_sees_no_task_constraints_and_records_its_audit(compiled):
    for c in compiled:
        assert all("task" not in e.labels for e in c.raw.edges)  # D-020a: no true edge reaches Compile
        assert c.record.compile.orders_replayed == c.replayed
    summary = audit_summary(compiled)
    assert summary["task_constraints_passed"] == 0 and summary["final_with_unsafe_orders"] == 0
    assert summary["missed_and_caught_by_replay"] == summary["raw_missed_a_true_constraint"]


def test_hoisted_orders_are_legal_and_put_each_node_early():
    (c,) = [compile_trace(t, FuncBenchGenEnv, demo_seed=0, audit_orders=0) for t in m1_tasks(sample("dev", 1))]
    preds = c.raw.preds()
    for v, order in enumerate(hoisted_orders(c.raw)):
        assert sorted(order) == list(range(c.raw.n))
        seen = set()
        for u in order:
            assert preds[u] <= seen
            seen.add(u)
        later = {u for u in range(v) if order.index(u) > order.index(v)}
        assert later == {u for u in range(v) if u not in seen_before(order, v)}


def seen_before(order, v):
    return set(order[: order.index(v)])


def test_the_stateful_extension_compiles_safely():
    comp = [compile_trace(t, ToyEnv, demo_seed=0, audit_orders=8) for t in m1_tasks(sample("stateful_train", 30))]
    summary = audit_summary(comp)
    assert summary["final"]["recall_by_reachability"] == 1.0 and summary["final_with_unsafe_orders"] == 0
    assert summary["final"]["precision"] < 1.0  # payments truly commute, which the contracts cannot certify


def test_the_compile_script_writes_its_run(tmp_path):
    sys.path.insert(0, str(REPO / "scripts"))
    import compile_m1

    log = tmp_path / "RUNLOG.md"
    from qet import runlog

    log.write_text(f"# Run log\n\n{runlog.HEADER}\n| {' | '.join('---' for _ in runlog.COLUMNS)} |\n")
    out = compile_m1.main(["--splits", "dev", "stateful_test", "--limit", "3", "--allow-dirty", "--out",
                           str(tmp_path / "raw"), "--runlog", str(log)])  # fmt: skip
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "completed" and set(manifest["splits"]) == {"dev", "stateful_test"}
    for split in ("dev", "stateful_test"):
        records = read_records(out / split / "records.jsonl")
        assert len(records) == 3 and set(manifest["splits"][split]["files"]) == {
            "audits.jsonl", "raw_records.jsonl", "records.jsonl", "report.json"}  # fmt: skip
        report = json.loads((out / split / "report.json").read_text())
        assert report["compile"]["tasks_compiled"] == 3 and report["audit"]["graphs"] == 3
    assert runlog.read_rows(log)[0]["status"] == "completed"

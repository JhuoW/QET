"""M1.2: the M1 manifests (configs/m1/manifests/*.json) built from configs/m1/funcbenchgen.yaml: disjoint splits,
the full EVAL_PLAN §3.1 grid, held-out sets that differ from training as specified, and tasks that rebuild exactly."""

import json
import sys
from pathlib import Path

import networkx as nx
import pytest
import yaml

from qet.envs.funcbenchgen import build_task, cell_config, graph_payloads, task_digest
from qet.envs.toy import stateful_task

REPO = Path(__file__).resolve().parents[2]
MANIFESTS = REPO / "configs" / "m1" / "manifests"
SPEC = yaml.safe_load((REPO / "configs" / "m1" / "funcbenchgen.yaml").read_text())
SPLITS = ["train", "dev", "test", "heldout_larger", "renamed", "stateful_train", "stateful_test"]


@pytest.fixture(scope="module")
def manifests():
    return {s: json.loads((MANIFESTS / f"{s}.json").read_text()) for s in SPLITS}


def test_splits_are_disjoint(manifests):
    fbg = {s: {(t["core"], t["depth"], t["irrelevant"], t["form"], t["graph_seed"]) for t in manifests[s]["tasks"]}
           for s in ("train", "dev", "test", "heldout_larger")}  # fmt: skip
    for a in fbg:
        for b in fbg:
            if a < b:
                assert not fbg[a] & fbg[b], (a, b)
    seeds = {s: {t["graph_seed"] for t in manifests[s]["tasks"]} for s in ("train", "dev", "test", "heldout_larger")}
    assert max(seeds["train"]) < min(seeds["dev"]) and max(seeds["dev"]) < min(seeds["test"])
    assert max(seeds["test"]) < min(seeds["heldout_larger"])
    st_train = {t["seed"] for t in manifests["stateful_train"]["tasks"]}
    assert not st_train & {t["seed"] for t in manifests["stateful_test"]["tasks"]}
    ids = [t["task_id"] for s in SPLITS for t in manifests[s]["tasks"]]
    assert len(ids) == len(set(ids))


def test_the_grid_is_the_eval_plans(manifests):
    cells = {(t["core"], t["depth"], t["irrelevant"], t["form"]) for t in manifests["test"]["tasks"]}
    expected = {(n, d, 0, "none") for n in (5, 10, 20) for d in range(1, n)}
    expected |= {(n, d, k, f) for n in (5, 10, 20) for d in range(1, n) for k in (10, 20)
                 for f in ("connected", "disconnected", "mixed")}  # fmt: skip
    assert cells == expected and len(cells) == 224
    train_cells = {(t["core"], t["depth"], t["irrelevant"], t["form"]) for t in manifests["train"]["tasks"]}
    assert train_cells == expected and manifests["train"]["count"] == 5000
    assert manifests["test"]["provisional"] and manifests["test"]["count"] == 224 * 20  # sized at Gate A


def test_held_out_sets(manifests):
    most = max(t["calls"] for t in manifests["train"]["tasks"])
    assert min(t["calls"] for t in manifests["heldout_larger"]["tasks"]) > most  # more calls than in training
    test = {t["task_id"]: t for t in manifests["test"]["tasks"]}
    renamed = manifests["renamed"]["tasks"]
    assert len(renamed) == len(test) and all(t["renamed"] for t in renamed)
    sample = renamed[::400]
    graph_payloads([(cell_config(t["core"], t["depth"], t["irrelevant"], t["form"]), t["graph_seed"]) for t in sample])
    for t in sample:  # the same graph under new names and values
        cfg = cell_config(t["core"], t["depth"], t["irrelevant"], t["form"])
        a, b = build_task(cfg, t["graph_seed"]), build_task(cfg, t["graph_seed"], renamed=True)
        ga = nx.DiGraph([(u, v) for v, ps in enumerate(a.true_preds) for u in ps])
        gb = nx.DiGraph([(u, v) for v, ps in enumerate(b.true_preds) for u in ps])
        ga.add_nodes_from(range(len(a.calls)))
        gb.add_nodes_from(range(len(b.calls)))
        assert nx.is_isomorphic(ga, gb)
        names_a, names_b = {x["name"] for x in a.tools}, {x["name"] for x in b.tools}
        assert names_a & names_b == {"finish"} and a.request != b.request


@pytest.mark.parametrize("split", ["train", "dev", "test", "heldout_larger", "renamed"])
def test_funcbenchgen_tasks_rebuild_exactly(manifests, split):
    sample = manifests[split]["tasks"][:: max(1, manifests[split]["count"] // 25)]
    cells = {t["task_id"]: cell_config(t["core"], t["depth"], t["irrelevant"], t["form"]) for t in sample}
    graph_payloads([(cells[t["task_id"]], t["graph_seed"]) for t in sample])  # one helper call for the sample
    for t in sample:
        cfg = cells[t["task_id"]]
        task = build_task(cfg, t["graph_seed"], t["trial"], renamed=t["renamed"])
        assert (task.task_id, task_digest(task), task.min_calls) == (t["task_id"], t["sha256"], t["calls"])


def test_stateful_tasks_rebuild_exactly(manifests):
    sys.path.insert(0, str(REPO / "scripts"))
    from make_m1_manifests import stateful_tasks

    for split in ("train", "test"):
        rebuilt = stateful_tasks(SPEC, split)
        assert rebuilt == manifests[f"stateful_{split}"]["tasks"]
    assert stateful_task(0, n_orders=3, n_accounts=1).task_id == "toy-st-0"


def test_manifests_name_their_spec_and_pinned_repository(manifests):
    for m in manifests.values():
        assert m["spec"] == "configs/m1/funcbenchgen.yaml"
        assert m["funcbenchgen_commit"] == SPEC["funcbenchgen_commit"] == "0729e2567dfaee1e53262ba9bfd5ba6a4b4e4343"

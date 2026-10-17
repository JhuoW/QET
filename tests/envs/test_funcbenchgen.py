"""M1.1: the FuncBenchGen adapter wraps the pinned repository (third_party/FuncBenchGen) without changing it: gold
replay scores 100%, wrong-input behaviour is the repository's own, success follows its rule, contracts are pure."""

import copy
import json
import random
import types

import pytest

from qet.core.lattice import node_masks, random_topological_order
from qet.envs.base import run_calls
from qet.envs.funcbenchgen import FuncBenchGenEnv, GridConfig, _global_rng, _repo, _seeded_state, build_task

CELLS = [
    GridConfig(core_nodes=5, max_critical_path=1, num_total_nodes=5),
    GridConfig(core_nodes=5, max_critical_path=3, num_total_nodes=15, num_disconnected_nodes=10),
    GridConfig(core_nodes=10, max_critical_path=4, num_total_nodes=20, num_disconnected_nodes=5),
    GridConfig(core_nodes=20, max_critical_path=10, num_total_nodes=30),
]


@pytest.mark.parametrize("cfg", CELLS, ids=lambda c: c.tag())
def test_gold_replay_scores_100_percent(cfg):
    for graph_seed in range(0, 1500, 100):
        task = build_task(cfg, graph_seed)
        assert len(task.calls) == cfg.min_calls  # the needed functions are the core functions
        rng = random.Random(graph_seed)
        for _ in range(3):  # any topological order of the true DAG
            env = FuncBenchGenEnv(task)
            order = random_topological_order(node_masks(len(task.calls), task.true_preds), rng)
            steps = run_calls(env, [task.calls[i] for i in order])
            assert env.failures == [] and all("error" not in s["obs"] for s in steps)
            assert env.check(task.answer) and len(task.calls) <= env.step_budget() == 2 * cfg.min_calls


def test_success_needs_the_minimum_number_of_calls_and_the_exact_answer():
    task = build_task(CELLS[1], 7)
    env = FuncBenchGenEnv(task)
    assert not env.check(task.answer)  # no calls yet: the repository requires at least the minimum
    run_calls(env, task.calls)
    assert env.check(task.answer) and env.check(float(task.answer))
    assert not env.check(task.answer + 1) and not env.check(str(task.answer)) and not env.check(None)


def test_wrong_inputs_behave_as_in_the_repository():
    """The adapter's observations are the repository evaluator's results on the same calls and the same RNG state."""
    fbg, _, _, _ = _repo()
    task = build_task(CELLS[2], 300)
    names = [t["name"] for t in task.tools if t["name"] != "finish"]
    rng = random.Random(1)
    calls = []
    for _ in range(40):  # right calls, calls before their inputs are known, wrong values, unknown names
        c = rng.choice(task.calls) if rng.random() < 0.6 else {"tool": rng.choice(names), "args": {}}
        args = dict(c["args"]) or {
            p: rng.randrange(1000) for p in next(t for t in task.tools if t["name"] == c["tool"])["params"]
        }
        if rng.random() < 0.2:
            k = rng.choice(sorted(args))
            args[k] = (args[k] + 1) % 1000
        calls.append({"tool": c["tool"] if rng.random() > 0.05 else "func_nope", "args": args})
    env = FuncBenchGenEnv(task)
    got = [env.step(c["tool"], c["args"]) for c in calls]
    evaluator, known, var_to_type = task._proto
    evaluator, known = copy.deepcopy(evaluator), set(known)
    _, _, _, utils = _repo()
    seed = utils.stable_int_from(task.task_id, "episode")
    state = _seeded_state(seed, seed + 7)
    expected, reasons = [], []
    for c in calls:
        call = types.SimpleNamespace(function=types.SimpleNamespace(name=c["tool"], arguments=json.dumps(c["args"])))
        with _global_rng(state):
            try:
                result, reason = evaluator._execute_tool(call, known, use_types=True, var_to_type=var_to_type)
            except Exception:
                result, reason = "Error occurred.", "Tool input was invalid"
        expected.append(result)
        reasons.append(reason)
    for obs, result in zip(got, expected, strict=True):
        if "error" in obs:
            assert obs == {"error": result}
        else:
            ((var, value),) = obs.items()
            assert result == f"Variable {var} = {value}."
    assert [f["reason"] for f in env.failures] == [r for r in reasons if r]
    assert {"Variable value not yet known", "Value of variable was incorrect"} <= set(reasons)
    assert FuncBenchGenEnv(task).step(calls[0]["tool"], calls[0]["args"]) == got[0]  # the same in every episode


def test_tasks_are_deterministic_and_trials_differ():
    a, b = build_task(CELLS[0], 5), build_task(CELLS[0], 5)
    assert (a.request, a.tools, a.calls, a.answer) == (b.request, b.tools, b.calls, b.answer)
    c = build_task(CELLS[0], 5, trial=1)
    assert c.request != a.request  # new values for a new trial
    state = random.getstate()
    build_task(CELLS[3], 11)
    assert random.getstate() == state  # the repository's global-RNG use leaves everyone else's alone


def test_public_task_and_contracts():
    task = build_task(CELLS[1], 9)
    x = task.public()
    assert x["initial_observation"] is None and x["finish"] == "finish"
    assert x["request"].startswith("Using the tools at your disposal")
    assert any(t["name"] == "finish" for t in x["tools"])
    env = FuncBenchGenEnv(task)
    assert set(env.contracts()) == {t["name"] for t in x["tools"]} - {"finish"}
    assert all(c == {"reads": [], "writes": []} for c in env.contracts().values())


def test_tasks_are_the_same_under_any_hash_seed():
    """FuncBenchGen's generator depends on string hashing; the adapter builds its graphs under PYTHONHASHSEED=0."""
    import os
    import subprocess
    import sys

    code = ("from qet.envs.funcbenchgen import build_task, cell_config, task_digest; "
            "print(task_digest(build_task(cell_config(10, 4, 10, 'mixed'), 7)), "
            "task_digest(build_task(cell_config(20, 9, 20, 'connected'), 123, renamed=True)))")  # fmt: skip
    outs = [
        subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        ).stdout.split()  # fmt: skip
        for seed in ("0", "1", "12345")
    ]
    assert outs[0] == outs[1] == outs[2] and len(outs[0]) == 2

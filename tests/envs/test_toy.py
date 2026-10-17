"""E1.5: the toy environment. Gold replay scores 100%; Compile on toy demos recalls every edge of the true DAG."""

import pytest

from qet.core.compile import incomparable_pairs, width
from qet.core.contracts import as_contracts
from qet.envs.base import edge_audit, replay_audit, run_calls
from qet.envs.toy import ToyEnv, compile_toy, demonstration, function_task, make_tasks, stateful_task

SETTINGS = [
    ("functions", {}),
    ("functions", {"n_core": 6, "n_given": 3, "n_distractors": 3, "max_inputs": 3}),
    ("functions", {"n_core": 3, "n_given": 1, "n_distractors": 2, "disconnected": True}),
    ("functions", {"n_core": 1, "n_given": 1, "n_distractors": 0}),
    ("stateful", {}),
    ("stateful", {"n_orders": 4, "n_accounts": 1}),
    ("stateful", {"n_orders": 1, "n_accounts": 1}),
]


def tasks(kind, kwargs, n=60):
    return make_tasks(kind, range(n), **kwargs)


@pytest.mark.parametrize(("kind", "kwargs"), SETTINGS)
def test_gold_replay_scores_100_percent(kind, kwargs):
    for toy in tasks(kind, kwargs):
        demo, _ = demonstration(toy)
        env = ToyEnv(toy)
        steps = run_calls(env, demo[:-1])
        assert all("error" not in s["obs"] for s in steps), toy.task_id
        assert env.check(demo[-1]["args"]["answer"]), toy.task_id
        env = ToyEnv(toy)  # the gold calls in generation order too
        run_calls(env, toy.calls)
        assert env.check(toy.answer)
        assert len(toy.calls) <= env.step_budget()


@pytest.mark.parametrize(("kind", "kwargs"), SETTINGS)
def test_compile_recalls_every_true_edge(kind, kwargs):
    precisions = []
    for toy in tasks(kind, kwargs):
        record, true_preds = compile_toy(toy)
        audit = edge_audit(record, true_preds)
        assert audit["recall"] == 1.0 and audit["recall_by_reachability"] == 1.0, toy.task_id
        assert audit["unsafe_order_rate"] == 0.0
        assert record.compile.underived == ()  # every argument comes from the request or an earlier observation
        assert record.compile.volatile_consumed == ()
        precisions.append(audit["precision"])
    if kind == "functions":
        assert set(precisions) == {1.0}  # pure functions over distinct values: Compile finds exactly the true DAG
    elif kwargs.get("n_orders", 3) > 1:
        assert min(precisions) < 1.0  # payments from one account commute, which the contracts cannot certify


@pytest.mark.parametrize(("kind", "kwargs"), SETTINGS)
def test_replay_audit_finds_no_mismatch(kind, kwargs):
    for toy in tasks(kind, kwargs, n=20):
        record, _ = compile_toy(toy)
        assert replay_audit(lambda toy=toy: ToyEnv(toy), record, k=8, seed=1) == (8, 0)


def test_graphs_are_wider_than_chains():
    fn = [compile_toy(t)[0] for t in make_tasks("functions", range(100), n_core=6, n_given=3)]
    st = [compile_toy(t)[0] for t in make_tasks("stateful", range(100))]
    for records in (fn, st):
        assert sum(width(r.n, r.preds()) >= 2 for r in records) >= 30
        assert sum(incomparable_pairs(r.n, r.preds()) for r in records) > 0


def test_generation_is_deterministic_and_seeded():
    assert function_task(7) == function_task(7) and stateful_task(7) == stateful_task(7)
    assert function_task(7) != function_task(8) and stateful_task(7) != stateful_task(8)
    assert demonstration(function_task(7)) == demonstration(function_task(7))
    orders = {tuple(demonstration(function_task(3, n_core=6, n_given=3), seed=s)[1]) for s in range(20)}
    assert len(orders) > 1  # demo orders are random topological orders of the true DAG (D-007)


def test_function_semantics():
    toy = function_task(11, n_core=4, n_given=2)
    env = ToyEnv(toy)
    call = toy.calls[0]
    right = env.step(call["tool"], call["args"])
    out = next(k for k in right if k != "request_id")
    assert right[out] == toy.world["values"][out]
    wrong_args = {k: v + 1 for k, v in call["args"].items()}
    w1, w2 = env.step(call["tool"], wrong_args), env.step(call["tool"], wrong_args)
    assert w1[out] == w2[out] != right[out] and 100 <= w1[out] <= 999  # wrong inputs: a seeded random value
    float_args = {k: float(v) for k, v in call["args"].items()}
    assert env.step(call["tool"], float_args)[out] == right[out]  # 412.0 is the canonical 412
    string_args = {k: str(v) for k, v in call["args"].items()}
    assert env.step(call["tool"], string_args)[out] != right[out]
    assert "error" in env.step(call["tool"], {})
    assert "error" in env.step("no_such_tool", {})
    assert w1["request_id"] != w2["request_id"]
    contracts = as_contracts(env.contracts())
    assert contracts.canon_obs(call["tool"], w1) == contracts.canon_obs(call["tool"], w2)  # request_id is volatile
    assert not env.check(toy.answer + 1) and env.check(toy.answer) and env.check(float(toy.answer))
    assert not env.check(str(toy.answer)) and not env.check(True)


def test_stateful_semantics():
    toy = stateful_task(5, n_orders=2, n_accounts=1)
    env = ToyEnv(toy)
    (o1, o2), account = toy.world["targets"], toy.world["balances"]
    acct = next(iter(account))
    amount = env.step("get_order", {"order": o1})["amount"]
    assert "error" in env.step("pay", {"account": acct, "order": o1, "amount": amount + 1})  # in full only
    snap = env.snapshot()
    assert env.step("pay", {"account": acct, "order": o1, "amount": amount}) == {"receipt": f"R-{o1}", "status": "paid"}
    assert "error" in env.step("pay", {"account": acct, "order": o1, "amount": amount})  # already paid
    assert not env.check(toy.answer)  # o2 is still open
    env.restore(snap)
    assert env.step("get_order", {"order": o1})["status"] == "open"
    assert "error" in env.step("get_balance", {"account": 0})
    assert "error" in env.step("get_order", {"order": "x"})
    run_calls(env, toy.calls)
    assert env.check(toy.answer)
    assert env.final_state()["orders"] == {str(o1): "paid", str(o2): "paid"}


def test_public_task_hides_the_semantics():
    for toy in (function_task(2), stateful_task(2)):
        x = toy.public()
        assert set(x) == {"request", "tools", "initial_observation", "finish"} and x["initial_observation"] is None
        assert [t["name"] for t in x["tools"]] == sorted(t["name"] for t in x["tools"]) and x["tools"][0]
        assert any(t["name"] == "finish" for t in x["tools"])
        assert str(toy.answer) not in x["request"]
    fn = function_task(2, n_core=4)
    hidden = [fn.world["values"][spec["output"]] for spec in fn.world["functions"].values()]
    assert len(hidden) == 6 and all(str(v) not in fn.request for v in hidden)  # computed values are never stated

"""Toy environment (E1.5): seeded function-DAG tasks (FuncBenchGen-like) and a stateful accounts-and-orders variant.

Function tasks: a hidden DAG of pure functions over distinct three-digit integers. The request states the input
values only; every intermediate value has to be computed by calling the function that returns it. Correct inputs give
the fixed output, wrong inputs a seeded random value. Distractor functions are callable but not needed: connected ones
take task variables, disconnected ones take extra stated values. Each observation carries a `request_id` that depends
on the order of calls; the contracts declare it volatile.

Stateful tasks: pay orders from accounts, then report a balance. `get_balance` and `get_order` read; `pay` reads and
writes its account and order. Two payments from one account commute (the balance covers every payment) but conflict
under the contracts: pairs that truly commute and that Compile cannot certify (EVAL_PLAN §3.1).

A task's gold calls and true DAG serve the gold replay and the Compile audit only; Compile sees the executed trace.
"""

from __future__ import annotations

import copy
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from qet.core.canonical import JSON, canon, canon_json
from qet.core.compile import GraphRecord, compile_demo
from qet.core.lattice import node_masks, random_topological_order
from qet.envs.base import FINISH_TOOL, run_calls


@dataclass(frozen=True)
class ToyTask:
    task_id: str
    kind: str  # "functions" or "stateful"
    seed: int
    request: str
    tools: tuple[dict[str, Any], ...]  # sorted by name, finish included
    contracts: dict[str, Any]
    calls: tuple[dict[str, Any], ...]  # the needed calls {tool, args}, in generation order (topological)
    true_preds: tuple[tuple[int, ...], ...]  # the generator's true DAG over `calls`
    answer: int
    world: dict[str, Any]  # hidden semantics: the function table and values, or the accounts and orders

    def public(self) -> dict[str, Any]:
        """The task x (SPEC §1)."""
        return {"request": self.request, "tools": [dict(t) for t in self.tools], "initial_observation": None,
                "finish": "finish"}  # fmt: skip


# ----------------------------------------------------------------------------- function tasks
def function_task(
    seed: int,
    *,
    n_core: int = 4,
    n_given: int = 2,
    n_distractors: int = 2,
    max_inputs: int = 2,
    disconnected: bool = False,
    task_id: str | None = None,
) -> ToyTask:
    """A random function DAG whose target depends on all `n_core` functions."""
    if n_core < 1 or n_given < 1 or n_distractors < 0 or max_inputs < 1:
        raise ValueError("need n_core >= 1, n_given >= 1, n_distractors >= 0, max_inputs >= 1")
    rng = random.Random(f"toy-functions-{seed}")
    n_extra = n_distractors if disconnected else 0  # stated values that only distractors take
    names = [f"v{i}" for i in range(1, n_given + n_extra + n_core + n_distractors + 1)]
    rng.shuffle(names)
    given = names[:n_given]
    extra = names[n_given : n_given + n_extra]
    core = names[n_given + n_extra : n_given + n_extra + n_core]  # function i returns core[i]; core[-1] is the target
    dist = names[n_given + n_extra + n_core :]
    values = dict(zip(names, rng.sample(range(100, 1000), len(names)), strict=True))

    inputs: list[set[str]] = [set() for _ in range(n_core)]
    for i in range(n_core - 1):  # each core value feeds a later core function, so the target needs every one
        inputs[rng.randrange(i + 1, n_core)].add(core[i])
    for g in given:  # each stated value is used
        inputs[rng.randrange(n_core)].add(g)
    for i in range(n_core):
        pool = [x for x in given + core[:i] if x not in inputs[i]]
        while pool and (not inputs[i] or (len(inputs[i]) < max_inputs and rng.random() < 0.5)):
            inputs[i].add(pool.pop(rng.randrange(len(pool))))
    table = {f"get_{core[i]}": {"output": core[i], "inputs": sorted(inputs[i])} for i in range(n_core)}
    for k, d in enumerate(dist):
        pool = sorted(extra) if disconnected else sorted(given + core)
        first = [extra[k]] if disconnected else []
        rest = [x for x in pool if x not in first]
        chosen = first + rng.sample(rest, min(len(rest), rng.randint(0 if first else 1, max_inputs - len(first))))
        table[f"get_{d}"] = {"output": d, "inputs": sorted(chosen)}

    calls = tuple({"tool": f"get_{core[i]}", "args": {x: values[x] for x in sorted(inputs[i])}} for i in range(n_core))
    true_preds = tuple(tuple(sorted(j for j in range(i) if core[j] in inputs[i])) for i in range(n_core))
    tools = [{"name": name, "params": spec["inputs"], "description": f"Returns {spec['output']}; needs "
              f"{', '.join(spec['inputs'])}."} for name, spec in table.items()]  # fmt: skip
    stated = ", ".join(f"{x} = {values[x]}" for x in sorted(given + extra, key=lambda s: int(s[1:])))
    request = f"Known values: {stated}. Find the value of {core[-1]} and submit it with finish."
    return ToyTask(
        task_id=task_id or f"toy-fn-{seed}",
        kind="functions",
        seed=seed,
        request=request,
        tools=tuple(sorted(tools + [FINISH_TOOL], key=lambda t: t["name"])),
        contracts={name: {"reads": [], "writes": [], "volatile": ["request_id"]} for name in sorted(table)},
        calls=calls,
        true_preds=true_preds,
        answer=values[core[-1]],
        world={"functions": table, "values": values},
    )


# ----------------------------------------------------------------------------- stateful tasks
def stateful_task(seed: int, *, n_orders: int = 3, n_accounts: int = 2, task_id: str | None = None) -> ToyTask:
    """Pay each order from its account, then report the balance of one paying account."""
    if n_orders < 1 or n_accounts < 1:
        raise ValueError("need at least one order and one account")
    rng = random.Random(f"toy-stateful-{seed}")
    accounts = sorted(rng.sample(range(1, 50), n_accounts))
    orders = sorted(rng.sample(range(50, 100), n_orders))  # ids never collide with accounts or amounts
    amounts = dict(zip(orders, rng.sample(range(100, 1000), n_orders), strict=True))
    payer = {o: rng.choice(accounts) for o in orders}
    balances = {a: sum(amounts[o] for o in orders if payer[o] == a) + rng.randrange(1000, 10000) for a in accounts}
    report = rng.choice(sorted(set(payer.values())))
    final = balances[report] - sum(amounts[o] for o in orders if payer[o] == report)

    calls = [{"tool": "get_order", "args": {"order": o}} for o in orders]
    calls += [{"tool": "pay", "args": {"account": payer[o], "amount": amounts[o], "order": o}} for o in orders]
    calls += [{"tool": "get_balance", "args": {"account": report}}]
    n = len(orders)
    true_preds = [()] * n  # reads of different orders commute
    true_preds += [(i,) for i in range(n)]  # pay(o) needs get_order(o): its amount, and o's status changes
    true_preds += [tuple(n + i for i, o in enumerate(orders) if payer[o] == report)]  # the balance after the payments
    tools = [
        {"name": "get_balance", "params": ["account"], "description": "Current balance of an account."},
        {"name": "get_order", "params": ["order"], "description": "Amount and status of an order."},
        {"name": "pay", "params": ["account", "order", "amount"], "description": "Pay an open order in full."},
    ]
    clauses = ", ".join(f"order {o} from account {payer[o]}" for o in orders)
    return ToyTask(
        task_id=task_id or f"toy-st-{seed}",
        kind="stateful",
        seed=seed,
        request=f"Pay {clauses}. Then report the balance of account {report}.",
        tools=tuple(sorted(tools + [FINISH_TOOL], key=lambda t: t["name"])),
        contracts={
            "get_balance": {"reads": ["acct:{account}"], "writes": []},
            "get_order": {"reads": ["order:{order}"], "writes": []},
            "pay": {"reads": ["acct:{account}", "order:{order}"], "writes": ["acct:{account}", "order:{order}"]},
        },
        calls=tuple(calls),
        true_preds=tuple(true_preds),
        answer=final,
        world={
            "balances": balances,
            "orders": {o: {"amount": amounts[o], "status": "open"} for o in orders},
            "targets": orders,
            "report": report,
        },  # fmt: skip
    )


# ----------------------------------------------------------------------------- the environment
def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


class ToyEnv:
    """One episode of a toy task; `step` never raises on model input, it returns an error observation."""

    def __init__(self, toy: ToyTask):
        self.toy = toy
        self.task_id = toy.task_id
        self.contract_source = "by construction" + (" (pure functions)" if toy.kind == "functions" else "")
        self.reset()

    def reset(self) -> None:
        self._state = copy.deepcopy(self.toy.world)
        self._counter = 0

    def task(self) -> dict[str, Any]:
        return self.toy.public()

    def contracts(self) -> dict[str, Any]:
        return copy.deepcopy(self.toy.contracts)

    def step_budget(self) -> int:
        return 2 * len(self.toy.calls)  # twice the minimum number of calls, as FuncBenchGen caps them

    def snapshot(self) -> Any:
        return copy.deepcopy((self._state, self._counter))

    def restore(self, snap: Any) -> None:
        self._state, self._counter = copy.deepcopy(snap)

    def step(self, tool: str, args: Mapping[str, JSON]) -> JSON:
        self._counter += 1
        try:
            canon(args)
        except (TypeError, ValueError) as e:  # NaN, infinities, colliding keys: the agent should not get here (D-020b)
            obs: dict[str, JSON] = {"error": f"arguments are not canonical JSON: {e}"}
        else:
            obs = self._function(tool, args) if self.toy.kind == "functions" else self._stateful(tool, args)
        return {**obs, "request_id": f"req-{self._counter:04d}"} if self.toy.kind == "functions" else obs

    def _function(self, tool: str, args: Mapping[str, JSON]) -> dict[str, JSON]:
        spec = self._state["functions"].get(tool)
        if spec is None:
            return {"error": f"unknown tool {tool}"}
        if sorted(args) != spec["inputs"]:
            return {"error": f"{tool} takes exactly the arguments {', '.join(spec['inputs'])}"}
        values = self._state["values"]
        given = {x: canon(args[x]) for x in spec["inputs"]}
        if all(_is_int(given[x]) and given[x] == values[x] for x in spec["inputs"]):
            return {spec["output"]: values[spec["output"]]}
        rng = random.Random(f"toy-wrong-{self.toy.seed}-{tool}-{canon_json(given)}")  # seeded, input-dependent
        wrong = rng.choice([v for v in range(100, 1000) if v != values[spec["output"]]])
        return {spec["output"]: wrong}

    def _stateful(self, tool: str, args: Mapping[str, JSON]) -> dict[str, JSON]:
        balances, orders = self._state["balances"], self._state["orders"]
        params = {"get_balance": ["account"], "get_order": ["order"], "pay": ["account", "amount", "order"]}
        if tool not in params:
            return {"error": f"unknown tool {tool}"}
        if sorted(args) != params[tool] or not all(_is_int(canon(v)) for v in args.values()):
            return {"error": f"{tool} takes integer arguments {', '.join(params[tool])}"}
        a = {k: canon(v) for k, v in args.items()}
        if "account" in a and a["account"] not in balances:
            return {"error": f"no account {a['account']}"}
        if "order" in a and a["order"] not in orders:
            return {"error": f"no order {a['order']}"}
        if tool == "get_balance":
            return {"balance": balances[a["account"]]}
        if tool == "get_order":
            return dict(orders[a["order"]])
        order = orders[a["order"]]
        if order["status"] != "open":
            return {"error": f"order {a['order']} is already {order['status']}"}
        if a["amount"] != order["amount"]:
            return {"error": f"order {a['order']} must be paid in full"}
        if balances[a["account"]] < a["amount"]:
            return {"error": f"insufficient funds in account {a['account']}"}
        balances[a["account"]] -= a["amount"]
        order["status"] = "paid"
        return {"receipt": f"R-{a['order']}", "status": "paid"}

    def check(self, answer: JSON) -> bool:
        try:
            answer = canon(answer)
        except (TypeError, ValueError):
            return False
        ok = _is_int(answer) and answer == self.toy.answer
        if self.toy.kind == "stateful":
            ok = ok and all(self._state["orders"][o]["status"] == "paid" for o in self._state["targets"])
        return ok

    def final_state(self) -> JSON:
        if self.toy.kind == "functions":
            return {}
        return {"balances": {str(a): b for a, b in self._state["balances"].items()},
                "orders": {str(o): v["status"] for o, v in self._state["orders"].items()}}  # fmt: skip


# ----------------------------------------------------------------------------- demonstrations (D-007)
def demo_order(toy: ToyTask, seed: int = 0) -> list[int]:
    """A seeded random topological order of the true DAG: a demonstrator's accidental order (D-007)."""
    rng = random.Random(f"toy-demo-{toy.task_id}-{seed}")
    return random_topological_order(node_masks(len(toy.calls), toy.true_preds), rng)


def demonstration(toy: ToyTask, seed: int = 0) -> tuple[list[dict[str, Any]], list[int]]:
    """The executed demonstration (steps with observations, then finish) and its order of the gold calls."""
    order = demo_order(toy, seed)
    steps = run_calls(ToyEnv(toy), [toy.calls[i] for i in order])
    return steps + [{"tool": "finish", "args": {"answer": toy.answer}}], order


def compile_toy(toy: ToyTask, seed: int = 0) -> tuple[GraphRecord, list[list[int]]]:
    """Compile the demonstration (Compile sees only the trace) and give the true DAG by demo position."""
    demo, order = demonstration(toy, seed)
    record = compile_demo(toy.public(), demo, toy.contracts, task_id=toy.task_id)
    position = {call: i for i, call in enumerate(order)}
    true_preds = [sorted(position[u] for u in toy.true_preds[call]) for call in order]
    return record, true_preds


def make_tasks(kind: str, seeds: Sequence[int], **kwargs: Any) -> list[ToyTask]:
    make = {"functions": function_task, "stateful": stateful_task}[kind]
    return [make(s, **kwargs) for s in seeds]

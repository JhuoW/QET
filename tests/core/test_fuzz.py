"""Differential and invariance tests on richer random inputs than the oracle's `random_demo` (E0.3-E0.6 review):
templated, glob and missing-argument resources, null and absent contracts, unknown tools, repeated calls, empty /
error / null / list observations, NFC and NFD strings, non-identifier keys, nested values, task orders, start
values, initial observations, truncation, and graphs above S_MAX (sub-lattice)."""

import copy
import random

import qet_ref as R

from qet.core.compile import canonical_order, compile_edges, predecessor_sets
from qet.core.lattice import END, build_lattice, sample_orders, sublattice_seed
from qet.core.reduce import name_calls, reduce
from qet.core.render import render
from qet.core.weights import summarize

KEYS = ["id", "x", "a b", "1st", "é", 'q"', "meta", "", "a.b"]  # one normal form per key (NFC collisions are errors)
STRINGS = ["NYC", "SG", "ord-7", "é", "é", "", "a*b", "x y", "1200", "true"]
RESOURCES = ["acct:{id}", "acct:*", "order:{x}", "r1", "*", "mail", "acct:7", "order:{missing}"]
REQUESTS = ["Book NYC to SG for 1200 and 7 nights.", "Do it.", "Use é and é, ord-7, 2.5 and true.", "1,200 x y a*b"]
TRUNCATIONS = [None, {"max_list_items": 1}, {"max_str_chars": 2}, {"max_list_items": 0, "max_str_chars": 0}]


def value(rng, depth=0):
    r = rng.random()
    if depth < 2 and r < 0.1:
        return [value(rng, depth + 1) for _ in range(rng.randint(0, 3))]
    if depth < 2 and r < 0.18:
        return {rng.choice(KEYS): value(rng, depth + 1) for _ in range(rng.randint(0, 3))}
    r = rng.random()
    if r < 0.3:
        return rng.choice([1, 2, 7, 1200, 3, -1, 0])
    if r < 0.4:
        return rng.choice([1.0, 1200.0, 2.5, -0.0, 0.1])
    if r < 0.5:
        return rng.choice([True, False, None])
    return rng.choice(STRINGS)


def observation(rng):
    r = rng.random()
    if r < 0.3:
        return rng.choice([{}, "error: 503 Service Unavailable", None, []])
    obs = {rng.choice(KEYS): value(rng) for _ in range(rng.randint(1, 3))}
    if rng.random() < 0.5:
        obs["ts"] = rng.randint(0, 10**6)
    if rng.random() < 0.3:
        obs["meta"] = {"ts": f"T{rng.randint(0, 99)}", "page": rng.randint(1, 3)}
    return obs


def random_case(seed, n_max=8):
    rng = random.Random(seed)
    tools = [f"t{i}" for i in range(rng.randint(1, 6))] + ["mystery"]  # "mystery" never has a contract
    contracts = {}
    for t in tools[:-1]:
        k = rng.random()
        if k < 0.1:
            continue
        if k < 0.15:
            contracts[t] = None
            continue
        c = {"reads": rng.sample(RESOURCES, rng.randint(0, 2))}
        c["writes"] = rng.sample(RESOURCES, rng.randint(0, 2)) if rng.random() < 0.35 else []
        if rng.random() < 0.5:
            c["volatile"] = rng.sample(["ts", "meta.ts", "request_id"], rng.randint(1, 2))
        contracts[t] = c
    steps, produced = [], []
    for _ in range(rng.randint(0, n_max)):
        if steps and rng.random() < 0.15:
            steps.append(copy.deepcopy(rng.choice(steps)))  # an identical repeated call
            continue
        args = {}
        for _ in range(rng.randint(0, 3)):
            args[rng.choice(KEYS)] = rng.choice(produced) if produced and rng.random() < 0.4 else value(rng)
        obs = observation(rng)
        produced += list(R.scalar_leaves(R.canon_value(obs)))
        steps.append({"tool": rng.choice(tools), "args": args, "obs": obs})
    n, kw = len(steps), {}
    if n >= 2 and rng.random() < 0.3:
        kw["task_order"] = sorted({tuple(sorted(rng.sample(range(n), 2))) for _ in range(rng.randint(1, 2))})
    if rng.random() < 0.3:
        kw["start_values"] = [v for v in (value(rng), value(rng)) if not isinstance(v, (list, dict))]
    if rng.random() < 0.3:
        kw["initial_observation"] = {"inbox": [value(rng) for _ in range(2)]}
    task = {
        "request": rng.choice(REQUESTS),
        "tools": [{"name": t, "params": ["id", "x"], "description": t} for t in tools],
    }
    return rng, task, steps, contracts, kw


def test_compile_reduce_and_render_match_the_oracle_on_rich_demos():
    labels = set()
    for seed in range(800):
        rng, task, steps, contracts, kw = random_case(seed)
        request, n = task["request"], len(steps)
        edges = compile_edges(request, steps, contracts, **kw)
        expected = R.compile_edges(request, steps, contracts, **kw)
        assert [[e.u, e.v, list(e.labels)] for e in edges] == expected, seed
        labels.update(label for e in edges for label in e.labels)
        keys = [key for _, key in name_calls((s["tool"], s["args"]) for s in steps)]
        assert keys == R.node_keys(steps)
        assert canonical_order(predecessor_sets(n, edges), keys) == R.canonical_order(
            R.preds_from_edges(n, expected), keys
        )
        history = rng.sample(steps, n)  # any history, legal or not
        got = [(e.name, e.obs, e.key) for e in reduce(history, contracts)]
        assert got == [(e["name"], e["obs"], e["key"]) for e in R.reduce_history(history, contracts)], seed
        trunc = rng.choice(TRUNCATIONS)
        for variant in ("quotient", "chronological"):
            p = render(task, history, contracts, variant, trunc)
            assert {"system": p.system, "user": p.user} == R.render(task, history, contracts, variant, trunc), seed
    assert labels == {"data", "effect", "decision", "dup", "task"}


def test_render_invariance_on_rich_demos():
    incomparable = 0
    for seed in range(300):
        rng, task, steps, contracts, kw = random_case(10_000 + seed)
        n = len(steps)
        preds = predecessor_sets(n, compile_edges(task["request"], steps, contracts, **kw))
        trunc = rng.choice(TRUNCATIONS)
        by_set, orders = {}, set()
        for _ in range(10):
            order = R.random_topological_order(preds, n, rng)
            orders.add(tuple(order))
            for k in range(n + 1):
                done = frozenset(order[:k])
                if done not in by_set:
                    demo_history = [steps[i] for i in range(n) if i in done]
                    by_set[done] = render(task, demo_history, contracts, truncation=trunc)
                    expected = R.render(task, demo_history, contracts, "quotient", trunc)
                    assert {"system": by_set[done].system, "user": by_set[done].user} == expected
                assert render(task, [steps[i] for i in order[:k]], contracts, truncation=trunc) == by_set[done]
        incomparable += len(orders) > 1
    assert incomparable > 50  # the invariance check is not trivial: many graphs have several legal orders


def test_sublattices_above_s_max_match_the_oracle():
    checked = 0
    for seed in range(150):
        rng = random.Random(seed)
        n = rng.randint(9, 11)
        preds = R.random_dag(n, rng.choice([0.0, 0.05, 0.1]), rng)
        if R.enumerate_states(preds, n, cap=256) is not None:
            continue  # small enough for the exact lattice
        k, gid = rng.choice([2, 4, 8]), f"graph-{seed}"
        lat = build_lattice(n, [sorted(p) for p in preds], k=k, seed=sublattice_seed(gid, 0, 1))
        assert not lat.exact
        orders = sample_orders(n, [sorted(p) for p in preds], k, sublattice_seed(gid, 0, 1))
        states = [frozenset(v for v in range(n) if st >> v & 1) for st in lat.states]
        assert set(states) == set(R.sublattice_states(orders))
        olp = R.random_policy(states, preds, n, rng)  # also scores ready actions that leave S (for B+)
        lp = {(sum(1 << v for v in st), END if v == "end" else v): x for (st, v), x in olp.items()}
        runs = [("qet", 0.5, {}), ("uniform_aug", 0.5, {}), ("frontier", 0.5, {}),
                ("path_sft", 0.5, {"order": orders[0]}), ("sampled_paths", 0.0, {"paths": orders})]  # fmt: skip
        for method, lam, kw in runs:
            got = summarize(method, lat, lp, lam=lam, plus=True, **kw)
            exp = R.summary(method, states=states, preds=preds, n=n, logp=olp, lam=lam, plus=True, **kw)
            for st, x in exp["logB"].items():
                assert abs(got.log_b[sum(1 << v for v in st)] - x) <= 1e-9 * max(1.0, abs(x))
            for st, x in exp["logB_plus"].items():
                assert abs(got.log_b_plus[sum(1 << v for v in st)] - x) <= 1e-9 * max(1.0, abs(x))
            for (st, v), x in exp["weights"].items():
                assert abs(got.weights[(sum(1 << u for u in st), END if v == "end" else v)] - x) <= 1e-9
            for name in ("qet_objective", "surrogate", "objective"):
                assert abs(getattr(got, name) - exp[name]) <= 1e-9 * max(1.0, abs(exp[name])), (seed, method, name)
        checked += 1
    assert checked >= 100

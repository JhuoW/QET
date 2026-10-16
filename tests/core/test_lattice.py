"""Lattice, start distribution, completion probabilities and configuration weights (SPEC §6, §8) against the
method page's numbers and, on random DAGs, against the oracle."""

import math
import random

import pytest
import qet_ref as R

from qet.core.lattice import (
    END,
    S_MAX,
    Lattice,
    build_lattice,
    enumerate_states,
    full_lattice,
    lattice_from_states,
    log_completion,
    node_masks,
    random_topological_order,
    sample_orders,
    start_weights,
    state_key,
    state_of,
    sublattice,
    sublattice_seed,
)
from qet.core.weights import completion_counts, edge_weights, summarize

S = frozenset
RUN_PREDS = [[], [], [0, 1]]
DOC_PI = {(S(), 0): 0.60, (S(), 1): 0.30, (S({0}), 1): 0.50, (S({1}), 0): 0.90, (S({0, 1}), 2): 0.80,
          (S({0, 1, 2}), "end"): 0.95}  # fmt: skip


def mask(fs):
    return sum(1 << v for v in fs)


def convert(lp):
    return {(mask(st), END if v == "end" else v): x for (st, v), x in lp.items()}


def random_instance(seed, n_max=8):
    rng = random.Random(seed)
    n = rng.randint(0, n_max)
    preds = R.random_dag(n, rng.choice([0.0, 0.15, 0.3, 0.6]), rng)
    lp = R.random_policy(R.enumerate_states(preds, n), preds, n, rng)
    return rng, n, preds, lp


def doc_lattice():
    return full_lattice(3, RUN_PREDS), convert({k: math.log(v) for k, v in DOC_PI.items()})


# ----------------------------------------------------------------------------- the method page's running example
def test_running_example_completion_probabilities_objective_and_weights():
    lat, lp = doc_lattice()
    s = summarize("qet", lat, lp, lam=0.5)
    B = {state_key(st): math.exp(x) for st, x in s.log_b.items()}
    assert B == pytest.approx({"": 0.4332, "0": 0.38, "1": 0.684, "0,1": 0.76, "0,1,2": 0.95}, abs=1e-12)
    assert [s.mu[mask(st)] for st in (S(), S({0}), S({1}), S({0, 1}), S({0, 1, 2}))] == [0.5, 0.125, 0.125, 0.25, 0.0]
    assert s.objective == pytest.approx(0.6553, abs=1e-4) and s.objective == s.qet_objective
    w = {(state_key(st), v): x for (st, v), x in s.weights.items()}
    assert w == pytest.approx(
        {("", 0): 0.263158, ("", 1): 0.236842, ("0", 1): 0.388158, ("1", 0): 0.361842, ("0,1", 2): 1.0,
         ("0,1,2", END): 1.0}, abs=1e-6)  # fmt: skip
    assert w[("", 0)] / (w[("", 0)] + w[("", 1)]) == pytest.approx(0.526, abs=1e-3)


def test_chain_with_lambda_zero_loss_is_sft():
    lat = full_lattice(3, [[], [0], [1]])
    lp = {(0, 0): math.log(0.5), (1, 1): math.log(0.25), (3, 2): math.log(0.8), (7, END): math.log(0.9)}
    s = summarize("qet", lat, lp, lam=0.0)
    assert s.objective == pytest.approx(-sum(lp.values()), abs=1e-15)
    assert all(w == 1.0 for w in s.weights.values())


# ----------------------------------------------------------------------------- states and mu
def test_state_counts():
    for n in range(7):
        assert len(enumerate_states(node_masks(n, [[j - 1] if j else [] for j in range(n)]), None)) == n + 1
        assert len(enumerate_states(node_masks(n, [[]] * n), None)) == 2**n
    two_chains = [[], [0], [], [2], [3]]  # lengths 2 and 3
    assert len(enumerate_states(node_masks(5, two_chains), None)) == 3 * 4


def test_enumeration_stops_above_the_cap_and_sorts_states():
    m = node_masks(9, [[]] * 9)
    assert enumerate_states(m, S_MAX) is None
    assert len(enumerate_states(node_masks(8, [[]] * 8), S_MAX)) == 256
    st = enumerate_states(node_masks(3, RUN_PREDS), None)
    assert [state_key(s) for s in st] == ["", "0", "1", "0,1", "0,1,2"]


def test_node_masks_validation():
    for bad in ([[1], [0]], [[0]], [[3], []], [[True], []]):
        with pytest.raises(ValueError):
            node_masks(len(bad), bad)
    with pytest.raises(ValueError):
        node_masks(2, [[]])


def test_mu_special_cases():
    for lam in (0.0, 0.5, 0.9):
        assert start_weights(full_lattice(0, []), lam) == {0: 1.0}
        assert start_weights(full_lattice(1, [[]]), lam) == {0: 1.0, 1: 0.0}
    with pytest.raises(ValueError):
        start_weights(full_lattice(2, [[], []]), 1.0)
    with pytest.raises(ValueError):
        start_weights(full_lattice(2, [[], []]), -0.1)


def test_no_tool_calls_loss_is_minus_log_pi_end():
    lat = full_lattice(0, [])
    s = summarize("qet", lat, {(0, END): math.log(0.7)}, lam=0.5)
    assert s.objective == pytest.approx(-math.log(0.7)) and s.weights == {(0, END): 1.0}


def test_mu_counts_layers_inside_a_sublattice():
    lat = sublattice(3, [[], [], []], [[0, 1, 2], [2, 1, 0]])
    mu = start_weights(lat, 0.5)
    assert math.fsum(mu.values()) == pytest.approx(1.0)
    assert mu[mask({0})] == mu[mask({2})] == pytest.approx(0.5 / 2 / 2)
    assert mu[mask({0, 1})] == mu[mask({1, 2})] == pytest.approx(0.5 / 2 / 2)


# ----------------------------------------------------------------------------- lattice validation and sub-lattices
def test_lattice_validation():
    preds = node_masks(2, [[], []])
    for states in ([1, 2, 3], [0, 1, 2], [0, 3], [0, 1, 2, 3, 4]):
        with pytest.raises(ValueError):
            Lattice(2, preds, tuple(sorted(states, key=lambda s: (s.bit_count(), s))))
    with pytest.raises(ValueError):
        lattice_from_states(2, [[], [0]], [0, 2, 3])  # {1} is not a downset when 0 -> 1
    with pytest.raises(ValueError):
        lattice_from_states(3, [[], [], []], [0, 1, 3, 4, 7])  # {2} has no successor inside S
    assert full_lattice(2, [[], []]).exact
    assert not sublattice(2, [[], []], [[0, 1]]).exact
    assert sublattice(2, [[], []], [[0, 1], [1, 0]]).exact


def test_sample_orders_and_seeds():
    preds = [[], [], [], [0]]
    orders = sample_orders(4, preds, 5, seed=123)
    assert orders[0] == [0, 1, 2, 3] and len(orders) == 5
    assert orders == sample_orders(4, preds, 5, seed=123)
    m = node_masks(4, preds)
    for o in orders[1:]:
        assert sorted(o) == [0, 1, 2, 3] and o.index(0) < o.index(3)
    assert sample_orders(4, preds, 3, seed=1, demo_order=[2, 0, 3, 1])[0] == [2, 0, 3, 1]
    with pytest.raises(ValueError):
        sample_orders(4, preds, 2, seed=1, demo_order=[3, 0, 1, 2])
    with pytest.raises(ValueError):
        sample_orders(4, preds, 0, seed=1)
    for seed in range(50):  # the same choice rule as the oracle: uniform over the ascending ready list
        assert random_topological_order(m, random.Random(seed)) == R.random_topological_order(
            [S(p) for p in preds], 4, random.Random(seed)
        )


def test_sublattice_seed_is_stable_and_shared():
    assert sublattice_seed("funcbench-000137", 2, 7) == sublattice_seed("funcbench-000137", 2, 7)
    assert sublattice_seed("funcbench-000137", 2, 7) != sublattice_seed("funcbench-000137", 3, 7)
    assert sublattice_seed("g", 0, 0) == 15601046116942758163  # pinned: changing the scheme changes every run


def test_build_lattice_switches_to_the_sublattice_above_s_max():
    exact = build_lattice(8, [[]] * 8, k=4, seed=0)
    assert exact.exact and len(exact.states) == 256
    sub = build_lattice(9, [[]] * 9, k=4, seed=0)
    assert not sub.exact and len(sub.states) <= 4 * 9 + 1
    assert sub.states == build_lattice(9, [[]] * 9, k=4, seed=0).states
    assert all(state_of(range(j), 9) in sub.succ for j in range(10))  # the demo order is inside S


def test_sublattice_bounds_on_the_seven_step_graph():
    rng = random.Random(7)
    preds = [[], [], [], [], [0, 1, 2, 3], [2, 4], [0, 1, 2, 3, 4, 5]]
    ostates = R.enumerate_states([S(p) for p in preds], 7)
    lp = convert(R.random_policy(ostates, [S(p) for p in preds], 7, rng))
    full = full_lattice(7, preds)
    sub = sublattice(7, preds, sample_orders(7, preds, 3, seed=5))
    exact_b = log_completion(full, lp)
    bs, bp = log_completion(sub, lp), log_completion(sub, lp, plus=True)
    assert all(bs[s] <= exact_b[s] + 1e-12 and exact_b[s] <= bp[s] + 1e-12 for s in sub.states)
    assert len(full.states) == 19


def test_completion_counts():
    c = completion_counts(full_lattice(3, RUN_PREDS))
    assert c[0] == 2 and c[mask({0})] == 1 and c[mask({0, 1, 2})] == 1
    assert completion_counts(full_lattice(5, [[]] * 5))[0] == math.factorial(5)


def test_method_argument_errors():
    lat, lp = doc_lattice()
    with pytest.raises(ValueError):
        edge_weights("path_sft", lat)
    with pytest.raises(ValueError):
        edge_weights("path_sft", lat, order=[2, 0, 1])
    with pytest.raises(ValueError):
        edge_weights("qet", lat)
    with pytest.raises(ValueError):
        edge_weights("sorted_sft", lat, lp)
    with pytest.raises(KeyError):
        edge_weights("qet", lat, {k: v for k, v in lp.items() if k != (0, 0)})
    with pytest.raises(ValueError):
        edge_weights("qet", lat, {**lp, (0, 0): math.nan})
    sub = sublattice(3, RUN_PREDS, [[0, 1, 2]])
    with pytest.raises(ValueError):
        edge_weights("path_sft", sub, order=[1, 0, 2])  # leaves the states in use


def test_zero_probability_edges():
    lat, lp = doc_lattice()
    lp = {**lp, (0, 1): -math.inf}  # pi(B | empty) = 0: the root only goes through A
    w = edge_weights("qet", lat, lp, lam=0.0).weights
    assert w[(0, 1)] == 0.0 and w[(0, 0)] == pytest.approx(1.0)
    assert w[(mask({1}), 0)] == 0.0  # {B} is never reached when lambda = 0
    dead = {**lp, (mask({0, 1}), 2): -math.inf}  # nobody can finish: B = 0 everywhere
    with pytest.raises(ValueError):
        edge_weights("qet", lat, dead, lam=0.5)


# ----------------------------------------------------------------------------- differential against the oracle
def compare(got, exp, label):
    def close(a, b):
        return a == b or math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)

    for name in ("log_b", "mu"):
        g, e = getattr(got, name), exp["logB" if name == "log_b" else "mu"]
        assert set(g) == {mask(st) for st in e}, label
        assert all(close(g[mask(st)], x) for st, x in e.items()), (label, name)
    ew = {(mask(st), END if v == "end" else v): x for (st, v), x in exp["weights"].items()}
    assert set(got.weights) == set(ew), label
    assert all(close(got.weights[k], x) for k, x in ew.items()), (label, "weights")
    for name in ("qet_objective", "surrogate", "objective"):
        assert close(getattr(got, name), exp[name]), (label, name)
    if exp["logB_plus"] is not None:
        assert all(close(got.log_b_plus[mask(st)], x) for st, x in exp["logB_plus"].items()), (label, "plus")


def test_every_method_matches_the_oracle_on_random_dags():
    for seed in range(300):
        rng, n, preds, olp = random_instance(seed)
        lp = convert(olp)
        ostates = R.enumerate_states(preds, n)
        lat = full_lattice(n, [sorted(p) for p in preds])
        runs = [("qet", 0.5, {}), ("qet", 0.0, {}), ("uniform_aug", 0.5, {}), ("frontier", 0.5, {}),
                ("path_sft", 0.5, {"order": list(range(n))}),
                ("path_sft", 0.0, {"order": R.random_topological_order(preds, n, rng)})]  # fmt: skip
        if n >= 1:
            paths = [list(range(n))] + [R.random_topological_order(preds, n, rng) for _ in range(rng.randint(0, 3))]
            runs.append(("sampled_paths", 0.0, {"paths": paths}))
        for method, lam, kw in runs:
            got = summarize(method, lat, lp, lam=lam, **kw)
            exp = R.summary(method, states=ostates, preds=preds, n=n, logp=olp, lam=lam, **kw)
            compare(got, exp, (seed, method, lam))
        if n >= 2:  # sub-lattice runs
            orders = [list(range(n))] + [R.random_topological_order(preds, n, rng) for _ in range(rng.randint(0, 3))]
            osub = R.sublattice_states(orders)
            sub = sublattice(n, [sorted(p) for p in preds], orders)
            assert set(sub.states) == {mask(st) for st in osub}
            for method in ("qet", "uniform_aug", "frontier"):
                got = summarize(method, sub, lp, lam=0.5, plus=True)
                exp = R.summary(method, states=osub, preds=preds, n=n, logp=olp, lam=0.5, plus=True)
                compare(got, exp, (seed, "sub", method))

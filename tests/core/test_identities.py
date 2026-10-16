"""SPEC §9 items 1-9 on 300 random DAGs (n <= 8) with random raw policies (failure mass included), float64.

Completions are enumerated by brute force here, independently of the lattice recursion. Item 10 (render
invariance) is in test_reduce_render.py.
"""

import math
import random

import pytest
import qet_ref as R

from qet.core.lattice import END, full_lattice, log_completion, members, start_weights, sublattice
from qet.core.weights import edge_weights, summarize

SEEDS = range(300)


def mask(fs):
    return sum(1 << v for v in fs)


def instance(seed):
    rng = random.Random(seed)
    n = rng.randint(0, 8)
    preds = R.random_dag(n, rng.choice([0.0, 0.15, 0.3, 0.6]), rng)
    olp = R.random_policy(R.enumerate_states(preds, n), preds, n, rng)
    lp = {(mask(st), END if v == "end" else v): x for (st, v), x in olp.items()}
    return rng, n, full_lattice(n, [sorted(p) for p in preds]), lp


def completions(lat, lp, state):
    """Log-probabilities of every legal completion from `state` (finish action included), by brute force."""
    out = []

    def rec(s, acc):
        if s == lat.full:
            out.append(acc + lp[(s, END)])
            return
        for v in lat.ready[s]:
            rec(s | 1 << v, acc + lp[(s, v)])

    rec(state, 0.0)
    return out


def neg_l(lat, lp, mu):
    log_b = log_completion(lat, lp)
    return math.fsum(mu[s] * -log_b[s] for s in lat.states if mu[s] > 0)


# 1 ----------------------------------------------------------------------------------------------------------------
def test_1_lattice_b_equals_the_sum_over_all_legal_completions():
    for seed in SEEDS:
        _, _, lat, lp = instance(seed)
        log_b = log_completion(lat, lp)
        for s in lat.states:
            direct = math.fsum(math.exp(x) for x in completions(lat, lp, s))
            assert math.isclose(math.exp(log_b[s]), direct, rel_tol=1e-12), (seed, members(s))


# 2 ----------------------------------------------------------------------------------------------------------------
@pytest.mark.slow
def test_2_weights_are_minus_the_gradient_of_l_by_finite_differences():
    eps = 1e-6
    for seed in SEEDS:
        _, n, lat, lp = instance(seed)
        for lam in (0.0, 0.5):
            mu = start_weights(lat, lam)
            w = edge_weights("qet", lat, lp, lam=lam).weights
            for e in lat.edges:
                up, dn = dict(lp), dict(lp)
                up[e] += eps
                dn[e] -= eps
                grad = (neg_l(lat, up, mu) - neg_l(lat, dn, mu)) / (2 * eps)
                assert abs(-grad - w[e]) < 1e-7, (seed, lam, e)


# 3 ----------------------------------------------------------------------------------------------------------------
def test_3_finish_weight_one_and_tool_weights_sum_to_remaining_work():
    for seed in SEEDS:
        _, n, lat, lp = instance(seed)
        for method, lam in (("uniform_aug", 0.5), ("frontier", 0.5), ("qet", 0.0), ("qet", 0.5)):
            mu = start_weights(lat, lam)
            w = edge_weights(method, lat, lp, lam=lam).weights
            assert w[(lat.full, END)] == pytest.approx(1.0, abs=1e-12)
            tools = math.fsum(x for (s, v), x in w.items() if v != END)
            assert tools == pytest.approx(math.fsum(mu[s] * (n - s.bit_count()) for s in lat.states), abs=1e-12)


# 4 ----------------------------------------------------------------------------------------------------------------
def test_4_surrogate_minus_objective_is_the_posterior_entropy():
    for seed in SEEDS:
        _, _, lat, lp = instance(seed)
        for lam in (0.0, 0.5):
            out = summarize("qet", lat, lp, lam=lam)
            h = 0.0
            for s in lat.states:
                if out.mu[s] > 0:
                    lps = completions(lat, lp, s)
                    h += out.mu[s] * -math.fsum(math.exp(x - out.log_b[s]) * (x - out.log_b[s]) for x in lps)
            assert out.surrogate - out.objective == pytest.approx(h, abs=1e-10), seed


# 5 ----------------------------------------------------------------------------------------------------------------
def test_5_sublattice_bounds():
    for seed in SEEDS:
        rng, n, lat, lp = instance(seed)
        preds = [members(p) for p in lat.preds]
        exact_b = log_completion(lat, lp)
        assert log_completion(lat, lp, plus=True) == exact_b
        orders = [list(range(n))] + [
            R.random_topological_order([frozenset(p) for p in preds], n, rng) for _ in range(rng.randint(0, 3))
        ]
        sub = sublattice(n, preds, orders)
        bs, bp = log_completion(sub, lp), log_completion(sub, lp, plus=True)
        for s in sub.states:
            assert bs[s] <= exact_b[s] + 1e-12 and exact_b[s] <= bp[s] + 1e-12, seed
        assert math.fsum(start_weights(sub, 0.5).values()) == pytest.approx(1.0, abs=1e-12)
        if sub.exact:
            assert all(bs[s] == pytest.approx(exact_b[s], abs=1e-12) == bp[s] for s in sub.states)


# 6 ----------------------------------------------------------------------------------------------------------------
def test_6_augmentation_identity_eq4():
    for seed in SEEDS:
        _, _, lat, lp = instance(seed)
        log_b = log_completion(lat, lp)
        expected_nll = {}
        for s in lat.states:
            lps = completions(lat, lp, s)
            k = len(lps)
            lhs = math.fsum(-x for x in lps) / k
            kl = math.fsum((1 / k) * (math.log(1 / k) - (x - log_b[s])) for x in lps)
            assert lhs == pytest.approx(-log_b[s] + math.log(k) + kl, abs=1e-10), seed
            expected_nll[s] = lhs
        out = summarize("uniform_aug", lat, lp, lam=0.5)  # exact uniform augmentation = E_omega[-log p] per start
        assert out.objective == pytest.approx(math.fsum(out.mu[s] * expected_nll[s] for s in lat.states), abs=1e-10)


# 7 ----------------------------------------------------------------------------------------------------------------
def test_7_local_bound_eq5_and_the_frontier_objective():
    for seed in SEEDS:
        _, _, lat, lp = instance(seed)
        log_b = log_completion(lat, lp)
        bound = {}
        for s in reversed(lat.states):
            if s == lat.full:
                bound[s] = -lp[(s, END)]
                continue
            z = math.fsum(math.exp(lp[(s, v)]) for v in lat.succ[s])
            local = -math.log(z)
            assert (
                -log_b[s]
                <= local + math.fsum(math.exp(lp[(s, v)]) / z * -log_b[s | 1 << v] for v in lat.succ[s]) + 1e-12
            )
            bound[s] = local + math.fsum(math.exp(lp[(s, v)]) / z * bound[s | 1 << v] for v in lat.succ[s])
        out = summarize("frontier", lat, lp, lam=0.5)
        assert out.objective == pytest.approx(math.fsum(out.mu[s] * bound[s] for s in lat.states), abs=1e-10)
        assert out.objective >= out.qet_objective - 1e-12


# 8 ----------------------------------------------------------------------------------------------------------------
def test_8_continuation_bound():
    for seed in SEEDS:
        rng, _, lat, lp = instance(seed)
        mu = start_weights(lat, 0.5)
        support = [s for s in lat.states if mu[s] > 0]
        raw = [rng.random() + 1e-3 for _ in support]
        nu = {s: x / math.fsum(raw) for s, x in zip(support, raw, strict=True)}
        kappa = max(nu[s] / mu[s] for s in support)
        log_b = log_completion(lat, lp)
        lhs = math.fsum(nu[s] * (1 - math.exp(log_b[s])) for s in support)
        assert lhs <= kappa * math.fsum(mu[s] * -log_b[s] for s in support) + 1e-12


# 9 ----------------------------------------------------------------------------------------------------------------
def test_9_on_a_chain_with_lambda_zero_qet_is_demo_order_sft():
    for seed in range(300):
        rng = random.Random(seed)
        n = rng.randint(0, 8)
        preds = [frozenset(range(j)) for j in range(n)]  # the full chain: every earlier node is a predecessor
        lat = full_lattice(n, [sorted(p) for p in preds])
        lp = {
            (mask(st), END if v == "end" else v): x
            for (st, v), x in R.random_policy(R.enumerate_states(preds, n), preds, n, rng).items()
        }
        q = summarize("qet", lat, lp, lam=0.0)
        s = summarize("path_sft", lat, lp, lam=0.0, order=list(range(n)))
        assert q.weights == pytest.approx(s.weights, abs=1e-12)
        assert q.objective == pytest.approx(s.objective, abs=1e-12)

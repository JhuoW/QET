"""Self-tests of the reference oracle: the method page's worked numbers, the identities of SPEC §9, and
fixture freshness. Run with `pytest reference -q`. These tests do not touch src/qet."""
import math
import random
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import qet_ref as R  # noqa: E402

S = frozenset
DOC = {(S(), 0): 0.60, (S(), 1): 0.30, (S({0}), 1): 0.50, (S({1}), 0): 0.90, (S({0, 1}), 2): 0.80,
       (S({0, 1, 2}), "end"): 0.95}
RUN_PREDS = [S(), S(), S({0, 1})]


def logp_of(table):
    return {k: math.log(v) for k, v in table.items()}


def random_instance(seed, n_max=7):
    rng = random.Random(seed)
    n = rng.randint(1, n_max)
    preds = R.random_dag(n, rng.choice([0.0, 0.15, 0.3, 0.6]), rng)
    states = R.enumerate_states(preds, n)
    return rng, n, preds, states, R.random_policy(states, preds, n, rng)


def test_fixtures_are_up_to_date():
    r = subprocess.run([sys.executable, "-I", str(HERE / "make_fixtures.py"), "--check"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_method_page_running_example_numbers():
    st = R.enumerate_states(RUN_PREDS, 3)
    out = R.summary("qet", states=st, preds=RUN_PREDS, n=3, logp=logp_of(DOC), lam=0.5)
    B = {R.skey(I): math.exp(x) for I, x in out["logB"].items()}
    assert B == pytest.approx({"": 0.4332, "0": 0.38, "1": 0.684, "0,1": 0.76, "0,1,2": 0.95}, abs=1e-12)
    assert out["objective"] == pytest.approx(0.65531, abs=1e-5)
    w = {R.ekey(*e): x for e, x in out["weights"].items()}
    assert w == pytest.approx({"|0": 0.26316, "|1": 0.23684, "0|1": 0.38816, "1|0": 0.36184, "0,1|2": 1.0,
                               "0,1,2|end": 1.0}, abs=1e-5)
    assert out["surrogate"] == pytest.approx(1.0012, abs=1e-4)
    assert w["|0"] / (w["|0"] + w["|1"]) == pytest.approx(0.5263, abs=1e-4)


def test_lattice_equals_path_enumeration():
    for seed in range(300):
        _, n, preds, states, lp = random_instance(seed)
        logB = R.lattice_logB(states, preds, n, lp)
        for I in states:
            direct = R.logsumexp([R.path_logp(o, lp, I) for o in R.all_orders(preds, n, I)])
            assert logB[I] == pytest.approx(direct, rel=1e-12, abs=1e-12)


def test_weights_are_the_gradient():
    eps = 1e-6
    for seed in range(60):
        _, n, preds, states, lp = random_instance(seed, n_max=5)
        for lam in (0.0, 0.5):
            w, _ = R.method_weights("qet", states=states, preds=preds, n=n, logp=lp, lam=lam)
            mu = R.start_weights(states, n, lam)

            def L(table):
                lb = R.lattice_logB(states, preds, n, table)
                return sum(mu[I] * -lb[I] for I in states if mu[I] > 0)

            for e in lp:
                up, dn = dict(lp), dict(lp)
                up[e] += eps
                dn[e] -= eps
                assert -(L(up) - L(dn)) / (2 * eps) == pytest.approx(w.get(e, 0.0), abs=1e-7)


def test_flow_identities():
    for seed in range(200):
        _, n, preds, states, lp = random_instance(seed)
        V = R.full(n)
        mu = R.start_weights(states, n, 0.5)
        assert sum(mu.values()) == pytest.approx(1.0)
        remaining = sum(mu[I] * (n - len(I)) for I in states)
        for method in ("qet", "uniform_aug", "frontier"):
            w, _ = R.method_weights(method, states=states, preds=preds, n=n, logp=lp, lam=0.5)
            assert w[(V, "end")] == pytest.approx(1.0)
            assert sum(x for (I, v), x in w.items() if v != "end") == pytest.approx(remaining)


def test_surrogate_equals_objective_plus_posterior_entropy():
    for seed in range(100):
        _, n, preds, states, lp = random_instance(seed, n_max=6)
        out = R.summary("qet", states=states, preds=preds, n=n, logp=lp, lam=0.5)
        H = 0.0
        for I, m in out["mu"].items():
            if m == 0:
                continue
            lps = [R.path_logp(o, lp, I) for o in R.all_orders(preds, n, I)]
            z = R.logsumexp(lps)
            H += m * -sum(math.exp(x - z) * (x - z) for x in lps)
        assert out["surrogate"] - out["objective"] == pytest.approx(H, abs=1e-10)


def test_sublattice_bounds():
    for seed in range(200):
        rng, n, preds, states, lp = random_instance(seed)
        orders = [list(range(n))] + [R.random_topological_order(preds, n, rng) for _ in range(rng.randint(0, 3))]
        sub = R.sublattice_states(orders)
        full = R.lattice_logB(states, preds, n, lp)
        bs = R.lattice_logB(sub, preds, n, lp)
        bp = R.lattice_logB(sub, preds, n, lp, outside=0.0)
        for I in sub:
            assert bs[I] <= full[I] + 1e-12 and full[I] <= bp[I] + 1e-12
        mu = R.start_weights(sub, n, 0.5)
        assert sum(mu.values()) == pytest.approx(1.0)
        exact = R.lattice_logB(states, preds, n, lp, outside=0.0)
        assert all(exact[I] == pytest.approx(full[I]) for I in states)  # S = full lattice: B+ = B


def test_augmentation_identity_eq4():
    for seed in range(100):
        _, n, preds, states, lp = random_instance(seed, n_max=6)
        for I in states:
            orders = list(R.all_orders(preds, n, I))
            lps = [R.path_logp(o, lp, I) for o in orders]
            z = R.logsumexp(lps)
            k = len(orders)
            lhs = sum(-x for x in lps) / k
            kl = sum((1 / k) * (math.log(1 / k) - (x - z)) for x in lps)
            assert lhs == pytest.approx(-z + math.log(k) + kl, abs=1e-10)
        # exact uniform-augmentation weights reproduce E_omega[-log p] from every start state
        out = R.summary("uniform_aug", states=states, preds=preds, n=n, logp=lp, lam=0.5)
        mu = out["mu"]
        direct = 0.0
        for I in states:
            if mu[I] > 0:
                orders = list(R.all_orders(preds, n, I))
                direct += mu[I] * sum(-R.path_logp(o, lp, I) for o in orders) / len(orders)
        assert out["objective"] == pytest.approx(direct, abs=1e-10)


def test_local_bound_eq5_and_frontier_objective():
    for seed in range(150):
        _, n, preds, states, lp = random_instance(seed)
        V = R.full(n)
        logB = R.lattice_logB(states, preds, n, lp)
        U = {}
        for I in sorted(states, key=len, reverse=True):
            if I == V:
                U[I] = -lp[(I, "end")]
                continue
            r = R.ready(I, preds, n)
            z = sum(math.exp(lp[(I, v)]) for v in r)
            U[I] = -math.log(z) + sum(math.exp(lp[(I, v)]) / z * U[I | {v}] for v in r)
            assert -logB[I] <= U[I] + 1e-12
        out = R.summary("frontier", states=states, preds=preds, n=n, logp=lp, lam=0.5)
        assert out["objective"] == pytest.approx(sum(out["mu"][I] * U[I] for I in states), abs=1e-10)
        assert out["objective"] >= out["qet_objective"] - 1e-12


def test_continuation_bound():
    for seed in range(100):
        rng, n, preds, states, lp = random_instance(seed)
        mu = R.start_weights(states, n, 0.5)
        support = [I for I in states if mu[I] > 0]
        raw = [rng.random() for _ in support]
        nu = {I: x / sum(raw) for I, x in zip(support, raw)}
        kappa = max(nu[I] / mu[I] for I in support)
        logB = R.lattice_logB(states, preds, n, lp)
        lhs = sum(nu[I] * (1 - math.exp(logB[I])) for I in support)
        assert lhs <= kappa * sum(mu[I] * -logB[I] for I in support) + 1e-12


def test_chain_with_lambda0_is_sft():
    for n in range(1, 7):
        preds = [S(range(j)) for j in range(n)]
        states = R.enumerate_states(preds, n)
        lp = R.random_policy(states, preds, n, random.Random(n))
        q = R.summary("qet", states=states, preds=preds, n=n, logp=lp, lam=0.0)
        s = R.summary("path_sft", states=states, preds=preds, n=n, logp=lp, lam=0.0, order=list(range(n)))
        assert q["weights"] == pytest.approx(s["weights"])
        assert q["objective"] == pytest.approx(s["objective"])


def test_compile_properties_and_render_invariance():
    task_tools = [{"name": f"tool{t}", "params": ["x"], "description": f"Tool {t}"} for t in range(5)]
    differs = 0
    for seed in range(150):
        rng = random.Random(seed)
        request, steps, contracts = R.random_demo(rng, rng.randint(2, 7))
        n = len(steps)
        edges = R.compile_edges(request, steps, contracts)
        preds = R.preds_from_edges(n, edges)
        eff = [R.effects(s["tool"], s["args"], contracts) for s in steps]
        for u, v, labels in edges:
            assert u < v and set(labels) <= set(R.LABELS)
        for j in range(n):
            for i in range(j):
                if R.conflict_eff(eff[i], eff[j]):
                    assert i in preds[j]
                if eff[j][1]:
                    assert i in preds[j]
        task = {"request": request, "tools": task_tools}
        for _ in range(10):
            order = R.random_topological_order(preds, n, rng)
            for k in range(n + 1):
                I = set(order[:k])
                a = R.render(task, [steps[i] for i in order[:k]], contracts)
                b = R.render(task, [steps[i] for i in range(n) if i in I], contracts)
                assert a == b
                if R.render(task, [steps[i] for i in order[:k]], contracts, "chronological") != \
                        R.render(task, [steps[i] for i in range(n) if i in I], contracts, "chronological"):
                    differs += 1
    assert differs > 0  # the chronological renderer is order-sensitive, as intended


def test_toy_training_matches_method_page():
    hist = R.table_train(RUN_PREDS, 3, {**{k: v for k, v in DOC.items()},
                                        **{(S(), 0): 0.9, (S(), 1): 0.05, (S({0}), 1): 0.9, (S({1}), 0): 0.1,
                                           (S({0, 1}), 2): 0.9}}, "qet", 200, 0.5)
    last = hist[-1]
    assert last["pi"]["1|0"] == pytest.approx(0.956, abs=1e-3)
    assert last["B"]["1"] == pytest.approx(0.947, abs=1e-3)

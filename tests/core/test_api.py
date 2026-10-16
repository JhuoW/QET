"""The test facade qet.core.api (SPEC §10): signatures, JSON-only outputs, and the whole pipeline against the
oracle (Compile -> lattice -> every configuration's weights)."""

import inspect
import json
import math
import random
from pathlib import Path

import pytest
import qet_ref as R

from qet.core import api

FIX = Path(__file__).resolve().parents[2] / "reference" / "fixtures"


def load(name):
    return json.loads((FIX / name).read_text(encoding="utf-8"))


def test_signatures_match_spec_10():
    expected = {
        "canonical_json": ["value"],
        "action_string": ["tool", "args"],
        "show_call": ["tool", "args"],
        "compile_edges": ["case"],
        "canonical_order": ["case"],
        "reduce": ["history", "contracts"],
        "render": ["task", "history", "contracts", "variant", "truncation"],
        "lattice_summary": ["n", "preds", "logp", "method", "lam", "order", "paths", "states", "plus"],
    }
    for name, params in expected.items():
        sig = inspect.signature(getattr(api, name))
        assert list(sig.parameters) == params, name
    render = inspect.signature(api.render).parameters
    assert render["variant"].default == "quotient" and render["truncation"].default is None
    ls = inspect.signature(api.lattice_summary).parameters
    assert [ls[p].default for p in ("order", "paths", "states", "plus")] == [None, None, None, False]


def roundtrip(x):
    assert json.loads(json.dumps(x, allow_nan=False)) == x
    return x


def test_outputs_are_plain_json_on_every_fixture():
    for c in load("canonical.json")["calls"]:
        roundtrip([api.action_string(c["tool"], c["args"]), api.show_call(c["tool"], c["args"])])
    for c in load("compile.json")["cases"]:
        roundtrip(api.compile_edges(c["input"]))
        roundtrip(api.canonical_order(c["input"]))
    for c in load("reduce.json")["cases"]:
        roundtrip(api.reduce(c["history"], c["contracts"]))
    rf = load("render.json")
    for c in rf["cases"]:
        roundtrip(
            api.render(c.get("task", rf["task"]), c["history"], c.get("contracts", rf["contracts"]), c["variant"])
        )
    for case in load("lattice.json")["cases"]:
        for run in case["runs"]:
            out = api.lattice_summary(
                case["n"], case["preds"], case["logp"], run["method"], run["lam"], run.get("order"),
                run.get("paths"), run.get("states"), run.get("plus", False),
            )  # fmt: skip
            roundtrip(out)
            assert set(out) == {"logB", "mu", "weights", "qet_objective", "surrogate", "objective", "logB_plus"}


def test_native_multiturn_through_the_facade():
    rf = load("render.json")
    hist = next(c["history"] for c in rf["cases"] if c["name"] == "quotient:B,A")
    out = roundtrip(api.render(rf["task"], hist, rf["contracts"], "native_multiturn"))
    assert set(out) == {"system", "messages"}
    assert [m["role"] for m in out["messages"]] == ["user", "assistant", "user", "assistant", "user"]
    assert out["messages"][1]["content"] == api.action_string(hist[0]["tool"], hist[0]["args"])


def test_explicit_full_states_equal_the_enumerated_lattice():
    case = next(c for c in load("lattice.json")["cases"] if c["name"] == "seven_step_random_policy")
    full = [sorted(st) for st in R.enumerate_states([frozenset(p) for p in case["preds"]], case["n"])]
    for method in ("qet", "uniform_aug", "frontier"):
        a = api.lattice_summary(case["n"], case["preds"], case["logp"], method, 0.5)
        b = api.lattice_summary(case["n"], case["preds"], case["logp"], method, 0.5, states=full, plus=True)
        assert a["weights"] == b["weights"] and a["objective"] == b["objective"]
        assert b["logB_plus"] == b["logB"]  # S is the full lattice: B+ = B


def test_lattice_summary_refuses_huge_lattices_without_states():
    with pytest.raises(ValueError):
        api.lattice_summary(17, [[]] * 17, {}, "qet", 0.5)


def test_pipeline_matches_the_oracle_on_random_demos():
    for seed in range(150):
        rng = random.Random(seed)
        request, steps, contracts = R.random_demo(rng, rng.randint(0, 7))
        case = {"request": request, "steps": steps, "contracts": contracts}
        edges = api.compile_edges(case)
        assert edges == R.compile_edges(request, steps, contracts)
        n = len(steps)
        preds = R.preds_from_edges(n, edges)
        assert api.canonical_order(case) == R.canonical_order(preds, R.node_keys(steps))
        states = R.enumerate_states(preds, n)
        lp = R.random_policy(states, preds, n, rng)
        logp = {R.ekey(st, v): x for (st, v), x in lp.items()}
        order = api.canonical_order(case)
        for method, lam, kw in (("qet", 0.5, {}), ("qet", 0.0, {}), ("uniform_aug", 0.5, {}), ("frontier", 0.5, {}),
                                ("path_sft", 0.5, {"order": order})):  # fmt: skip
            got = api.lattice_summary(n, [sorted(p) for p in preds], logp, method, lam, **kw)
            exp = R.summary(method, states=states, preds=preds, n=n, logp=lp, lam=lam, **kw)
            for st, x in exp["logB"].items():
                assert math.isclose(got["logB"][R.skey(st)], x, rel_tol=1e-9, abs_tol=1e-12)
            for e, x in exp["weights"].items():
                assert math.isclose(got["weights"][R.ekey(*e)], x, rel_tol=1e-9, abs_tol=1e-12), (seed, method)
            assert math.isclose(got["objective"], exp["objective"], rel_tol=1e-9, abs_tol=1e-12)

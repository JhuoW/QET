"""Fixture fields that reference/acceptance does not compare against src/qet (E0.7 review, note 12):
identities.json (method page eq. 4 and eq. 5), render.json's targets per state, and compile.json's state and
order counts. toy_training.json is checked in E1.3(b)."""

import json
import math
from pathlib import Path

import pytest

from qet.core import api
from qet.core.canonical import action_string
from qet.core.compile import compile_edges, predecessor_sets
from qet.core.lattice import full_lattice, parse_edge_key, state_key
from qet.core.weights import completion_counts, summarize

FIX = Path(__file__).resolve().parents[2] / "reference" / "fixtures"
RUN_PREDS = [[], [], [0, 1]]


def load(name):
    return json.loads((FIX / name).read_text(encoding="utf-8"))


def test_augmentation_identity_eq4_examples():
    lat = full_lattice(3, RUN_PREDS)
    for name, case in load("identities.json")["augmentation_eq4"].items():
        lp = {parse_edge_key(k): v for k, v in case["logp"].items()}
        qet = summarize("qet", lat, lp, lam=0.0).objective
        aug = summarize("uniform_aug", lat, lp, lam=0.0).objective
        assert qet == pytest.approx(case["qet_root_objective"], rel=1e-12), name
        assert aug == pytest.approx(case["uniform_aug_objective"], rel=1e-12), name
        assert aug == pytest.approx(qet + case["entropy_omega"] + case["kl_omega_posterior"], rel=1e-12), name


def test_local_bound_eq5():
    fx = load("identities.json")["local_bound_eq5"]
    doc = next(c for c in load("lattice.json")["cases"] if c["name"] == "running_doc_model")
    out = api.lattice_summary(3, RUN_PREDS, doc["logp"], "qet", 0.0)
    assert -out["logB"][""] == pytest.approx(fx["neglogB_root"], rel=1e-12)
    lp = {parse_edge_key(k): v for k, v in doc["logp"].items()}
    z = math.exp(lp[(0, 0)]) + math.exp(lp[(0, 1)])
    bound = -math.log(z) + (math.exp(lp[(0, 0)]) * -out["logB"]["0"] + math.exp(lp[(0, 1)]) * -out["logB"]["1"]) / z
    assert bound == pytest.approx(fx["local_upper_bound_root"], rel=1e-12)
    assert -out["logB"][""] <= bound


def test_chronological_counterexample():
    """With an order-sensitive prompt, pi(C) after A,B (0.8) differs from pi(C) after B,A (0.6): the lattice, which
    has one prompt per state, gives either merged value, and neither is the true completion probability."""
    cx = load("identities.json")["chronological_counterexample"]
    doc = next(c for c in load("lattice.json")["cases"] if c["name"] == "running_doc_model")
    lp = {parse_edge_key(k): v for k, v in doc["logp"].items()}
    lat = full_lattice(3, RUN_PREDS)
    for name, pi_c in (("merged_with_0.8", cx["pi_C_after_A_then_B"]), ("merged_with_0.6", cx["pi_C_after_B_then_A"])):
        b = math.exp(summarize("qet", lat, {**lp, (3, 2): math.log(pi_c)}, lam=0.0).log_b[0])
        assert b == pytest.approx(cx[name], rel=1e-12), name
        assert b != pytest.approx(cx["true_completion_probability"], rel=1e-6)


def test_targets_of_every_running_example_state():
    rf = load("render.json")
    steps = [c["history"][-1] for c in rf["cases"] if c["name"] in ("quotient:A", "quotient:B")]
    steps.append(next(c for c in rf["cases"] if c["name"] == "quotient:A,B,C")["history"][-1])
    lat = full_lattice(3, RUN_PREDS)
    targets = {}
    for s in lat.states:
        if s == lat.full:
            targets[state_key(s)] = [action_string("finish", {"answer": 1800})]
        else:
            targets[state_key(s)] = [action_string(steps[v]["tool"], steps[v]["args"]) for v in lat.ready[s]]
    assert targets == rf["running_targets_by_state"]


def test_state_and_order_counts_of_every_compile_case():
    for case in load("compile.json")["cases"]:
        inp = case["input"]
        n = len(inp["steps"])
        edges = compile_edges(
            inp["request"], inp["steps"], inp["contracts"], initial_observation=inp.get("initial_observation"),
            task_order=inp.get("task_order", ()), start_values=inp.get("start_values", ()),
        )  # fmt: skip
        lat = full_lattice(n, predecessor_sets(n, edges), cap=None)
        assert len(lat.states) == case["expected_n_states"], case["name"]
        assert completion_counts(lat)[0] == case["expected_n_orders"], case["name"]

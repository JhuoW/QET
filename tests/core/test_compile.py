"""Compile (SPEC §4): edges, available values, canonical order (§8 a2), graph records and the compile report."""

import itertools
import json
import random

import pytest
import qet_ref as R

from qet.core.compile import (
    LABELS,
    GraphRecord,
    StartValues,
    canonical_order,
    compile_demo,
    compile_edges,
    compile_report,
    fallback_to_chain,
    incomparable_pairs,
    predecessor_sets,
    read_records,
    width,
    write_records,
)
from qet.core.contracts import Contracts, conflict
from qet.core.reduce import name_calls

PURE = {"f": {"reads": [], "writes": []}, "g": {"reads": [], "writes": []}, "h": {"reads": [], "writes": []}}


def edges_json(edges):
    return [[e.u, e.v, list(e.labels)] for e in edges]


# ----------------------------------------------------------------------------- values available from the start
def test_request_numbers_are_standalone_numeric_tokens():
    sv = StartValues("Book 3 nights for 1,200 SGD at 4.5 stars; code v2, item x.5, -7 degrees, 2026-10-16.")
    for v in (3, 3.0, 1, 200, 4.5, -7, 2026, 10, 16):
        assert v in sv, v
    for v in (2, 5, 0.5, 1200, 7):
        assert v not in sv, v


def test_request_strings_match_whole_word_sequences_case_sensitively():
    sv = StartValues("Fly from NYC to SG, then email ann@x.org.")
    assert "NYC" in sv and "SG" in sv and "NYC to SG" in sv and "ann@x.org" in sv and "" in sv
    assert "S" not in sv and "sg" not in sv and "NY" not in sv and "to SG then" not in sv


def test_bool_and_null_are_never_request_values():
    sv = StartValues("1 0 true null")
    assert 1 in sv and True not in sv and False not in sv
    assert "true" in sv and None not in sv


def test_initial_observation_leaves_and_declared_values():
    sv = StartValues("Go.", initial_observation={"inbox": [{"id": 42, "flag": True}]}, declared=["EUR", 7.0])
    assert 42 in sv and 42.0 in sv and True in sv and "EUR" in sv and 7 in sv
    assert 1 not in sv


def test_available_values_match_the_oracle_on_random_requests():
    rng = random.Random(0)
    words = ["3", "3.0", "-2", "1,200", "x7", "7x", "4.25", ".5", "NYC", "SG", "to", "é", "é", "a_b", "--", "("]
    values = [3, 3.0, -2, 1, 200, 7, 4.25, 0.5, 5, "NYC", "SG", "NYC to", "é", "a_b", "a", "", True, None, "3"]
    for _ in range(500):
        request = " ".join(rng.choice(words) for _ in range(rng.randint(0, 8)))
        sv = StartValues(request)
        nums = R.request_numbers(request)
        for v in values:
            k = R.vkey(v)
            expected = k in nums or (k[0] == "str" and R.string_in_request(k[1], request))
            assert (v in sv) == expected, (request, v)


# ----------------------------------------------------------------------------- rules
def test_data_edges_from_every_producer_or_every_earlier_call():
    f = {"tool": "f", "args": {}, "obs": {"x": 5}}
    g = {"tool": "g", "args": {}, "obs": {"y": 5.0}}
    assert edges_json(compile_edges("Compute h.", [f, g, {"tool": "h", "args": {"a": 5}, "obs": {}}], PURE)) == [
        [0, 2, ["data"]],
        [1, 2, ["data"]],
    ]
    underived = [f, g, {"tool": "h", "args": {"a": 99}, "obs": {}}]
    assert edges_json(compile_edges("Compute h.", underived, PURE)) == [[0, 2, ["data"]], [1, 2, ["data"]]]
    assert compile_edges("Compute h of 99.", underived, PURE) == []
    assert compile_edges("Compute h.", underived, PURE, start_values=[99]) == []


def test_first_call_with_an_underived_value_has_no_edge():
    assert compile_edges("Go.", [{"tool": "f", "args": {"a": 99}, "obs": {}}], PURE) == []


def test_true_is_a_value_but_never_equals_one():
    steps = [{"tool": "f", "args": {}, "obs": {"ok": True}}, {"tool": "g", "args": {"n": 1}, "obs": {}}]
    assert edges_json(compile_edges("Go 1.", steps, PURE)) == []
    steps[1]["args"] = {"flag": True}
    assert edges_json(compile_edges("Go 1.", steps, PURE)) == [[0, 1, ["data"]]]


def test_volatile_fields_are_not_sources():
    c = {"f": {"reads": [], "writes": [], "volatile": ["ts"]}, "h": {"reads": [], "writes": []}}
    steps = [{"tool": "f", "args": {}, "obs": {"ts": "T1", "id": 1}}, {"tool": "h", "args": {"t": "T1"}, "obs": {}}]
    assert edges_json(compile_edges("Go.", steps, c)) == [[0, 1, ["data"]]]  # underived: every earlier call


def test_effect_decision_dup_and_task_rules():
    c = {
        "read": {"reads": ["r"], "writes": []},
        "write": {"reads": [], "writes": ["r"]},
        "other": {"reads": ["s"], "writes": []},
    }
    steps = [
        {"tool": "read", "args": {}, "obs": {}},
        {"tool": "other", "args": {}, "obs": {}},
        {"tool": "write", "args": {}, "obs": {}},
        {"tool": "read", "args": {}, "obs": {}},
        {"tool": "other", "args": {}, "obs": {}},
    ]
    got = edges_json(compile_edges("Go.", steps, c, task_order=[(1, 4)]))
    assert got == [
        [0, 2, ["decision", "effect"]],
        [0, 3, ["dup"]],
        [1, 2, ["decision"]],
        [1, 4, ["dup", "task"]],
        [2, 3, ["effect"]],
    ]


def test_task_constraints_must_point_forward():
    steps = [{"tool": "f", "args": {}, "obs": {}}, {"tool": "g", "args": {}, "obs": {}}]
    for bad in ([(1, 0)], [(0, 0)], [(0, 2)], [(-1, 1)]):
        with pytest.raises(ValueError):
            compile_edges("Go.", steps, PURE, task_order=bad)


def test_unknown_tools_conflict_with_everything_but_pure_tools():
    c = {"read1": {"reads": ["r1"], "writes": []}, "calc": {"reads": [], "writes": []}}
    steps = [
        {"tool": "read1", "args": {}, "obs": {}},
        {"tool": "mystery", "args": {}, "obs": {}},
        {"tool": "calc", "args": {}, "obs": {}},
    ]
    assert edges_json(compile_edges("Go.", steps, c)) == [[0, 1, ["decision", "effect"]]]


# ----------------------------------------------------------------------------- canonical order (§8 a2)
def test_canonical_order_takes_the_ready_node_with_the_smallest_reduce_key():
    keys = [("b", "{}", 1), ("a", "{}", 1), ("a", "{}", 2), ("c", "{}", 1)]
    assert canonical_order([frozenset(), frozenset(), frozenset(), frozenset()], keys) == [1, 2, 0, 3]
    assert canonical_order([frozenset(), frozenset({0}), frozenset(), frozenset()], keys) == [2, 0, 1, 3]
    with pytest.raises(ValueError):
        canonical_order([frozenset({1}), frozenset({0})], keys[:2])


# ----------------------------------------------------------------------------- differential and property tests
def random_case(seed):
    rng = random.Random(seed)
    request, steps, contracts = R.random_demo(rng, rng.randint(1, 9))
    n = len(steps)
    kw = {}
    if rng.random() < 0.3 and n >= 2:
        kw["task_order"] = sorted({tuple(sorted(rng.sample(range(n), 2))) for _ in range(rng.randint(1, 3))})
    if rng.random() < 0.3:
        kw["start_values"] = [s["obs"]["y"] for s in rng.sample(steps, rng.randint(1, n))]
    if rng.random() < 0.3:
        kw["initial_observation"] = {"seen": [s["obs"]["y"] for s in rng.sample(steps, rng.randint(1, n))]}
    return request, steps, contracts, kw


def test_edges_and_canonical_order_match_the_oracle_on_random_demos():
    for seed in range(600):
        request, steps, contracts, kw = random_case(seed)
        n = len(steps)
        got = compile_edges(request, steps, contracts, **kw)
        assert edges_json(got) == R.compile_edges(request, steps, contracts, **kw), seed
        preds = predecessor_sets(n, got)
        keys = [key for _, key in name_calls((s["tool"], s["args"]) for s in steps)]
        assert keys == R.node_keys(steps)
        assert canonical_order(preds, keys) == R.canonical_order(R.preds_from_edges(n, edges_json(got)), keys)


def test_compiled_graphs_are_conservative_dags_on_random_demos():
    for seed in range(600):
        request, steps, raw, kw = random_case(seed)
        contracts = Contracts.from_json(raw)
        n = len(steps)
        edges = compile_edges(request, steps, contracts, **kw)
        pairs = {(e.u, e.v) for e in edges}
        assert len(pairs) == len(edges)
        for e in edges:
            assert 0 <= e.u < e.v < n  # forward, so the demo order is topological
            assert e.labels and list(e.labels) == sorted(set(e.labels)) and set(e.labels) <= set(LABELS)
        eff = [contracts.effects(s["tool"], s["args"]) for s in steps]
        for v in range(n):
            for u in range(v):
                if conflict(eff[u], eff[v]):
                    assert (u, v) in pairs
                if eff[v].writes:
                    assert (u, v) in pairs
        preds = predecessor_sets(n, edges)
        assert all(u in range(v) for v in range(n) for u in preds[v])
        assert sorted(canonical_order(preds, [(f"t{i}", "{}", 1) for i in range(n)])) == list(range(n))


# ----------------------------------------------------------------------------- width and incomparable pairs
def brute_width(n, preds):
    anc = [set() for _ in range(n)]
    for v in range(n):  # preds point backwards in random_dag, so ancestors are complete when v is reached
        for u in preds[v]:
            anc[v] |= {u} | anc[u]
    best = 0
    for r in range(1, n + 1):
        for subset in itertools.combinations(range(n), r):
            if all(a not in anc[b] and b not in anc[a] for a, b in itertools.combinations(subset, 2)):
                best = r
                break
    return best, sum(1 for a, b in itertools.combinations(range(n), 2) if a not in anc[b] and b not in anc[a])


def test_width_and_incomparable_pairs_match_brute_force():
    rng = random.Random(3)
    for _ in range(300):
        n = rng.randint(0, 8)
        preds = R.random_dag(n, rng.choice([0.0, 0.15, 0.3, 0.6]), rng)
        w, inc = brute_width(n, preds)
        assert width(n, preds) == w
        assert incomparable_pairs(n, preds) == inc


# ----------------------------------------------------------------------------- graph records and report
RUN_TASK = {
    "request": "Find the price of a flight from NYC to SG and the price of 3 nights at a hotel in SG, "
    "then compute the total cost.",
    "tools": [
        {"name": "get_flight_price", "params": ["from", "to"], "description": "Look up a flight price"},
        {"name": "get_hotel_price", "params": ["city", "nights"], "description": "Look up a hotel price"},
        {"name": "calculate_total", "params": ["flight_price", "hotel_price"], "description": "Compute the total"},
        {"name": "finish", "params": ["answer"], "description": "Submit the final answer"},
    ],
}
RUN_CONTRACTS = {
    "get_flight_price": {"reads": ["flights"], "writes": [], "volatile": ["request_id"]},
    "get_hotel_price": {"reads": ["hotels"], "writes": [], "volatile": ["request_id"]},
    "calculate_total": {"reads": [], "writes": []},
}
RUN_DEMO = [
    {"tool": "get_flight_price", "args": {"to": "SG", "from": "NYC"}, "obs": {"price": 1200.0, "request_id": "r-1"}},
    {"tool": "get_hotel_price", "args": {"city": "SG", "nights": 3}, "obs": {"price": 600, "request_id": "r-2"}},
    {"tool": "calculate_total", "args": {"flight_price": 1200, "hotel_price": 600}, "obs": {"total": 1800}},
    {"tool": "finish", "args": {"answer": 1800}},
]


def test_compile_demo_builds_the_graph_record_of_spec_1():
    rec = compile_demo(RUN_TASK, RUN_DEMO, RUN_CONTRACTS, task_id="run-0", snapshot="env@1")
    j = rec.to_json()
    assert list(j) == [
        "task_id",
        "request",
        "tools",
        "initial_observation",
        "snapshot",
        "nodes",
        "edges",
        "final",
        "compile",
    ]
    assert j["nodes"][0] == {
        "id": 0,
        "call": {"tool": "get_flight_price", "args": {"from": "NYC", "to": "SG"}},
        "obs": {"price": 1200},
    }
    assert j["edges"] == [{"from": 0, "to": 2, "why": ["data"]}, {"from": 1, "to": 2, "why": ["data"]}]
    assert j["final"] == {"tool": "finish", "args": {"answer": 1800}}
    assert j["compile"]["fallback_to_chain"] is False
    assert j["compile"]["audit"] == {"orders_replayed": 0, "mismatches": 0}
    assert rec.preds() == [frozenset(), frozenset(), frozenset({0, 1})]
    assert GraphRecord.from_json(json.loads(json.dumps(j))) == rec


def test_compile_demo_requires_a_final_finish_call():
    with pytest.raises(ValueError):
        compile_demo(RUN_TASK, RUN_DEMO[:-1], RUN_CONTRACTS, task_id="x")
    with pytest.raises(ValueError):
        compile_demo(RUN_TASK, [*RUN_DEMO[:2], RUN_DEMO[3], RUN_DEMO[2]], RUN_CONTRACTS, task_id="x")


def test_records_round_trip_through_jsonl(tmp_path):
    rec = compile_demo(RUN_TASK, RUN_DEMO, RUN_CONTRACTS, task_id="run-0")
    path = tmp_path / "graphs.jsonl"
    write_records(path, [rec, fallback_to_chain(rec, orders_replayed=4, mismatches=1)])
    back = read_records(path)
    assert back[0] == rec
    assert edges_json(back[1].edges) == [[0, 1, ["audit"]], [1, 2, ["audit"]]]
    assert back[1].compile.fallback_to_chain and back[1].compile.mismatches == 1


def test_volatile_values_consumed_later_are_reported():
    task = {"request": "Go.", "tools": []}
    c = {"f": {"reads": [], "writes": [], "volatile": ["meta.ts"]}, "h": {"reads": [], "writes": []}}
    demo = [
        {"tool": "f", "args": {}, "obs": {"meta": {"ts": "T1"}, "id": 1}},
        {"tool": "h", "args": {"t": "T1"}, "obs": {}},
        {"tool": "finish", "args": {"answer": 1}},
    ]
    rec = compile_demo(task, demo, c, task_id="v")
    assert rec.compile.volatile_consumed == ((0, 1, "meta.ts"),)


def test_a_step_without_an_observation_is_an_error():
    with pytest.raises(ValueError):
        compile_edges("Go.", [{"tool": "f", "args": {}}], PURE)


def test_declared_start_values_may_be_any_iterable():
    task = {"request": "Go.", "tools": []}
    c = {"f": {"reads": [], "writes": [], "volatile": ["ts"]}, "h": {"reads": [], "writes": []}}
    demo = [
        {"tool": "f", "args": {}, "obs": {"ts": "T1"}},
        {"tool": "h", "args": {"t": "T1"}, "obs": {}},
        {"tool": "finish", "args": {"answer": 1}},
    ]
    from_list = compile_demo(task, demo, c, task_id="v", start_values=["T1"])
    from_generator = compile_demo(task, demo, c, task_id="v", start_values=(v for v in ["T1"]))
    assert from_generator == from_list and from_list.compile.volatile_consumed == () and from_list.edges == ()


def test_compile_report_counts():
    rec = compile_demo(RUN_TASK, RUN_DEMO, RUN_CONTRACTS, task_id="run-0")
    chain = fallback_to_chain(rec, orders_replayed=4, mismatches=1)
    rep = compile_report([rec, chain], contract_source="by construction (test)", s_max=4)
    assert rep["tasks_compiled"] == 2
    assert rep["fallback_share"] == 0.5
    assert rep["audit_failures"] == 1
    assert rep["contract_source"] == "by construction (test)"
    assert rep["label_counts"] == {"audit": 2, "data": 2, "decision": 0, "dup": 0, "effect": 0, "task": 0}
    assert rep["width"] == {"1": 1, "2": 1}
    assert rep["incomparable_pair_share"] == pytest.approx(1 / 6)
    assert rep["states_per_graph"] == {"4": 1, ">4": 1}
    assert rep["volatile_consumed"] == []

"""Reduce (SPEC §5.1), Render (§5.2-§5.5, §5.7) and render invariance (§5.6 a, §9 item 10)."""

import random

import pytest
import qet_ref as R

from qet.core.compile import compile_edges, predecessor_sets
from qet.core.contracts import Contracts
from qet.core.reduce import canonical_entries, reduce, smallest_key_order
from qet.core.render import SYSTEM_TEXT, Truncation, render, render_messages, truncate, value_lines

TASK = {
    "request": "Find the price of a flight from NYC to SG and the price of 3 nights at a hotel in SG.",
    "tools": [
        {"name": "get_hotel_price", "params": ["city", "nights"], "description": "Look up a hotel price"},
        {"name": "get_flight_price", "params": ["from", "to"], "description": "Look up a flight price"},
        {"name": "finish", "params": ["answer"], "description": "Submit the final answer"},
    ],
}
CONTRACTS = {
    "get_flight_price": {"reads": ["flights"], "writes": [], "volatile": ["request_id"]},
    "get_hotel_price": {"reads": ["hotels"], "writes": [], "volatile": ["request_id"]},
}
A = {"tool": "get_flight_price", "args": {"from": "NYC", "to": "SG"}, "obs": {"price": 1200}}
B = {"tool": "get_hotel_price", "args": {"city": "SG", "nights": 3}, "obs": {"price": 600}}


# ----------------------------------------------------------------------------- Reduce
def names(H):
    return [e.name for e in H]


def test_independent_entries_are_sorted_by_key_and_conflicting_ones_keep_their_order():
    c = {
        "update_order": {"reads": ["order:{id}"], "writes": ["order:{id}"]},
        "get_order": {"reads": ["order:{id}"], "writes": []},
        "get_flight_price": {"reads": ["flights"], "writes": []},
    }
    paid = {"tool": "update_order", "args": {"id": 7, "status": "paid"}, "obs": {"ok": True}}
    shipped = {"tool": "update_order", "args": {"id": 7, "status": "shipped"}, "obs": {"ok": True}}
    read = {"tool": "get_order", "args": {"id": 7}, "obs": {"status": "shipped"}}
    assert names(reduce([shipped, A, paid, read], c)) == [
        'get_flight_price(from="NYC", to="SG")',
        'update_order(id=7, status="shipped")',
        'update_order(id=7, status="paid")',
        "get_order(id=7)",
    ]
    assert names(reduce([read, paid], c)) == ["get_order(id=7)", 'update_order(id=7, status="paid")']


def test_repeated_identical_calls_are_numbered_in_execution_order():
    c = {"get_balance": {"reads": ["acct:{account}"], "writes": []}, "pay": {"reads": [], "writes": ["acct:{account}"]}}
    b1 = {"tool": "get_balance", "args": {"account": 1}, "obs": {"balance": 2000}}
    pay = {"tool": "pay", "args": {"account": 1.0}, "obs": {"ok": True}}
    b2 = {"tool": "get_balance", "args": {"account": 1.0}, "obs": {"balance": 800}}
    H = reduce([b1, pay, b2], c)
    assert names(H) == ["get_balance(account=1)", "pay(account=1)", "get_balance(account=1)#2"]
    assert [e.obs for e in H] == [{"balance": 2000}, {"ok": True}, {"balance": 800}]
    assert [e.key for e in H] == [
        ("get_balance", '{"account":1}', 1),
        ("pay", '{"account":1}', 1),
        ("get_balance", '{"account":1}', 2),
    ]


def test_unknown_tools_and_error_observations_are_reduced_too():
    c = {"read1": {"reads": ["r1"], "writes": []}, "calc": {"reads": [], "writes": []}}
    h = [
        {"tool": "read1", "args": {}, "obs": "error: 503"},
        {"tool": "calc", "args": {"a": 1}, "obs": None},
        {"tool": "mystery", "args": {"q": "x"}, "obs": [1, 2]},
        {"tool": "aaa", "args": {}, "obs": {}},
    ]
    H = reduce(h, c)
    # mystery and aaa have no contract: they keep their order relative to every non-pure call
    assert names(H) == ["calc(a=1)", "read1()", 'mystery(q="x")', "aaa()"]
    assert [e.obs for e in H] == [None, "error: 503", [1, 2], {}]


def test_reduce_depends_on_canonical_content_only():
    c = Contracts.from_json(CONTRACTS)
    noisy_a = {
        "tool": "get_flight_price",
        "args": {"to": "SG", "from": "NYC"},
        "obs": {"request_id": "r-9", "price": 1200.0},
    }
    assert reduce([B, noisy_a], c) == reduce([A, B], c)
    assert reduce([], c) == []


def test_canonical_entries_keep_execution_order():
    E = canonical_entries([B, A, B], CONTRACTS)
    assert names(E) == [
        'get_hotel_price(city="SG", nights=3)',
        'get_flight_price(from="NYC", to="SG")',
        'get_hotel_price(city="SG", nights=3)#2',
    ]


def test_repeats_are_numbered_per_identical_call_not_per_display_name():
    # E0.3-E0.6 review counterexample: two different calls that print the same name (D-019)
    c = {"f": {"reads": [], "writes": []}}
    x = {"tool": "f", "args": {"a": 1, "b": 2}, "obs": {"r": 10}}
    y = {"tool": "f", "args": {"a=1, b": 2}, "obs": {"r": 20}}
    task = {"request": "Use 1 and 2.", "tools": [{"name": "f", "params": ["a", "b"], "description": "F"}]}
    assert compile_edges("Use 1 and 2.", [x, y], c) == []  # independent, so both orders are legal
    assert render(task, [x, y], c) == render(task, [y, x], c)
    assert names(reduce([x, y], c)) == names(reduce([y, x], c)) == ["f(a=1, b=2)", "f(a=1, b=2)"]
    assert names(reduce([x, y, x], c)) == ["f(a=1, b=2)", "f(a=1, b=2)#2", "f(a=1, b=2)"]


def test_a_step_without_an_observation_is_an_error():
    with pytest.raises(ValueError):
        reduce([{"tool": "f", "args": {}}], {})
    assert names(reduce([{"tool": "f", "args": {}, "obs": None}], {})) == ["f()"]


def test_smallest_key_order():
    keys = [("b", "", 1), ("a", "", 1), ("c", "", 1)]
    assert smallest_key_order([[], [], []], keys) == [1, 0, 2]
    assert smallest_key_order([[], [0], []], keys) == [0, 1, 2]
    with pytest.raises(ValueError):
        smallest_key_order([[1], [0], []], keys)


def test_reduce_matches_the_oracle_on_random_histories():
    for seed in range(300):
        rng = random.Random(seed)
        _, steps, raw = R.random_demo(rng, rng.randint(0, 9))
        rng.shuffle(steps)  # any history, legal or not
        got = reduce(steps, raw)
        exp = R.reduce_history(steps, raw)
        assert [(e.name, e.obs, e.key) for e in got] == [(e["name"], e["obs"], e["key"]) for e in exp]


# ----------------------------------------------------------------------------- Render
def test_quotient_prompt_layout():
    p = render(TASK, [B, A], CONTRACTS)
    assert p.system == SYSTEM_TEXT
    assert p.user == "\n".join(
        [
            "[Task] " + TASK["request"],
            "[Tools]",
            "  finish(answer): Submit the final answer",
            "  get_flight_price(from, to): Look up a flight price",
            "  get_hotel_price(city, nights): Look up a hotel price",
            "[Completed calls]",
            '  get_flight_price(from="NYC", to="SG") -> {"price":1200}',
            '  get_hotel_price(city="SG", nights=3) -> {"price":600}',
            "[Available values]",
            '  get_flight_price(from="NYC", to="SG").price = 1200',
            '  get_hotel_price(city="SG", nights=3).price = 600',
            "[Next action]",
        ]
    )
    assert p.messages() == [{"role": "system", "content": p.system}, {"role": "user", "content": p.user}]


def test_empty_sections_render_as_none():
    user = render({"request": "Nothing.", "tools": []}, [], {}).user
    assert (
        user == "[Task] Nothing.\n[Tools]\n  (none)\n[Completed calls]\n  (none)\n"
        "[Available values]\n  (none)\n[Next action]"
    )


def test_chronological_prompt_numbers_the_history_in_execution_order():
    user = render(TASK, [B, A], CONTRACTS, variant="chronological").user
    assert '[History]\n  1. get_hotel_price(city="SG", nights=3) -> {"price":600}\n  2. get_flight_price' in user
    assert user.index('get_hotel_price(city="SG", nights=3).price') < user.index(
        'get_flight_price(from="NYC", to="SG").price'
    )
    assert render(TASK, [A, B], CONTRACTS, variant="chronological") != render(
        TASK, [B, A], CONTRACTS, variant="chronological"
    )


def test_native_multiturn_messages():
    msgs = render_messages(
        TASK, [B, {**A, "obs": {"price": 1200.0, "request_id": "r-1"}}], CONTRACTS, variant="native_multiturn"
    )
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert msgs[0]["content"] == SYSTEM_TEXT
    assert msgs[1]["content"] == "\n".join(render(TASK, [], CONTRACTS).user.split("\n")[:5])
    assert msgs[2]["content"] == '{"tool":"get_hotel_price","args":{"city":"SG","nights":3}}'
    assert msgs[3]["content"] == '[Observation] {"price":600}'
    assert msgs[5]["content"] == '[Observation] {"price":1200}'
    assert render_messages(TASK, [], CONTRACTS, variant="native_multiturn") == msgs[:2]
    assert render_messages(TASK, [A], CONTRACTS) == render(TASK, [A], CONTRACTS).messages()


def test_unknown_variant_is_an_error():
    with pytest.raises(ValueError):
        render(TASK, [], CONTRACTS, variant="native_multiturn")  # chat messages only: use render_messages
    with pytest.raises(ValueError):
        render_messages(TASK, [], CONTRACTS, variant="sorted")


def test_value_lines():
    assert value_lines("f()", {"b": [1, {"c d": None}], "a": {}, "e": []}) == [
        "  f().a = {}",
        "  f().b[0] = 1",
        '  f().b[1]["c d"] = null',
        "  f().e = []",
    ]
    assert value_lines("f()", {}) == [] and value_lines("f()", []) == []
    assert value_lines("f()", "error") == ['  f() = "error"']
    assert value_lines("f()", {"_ok1": 1, "1st": 2, "é": 3, 'q"': 4}) == [
        '  f()["1st"] = 2',
        "  f()._ok1 = 1",
        '  f()["q\\""] = 4',
        '  f()["é"] = 3',
    ]


def test_truncation_is_content_based():
    t = Truncation.from_json({"max_list_items": 2, "max_str_chars": 3})
    assert truncate({"a": ["xxxx", "y", "z"], "b": "abc", "c": [[1, 2, 3]]}, t) == {
        "a": ["xxx<+1 chars>", "y", "<+1 more>"],
        "b": "abc",
        "c": [[1, 2, "<+1 more>"]],
    }
    assert truncate(["a"] * 2, Truncation.from_json({"max_list_items": 0})) == ["<+2 more>"]
    assert Truncation.from_json(None) == Truncation()
    for bad in ({"max_items": 2}, {"max_list_items": -1}, {"max_str_chars": 2.5}, {"max_str_chars": True}):
        with pytest.raises((ValueError, TypeError)):
            Truncation.from_json(bad)


def test_truncation_applies_before_both_sections():
    task = {"request": "Summarize.", "tools": [{"name": "list_mail", "params": [], "description": "List mail"}]}
    h = [{"tool": "list_mail", "args": {}, "obs": {"items": ["a", "b", "c"], "note": "abcdef"}}]
    user = render(
        task, h, {"list_mail": {"reads": ["mail"], "writes": []}}, truncation={"max_list_items": 1, "max_str_chars": 2}
    ).user
    assert '  list_mail() -> {"items":["a","<+2 more>"],"note":"ab<+4 chars>"}' in user
    assert '  list_mail().items[1] = "<+2 more>"' in user and '  list_mail().note = "ab<+4 chars>"' in user


def test_prompt_is_a_function_of_canonical_content():
    shuffled_args = {
        "tool": "get_flight_price",
        "args": {"to": "SG", "from": "NYC"},
        "obs": {"price": 1200.0, "request_id": "x"},
    }
    assert render(TASK, [A, B], CONTRACTS) == render(TASK, [B, shuffled_args], CONTRACTS)
    assert "request_id" not in render(TASK, [shuffled_args], CONTRACTS).user


def test_render_matches_the_oracle_on_random_histories():
    tools = [{"name": f"tool{t}", "params": ["x"], "description": f"Tool {t}"} for t in range(5)]
    for seed in range(200):
        rng = random.Random(seed)
        request, steps, raw = R.random_demo(rng, rng.randint(0, 7))
        rng.shuffle(steps)
        task = {"request": request, "tools": rng.sample(tools, len(tools))}
        trunc = rng.choice([None, {"max_list_items": 1}, {"max_str_chars": 2, "max_list_items": 0}])
        for variant in ("quotient", "chronological"):
            got = render(task, steps, raw, variant=variant, truncation=trunc)
            exp = R.render(task, steps, raw, variant, trunc)
            assert {"system": got.system, "user": got.user} == exp


# ----------------------------------------------------------------------------- invariance (SPEC §5.6 a)
def test_every_prefix_of_every_legal_order_renders_like_the_demo_order_history():
    tools = [{"name": f"tool{t}", "params": ["x"], "description": f"Tool {t}"} for t in range(5)]
    checked = 0
    for seed in range(300):
        rng = random.Random(10_000 + seed)
        request, steps, raw = R.random_demo(rng, rng.randint(1, 8))
        n = len(steps)
        contracts = Contracts.from_json(raw)
        task = {"request": request, "tools": tools}
        preds = predecessor_sets(n, compile_edges(request, steps, contracts))
        by_set = {}
        for _ in range(20):
            order = R.random_topological_order(preds, n, rng)
            for k in range(n + 1):
                done = frozenset(order[:k])
                if done not in by_set:
                    demo_history = [steps[i] for i in range(n) if i in done]
                    by_set[done] = render(task, demo_history, contracts)
                    assert {"system": by_set[done].system, "user": by_set[done].user} == R.render(
                        task, demo_history, raw
                    )
                assert render(task, [steps[i] for i in order[:k]], contracts) == by_set[done], (seed, order, k)
                checked += 1
    assert checked >= 300 * 20

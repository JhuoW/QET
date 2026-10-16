"""Tool contracts, resource patterns and the conflict function (SPEC §3)."""

import random

import pytest
import qet_ref as R

from qet.core.contracts import WILDCARD, Contract, Contracts, Effects, as_contracts, conflict, patterns_meet

EVERYTHING = Effects(frozenset({WILDCARD}), frozenset({WILDCARD}))
PURE = Effects(frozenset(), frozenset())


def test_glob_resources():
    assert patterns_meet("order:*", "order:7")
    assert patterns_meet("order:7", "order:*")
    assert patterns_meet("*", "anything at all")
    assert patterns_meet("order:*", "order:")
    assert patterns_meet("a*c*e", "abcde")
    assert not patterns_meet("order:7", "order:8")
    assert not patterns_meet("order:*", "orders")
    # two patterns meet only when one glob-matches the other (SPEC §3), not when they merely share a match
    assert not patterns_meet("*:7", "order:*")


def test_glob_star_matches_any_character_and_nothing_else_is_special():
    assert patterns_meet("note:*", "note:line one\nline two")
    assert not patterns_meet("a.c", "abc")
    assert patterns_meet("a+b(c)$", "a+b(c)$")


def test_missing_template_argument_makes_the_resource_a_wildcard():
    c = Contracts.from_json(
        {
            "update_order": {"reads": ["order:{id}"], "writes": ["order:{id}", "log"]},
            "get_flight_price": {"reads": ["flights"], "writes": []},
        }
    )
    e = c.effects("update_order", {"status": "paid"})
    assert e == Effects(frozenset({WILDCARD}), frozenset({WILDCARD, "log"}))
    assert conflict(e, c.effects("get_flight_price", {"from": "NYC"}))


def test_template_values_strings_verbatim_other_values_canonical():
    c = Contracts.from_json({"pay": {"reads": ["acct:{account}"], "writes": ["acct:{account}/{memo}"]}})
    e = c.effects("pay", {"account": 1.0, "memo": "x y"})
    assert e == Effects(frozenset({"acct:1"}), frozenset({"acct:1/x y"}))
    e = c.effects("pay", {"account": {"b": True, "a": [2.0]}, "memo": None})
    assert e.reads == frozenset({'acct:{"a":[2],"b":true}'}) and e.writes == frozenset({'acct:{"a":[2],"b":true}/null'})


def test_tool_without_contract_reads_and_writes_everything():
    c = Contracts.from_json({"read1": {"reads": ["r1"], "writes": []}})
    assert c.effects("mystery", {"x": 1}) == EVERYTHING
    assert conflict(c.effects("mystery", {}), c.effects("read1", {}))
    assert as_contracts(None).effects("anything", {}) == EVERYTHING
    assert Contracts.from_json({"t": None}).effects("t", {}) == EVERYTHING


def test_pure_tool_commutes_even_with_an_unknown_tool():
    c = Contracts.from_json({"calc": {"reads": [], "writes": []}})
    assert c.effects("calc", {"a": 4}) == PURE
    assert not conflict(c.effects("calc", {}), EVERYTHING)
    assert not conflict(EVERYTHING, c.effects("calc", {}))


def test_bernstein_condition():
    r = Effects(frozenset({"x"}), frozenset())
    w = Effects(frozenset({"x"}), frozenset({"x"}))
    blind_w = Effects(frozenset(), frozenset({"x"}))
    other = Effects(frozenset({"y"}), frozenset({"y"}))
    assert not conflict(r, r)
    assert conflict(w, r) and conflict(r, w)
    assert conflict(blind_w, blind_w)
    assert conflict(blind_w, r) and conflict(r, blind_w)
    assert not conflict(w, other) and not conflict(other, w)


def test_volatile_paths_are_deleted_from_the_canonical_observation():
    c = Contracts.from_json(
        {"f": {"reads": [], "writes": [], "volatile": ["request_id", "meta.ts", "missing.path", "rows.x", "n.deep"]}}
    )
    obs = {"request_id": "r-1", "meta": {"ts": 5, "keep": 1}, "price": 1200.0, "rows": [{"x": 1}], "n": 3}
    assert c.canon_obs("f", obs) == {"meta": {"keep": 1}, "price": 1200, "rows": [{"x": 1}], "n": 3}
    assert obs["request_id"] == "r-1" and obs["meta"]["ts"] == 5  # the input is not modified
    assert c.canon_obs("g", {"request_id": "r-1"}) == {"request_id": "r-1"}  # only the tool's own fields
    assert c.canon_obs("f", "error: timeout") == "error: timeout"
    assert c.canon_obs("f", None) is None


def test_contract_validation():
    with pytest.raises(ValueError):
        Contract.from_json({"reads": [], "writes": [], "volatle": ["ts"]})
    with pytest.raises(TypeError):
        Contract.from_json({"reads": "order:{id}", "writes": []})
    with pytest.raises(TypeError):
        Contract.from_json({"reads": [1], "writes": []})
    with pytest.raises(ValueError):
        Contract.from_json({"reads": ["order:?"], "writes": []})
    with pytest.raises(ValueError):
        Contract.from_json({"reads": [], "writes": ["order:[0-9]"]})
    with pytest.raises(ValueError):
        Contract.from_json({"reads": [], "writes": [], "volatile": [""]})
    with pytest.raises(TypeError):
        Contracts.from_json({"t": ["reads"]})


def test_omitted_fields_are_empty():
    assert Contract.from_json({}) == Contract()
    assert Contracts.from_json({"t": {"volatile": ["ts"]}}).effects("t", {}) == PURE


# ----------------------------------------------------------------------------- differential against the oracle
def random_contracts(rng: random.Random) -> dict:
    templates = ["r0", "r1", "acct:{account}", "order:{id}", "order:*", "*", "{account}:{id}", "log"]
    out = {}
    for t in range(5):
        if rng.random() < 0.15:
            continue  # no contract
        out[f"tool{t}"] = {
            "reads": rng.sample(templates, rng.randint(0, 2)),
            "writes": rng.sample(templates, rng.randint(0, 2)) if rng.random() < 0.5 else [],
        }
    return out


def random_args(rng: random.Random) -> dict:
    values = [1, 1.0, 2, "1", "7", True, None, "a b", [1, 2], {"k": 1}]
    return {k: rng.choice(values) for k in rng.sample(["account", "id", "x"], rng.randint(0, 3))}


def test_effects_and_conflict_match_the_oracle():
    rng = random.Random(0)
    for _ in range(300):
        raw = random_contracts(rng)
        c = Contracts.from_json(raw)
        calls = [(f"tool{rng.randrange(6)}", random_args(rng)) for _ in range(6)]
        eff = [c.effects(t, a) for t, a in calls]
        for (t, a), e in zip(calls, eff, strict=True):
            reads, writes = R.effects(t, a, raw)
            assert e == Effects(reads, writes)
        for i in range(len(calls)):
            for j in range(len(calls)):
                assert conflict(eff[i], eff[j]) == R.conflict(calls[i], calls[j], raw)


def test_volatile_deletion_matches_the_oracle_on_random_demos():
    for seed in range(100):
        rng = random.Random(seed)
        _, steps, raw = R.random_demo(rng, rng.randint(1, 7))
        c = Contracts.from_json(raw)
        for s in steps:
            assert c.canon_obs(s["tool"], s["obs"]) == R.canon_obs(s["tool"], s["obs"], raw)

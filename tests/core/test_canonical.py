"""Canonical form, action strings and call names (SPEC §2)."""

import math
import random

import pytest
import qet_ref as R

from qet.core.canonical import action_string, canon, canon_json, scalar_leaves, show_call, value_key


def test_integral_floats_become_ints():
    assert canon(1200.0) == 1200 and type(canon(1200.0)) is int
    assert canon_json(-0.0) == "0"
    assert canon_json(0.0) == "0"
    assert canon_json(-3.0) == "-3"
    assert canon_json(2.0**53 - 1) == str(2**53 - 1)


def test_large_and_fractional_floats_keep_their_shortest_repr():
    assert canon_json(2.0**53) == "9007199254740992.0"
    assert canon_json(1e16) == "1e+16"
    assert canon_json(2.5) == "2.5"
    assert canon_json(0.1) == "0.1"
    assert canon_json(-1e-7) == "-1e-07"


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, [1, {"x": math.nan}]])
def test_non_finite_numbers_are_errors(bad):
    with pytest.raises(ValueError):
        canon(bad)


def test_strings_and_keys_are_nfc_without_other_changes():
    assert canon("é") == "é"
    assert canon({"é": "é"}) == {"é": "é"}
    assert canon(" Mixed Case ") == " Mixed Case "
    assert canon_json("Zürich\n") == '"Zürich\\n"'


def test_object_keys_are_sorted_by_code_point():
    assert canon_json({"b": 1, "B": 2, "a": 3, "é": 4, "_": 5}) == '{"B":2,"_":5,"a":3,"b":1,"é":4}'
    assert list(canon({"b": 1, "a": {"d": 1, "c": 2}})) == ["a", "b"]


def test_nfc_key_collision_is_an_error():
    with pytest.raises(ValueError):
        canon({"é": 1, "é": 2})


def test_non_string_keys_are_errors():
    with pytest.raises(TypeError):
        canon({1: "x"})


def test_unsupported_types_are_errors():
    for bad in ({1, 2}, object(), b"bytes"):
        with pytest.raises(TypeError):
            canon(bad)


def test_bool_null_int_and_lists_are_unchanged():
    assert canon(True) is True and canon(None) is None and canon(10**30) == 10**30
    assert canon((3, 2.0, "1")) == [3, 2, "1"]
    assert canon_json([3, 2.0, "1"]) == '[3,2,"1"]'


def test_value_keys_are_typed():
    assert value_key(1.0) == value_key(1)
    assert value_key(True) != value_key(1)
    assert value_key(False) != value_key(0)
    assert value_key(None) != value_key(0)
    assert value_key("1") != value_key(1)
    assert value_key("é") == value_key("é")


def test_scalar_leaves_skip_empty_containers():
    assert list(scalar_leaves(canon({"b": [1, {"c": None}], "a": "x", "e": {}, "f": []}))) == ["x", 1, None]
    assert list(scalar_leaves(5)) == [5]


def test_action_string_is_tool_first_with_sorted_args():
    assert (
        action_string("get_flight_price", {"to": "SG", "from": "NYC"})
        == '{"tool":"get_flight_price","args":{"from":"NYC","to":"SG"}}'
    )
    assert action_string("ping", {}) == '{"tool":"ping","args":{}}'


def test_action_string_requires_a_string_tool_and_object_args():
    with pytest.raises(TypeError):
        action_string("f", [1])
    with pytest.raises(TypeError):
        action_string(3, {})


def test_show_call():
    assert show_call("ping", {}) == "ping()"
    assert show_call("get_balance", {"account": 1.0}) == "get_balance(account=1)"
    assert show_call("search", {"q": {"z": 1, "a": [2, 3.0]}, "limit": 10}) == 'search(limit=10, q={"a":[2,3],"z":1})'


# ----------------------------------------------------------------------------- differential against the oracle
_STRINGS = ["", "a", "NYC", "é", "é", "Zürich", "x y", 'q"uote', "back\\slash", "tab\tline\n", "日本"]
_FLOATS = [0.0, -0.0, 1.0, -2.0, 2.5, 1e16, 2.0**53, 2.0**53 - 1, 1e-7, 123.456, -0.5]


def random_json(rng: random.Random, depth: int = 0):
    kind = rng.randrange(8 if depth < 3 else 6)
    if kind == 0:
        return None
    if kind == 1:
        return rng.random() < 0.5
    if kind == 2:
        return rng.randint(-(10**6), 10**6)
    if kind == 3:
        return rng.choice(_FLOATS)
    if kind in (4, 5):
        return rng.choice(_STRINGS)
    if kind == 6:
        return [random_json(rng, depth + 1) for _ in range(rng.randrange(4))]
    keys = rng.sample(["a", "b", "B", "z", "é", "k k", "1", "_x"], rng.randrange(5))
    return {k: random_json(rng, depth + 1) for k in keys}


def test_matches_the_oracle_on_random_values():
    rng = random.Random(0)
    for _ in range(3000):
        x = random_json(rng)
        assert canon_json(x) == R.canon_json(x)
        assert canon(x) == R.canon_value(x)


def test_action_strings_and_call_names_match_the_oracle():
    rng = random.Random(1)
    for _ in range(1000):
        tool = rng.choice(["f", "get_flight_price", "tool_3"])
        args = {k: random_json(rng, 1) for k in rng.sample(["a", "b", "from", "to", "é"], rng.randrange(4))}
        assert action_string(tool, args) == R.action_string(tool, args)
        assert show_call(tool, args) == R.show_call(tool, args)

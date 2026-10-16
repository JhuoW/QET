"""Canonical JSON form, action strings and call names (SPEC §2).

Every string the model reads or emits is built from canonical values, so equal values always give equal bytes:
NFC strings, objects with keys sorted by code point, integral floats as ints, lists in their own order.
"""

from __future__ import annotations

import json
import math
import unicodedata
from collections.abc import Iterator, Mapping
from typing import Any

JSON = Any  # a JSON value: None, bool, int, float, str, list of JSON, or dict from str to JSON

_EXACT_INT_LIMIT = 2**53  # integral floats below this magnitude convert to int without loss


def canon(x: JSON) -> JSON:
    """The canonical form of a JSON value (SPEC §2).

    Raises ValueError for NaN, infinities and two keys that collide after NFC normalization, and TypeError for
    non-string keys and values that are not JSON.
    """
    if x is None or isinstance(x, bool):
        return x
    if isinstance(x, int):
        return int(x)
    if isinstance(x, float):
        if not math.isfinite(x):
            raise ValueError(f"non-finite number {x!r} has no canonical form")
        if x == 0:
            return 0  # 0.0 and -0.0
        if x.is_integer() and abs(x) < _EXACT_INT_LIMIT:
            return int(x)
        return float(x)
    if isinstance(x, str):
        return unicodedata.normalize("NFC", x)
    if isinstance(x, Mapping):
        for key in x:
            if not isinstance(key, str):
                raise TypeError(f"object keys must be strings, got {type(key).__name__} {key!r}")
        out: dict[str, JSON] = {}
        for key, value in sorted(((unicodedata.normalize("NFC", k), v) for k, v in x.items()), key=lambda kv: kv[0]):
            if key in out:
                raise ValueError(f"two object keys normalize to the same NFC string {key!r}")
            out[key] = canon(value)
        return out
    if isinstance(x, (list, tuple)):
        return [canon(v) for v in x]
    raise TypeError(f"{type(x).__name__} is not a JSON value")


def canon_json(x: JSON) -> str:
    """Compact JSON of the canonical form: sorted keys, no spaces, non-ASCII kept (SPEC §2)."""
    return json.dumps(canon(x), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def action_string(tool: str, args: Mapping[str, JSON]) -> str:
    """The target string of a call: tool first, then the canonical args (SPEC §2, D-001).

    Example: {"tool":"get_flight_price","args":{"from":"NYC","to":"SG"}}. The end-of-turn token that completes the
    target is appended at tokenization (D-002).
    """
    if not isinstance(tool, str):
        raise TypeError(f"tool must be a string, got {type(tool).__name__}")
    if not isinstance(args, Mapping):
        raise TypeError(f"args must be an object, got {type(args).__name__}")
    return '{"tool":' + json.dumps(canon(tool), ensure_ascii=False) + ',"args":' + canon_json(args) + "}"


def show_call(tool: str, args: Mapping[str, JSON]) -> str:
    """Display name of a call, `tool(k1=v1, k2=v2)` with sorted keys and canon_json values (SPEC §2).

    The `#2`, `#3`, ... suffixes of repeated identical calls are added by Reduce, which sees the whole history.
    """
    if not isinstance(args, Mapping):
        raise TypeError(f"args must be an object, got {type(args).__name__}")
    a = canon(args)
    return f"{tool}(" + ", ".join(f"{k}={canon_json(a[k])}" for k in sorted(a)) + ")"


def canon_call(tool: object, args: object) -> tuple[str, dict[str, JSON]]:
    """A call {tool, args} as (tool, canonical args); the tool must be a string and the args an object."""
    if not isinstance(tool, str):
        raise TypeError(f"a call needs a string tool name, got {tool!r}")
    if not isinstance(args, Mapping):
        raise TypeError(f"the args of a {tool!r} call must be an object, got {type(args).__name__}")
    return tool, canon(args)


def scalar_leaves(x: JSON) -> Iterator[JSON]:
    """The scalar leaves of a value: objects in key order, lists in order; empty containers yield nothing."""
    if isinstance(x, Mapping):
        for k in sorted(x):
            yield from scalar_leaves(x[k])
    elif isinstance(x, (list, tuple)):
        for v in x:
            yield from scalar_leaves(v)
    else:
        yield x


def value_key(v: JSON) -> tuple[str, JSON]:
    """Typed comparison key of a scalar after canonicalization: 1.0 equals 1, but true never equals 1 (SPEC §4)."""
    if isinstance(v, bool):
        return ("bool", v)
    if v is None:
        return ("null", None)
    if isinstance(v, (int, float)):
        return ("num", canon(v))
    if isinstance(v, str):
        return ("str", canon(v))
    raise TypeError(f"{type(v).__name__} is not a scalar JSON value")

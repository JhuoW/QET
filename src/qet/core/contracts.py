"""Tool contracts, resource patterns and the conflict function (SPEC §3).

A contract declares once per tool which resources a call reads and writes, and which observation fields are
volatile (deleted during canonicalization). `conflict` is the Bernstein condition on the effects of two calls.
Compile (§4) and Reduce (§5) both call it, so the orders Compile keeps are exactly the orders Reduce preserves.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from qet.core.canonical import JSON, canon, canon_json

WILDCARD = "*"

_PLACEHOLDER = re.compile(r"\{(\w+)\}")
_OTHER_GLOB_CHARS = re.compile(r"[?\[\]]")
_FIELDS = ("reads", "writes", "volatile")


@dataclass(frozen=True)
class Effects:
    """The resources one call reads and writes, as patterns in which `*` is a glob."""

    reads: frozenset[str]
    writes: frozenset[str]


EVERYTHING = Effects(frozenset({WILDCARD}), frozenset({WILDCARD}))  # a tool without a contract


@dataclass(frozen=True)
class Contract:
    """Resource templates (`order:{id}`) read and written by a tool, and its volatile observation paths."""

    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    volatile: tuple[str, ...] = ()

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Contract:
        """Parse `{"reads": [...], "writes": [...], "volatile": [...]}`; an omitted field is empty."""
        if not isinstance(data, Mapping):
            raise TypeError(f"a contract must be an object, got {type(data).__name__}")
        unknown = sorted(set(data) - set(_FIELDS))
        if unknown:
            raise ValueError(f"unknown contract fields {unknown}; expected {list(_FIELDS)}")
        fields = {}
        for name in _FIELDS:
            items = data.get(name, [])
            if not isinstance(items, (list, tuple)) or not all(isinstance(s, str) for s in items):
                raise TypeError(f"contract field {name!r} must be a list of strings, got {items!r}")
            fields[name] = tuple(items)
        for pattern in fields["reads"] + fields["writes"]:
            if _OTHER_GLOB_CHARS.search(_PLACEHOLDER.sub("", pattern)):
                raise ValueError(f"resource pattern {pattern!r} uses a glob character other than '*'")
        if any(not path for path in fields["volatile"]):
            raise ValueError("volatile paths must be non-empty dotted paths")
        return cls(**fields)

    def effects(self, args: Mapping[str, JSON]) -> Effects:
        a = canon(args)
        return Effects(frozenset(_resource(t, a) for t in self.reads), frozenset(_resource(t, a) for t in self.writes))


def _resource(template: str, args: Mapping[str, JSON]) -> str:
    """Fill `{arg}` with the argument's value (strings verbatim, other values as canon_json). A template that names
    a missing argument becomes the wildcard, so uncertainty adds conflicts instead of removing them."""
    try:
        return _PLACEHOLDER.sub(lambda m: _show_value(args[m.group(1)]), template)
    except KeyError:
        return WILDCARD


def _show_value(v: JSON) -> str:
    return v if isinstance(v, str) else canon_json(v)


class Contracts:
    """The contracts of a task's tools. A tool without a contract reads and writes everything."""

    def __init__(self, by_tool: Mapping[str, Contract | None] | None = None):
        self._by_tool: dict[str, Contract] = {t: c for t, c in (by_tool or {}).items() if c is not None}

    @classmethod
    def from_json(cls, data: Mapping[str, Any] | None) -> Contracts:
        """Parse `{tool: contract}`; a null contract means the tool has none."""
        if data is None:
            return cls()
        if not isinstance(data, Mapping):
            raise TypeError(f"contracts must be an object keyed by tool name, got {type(data).__name__}")
        return cls({tool: None if c is None else Contract.from_json(c) for tool, c in data.items()})

    def get(self, tool: str) -> Contract | None:
        return self._by_tool.get(tool)

    def effects(self, tool: str, args: Mapping[str, JSON]) -> Effects:
        c = self._by_tool.get(tool)
        return EVERYTHING if c is None else c.effects(args)

    def canon_obs(self, tool: str, obs: JSON) -> JSON:
        """Canonical observation of a call to `tool` with the tool's volatile fields deleted (SPEC §2-§3)."""
        o = canon(obs)
        c = self._by_tool.get(tool)
        if c is not None:
            for path in c.volatile:
                _delete_path(o, path.split("."))
        return o


def as_contracts(contracts: Contracts | Mapping[str, Any] | None) -> Contracts:
    """Accept parsed contracts or their JSON form."""
    return contracts if isinstance(contracts, Contracts) else Contracts.from_json(contracts)


def _delete_path(obj: JSON, parts: list[str]) -> None:
    """Delete the dotted path from nested objects; a path that is absent or crosses a non-object is ignored."""
    for part in parts[:-1]:
        if not isinstance(obj, dict):
            return
        obj = obj.get(part)
    if isinstance(obj, dict):
        obj.pop(parts[-1], None)


@lru_cache(maxsize=4096)
def _glob(pattern: str) -> re.Pattern[str]:
    return re.compile(".*".join(re.escape(part) for part in pattern.split(WILDCARD)), re.DOTALL)


def patterns_meet(p: str, q: str) -> bool:
    """Two resource patterns meet when either one glob-matches the other; `*` matches any string (SPEC §3)."""
    if p == q:
        return True
    return (WILDCARD in p and _glob(p).fullmatch(q) is not None) or (
        WILDCARD in q and _glob(q).fullmatch(p) is not None
    )


def _meet(a: frozenset[str], b: frozenset[str]) -> bool:
    return any(patterns_meet(p, q) for p in a for q in b)


def conflict(a: Effects, b: Effects) -> bool:
    """Bernstein condition: W(a) meets R(b) or W(b), or W(b) meets R(a) (SPEC §3). Symmetric."""
    return _meet(a.writes, b.reads | b.writes) or _meet(b.writes, a.reads)

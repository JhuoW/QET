"""Tool contracts, resource patterns and the conflict function (SPEC §3).

A contract declares once per tool which resources a call reads and writes, and which observation fields are
volatile (deleted during canonicalization). `conflict` is the Bernstein condition on the effects of two calls.
Compile (§4) and Reduce (§5) both call it, so the orders Compile keeps are exactly the orders Reduce preserves.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from qet.core.canonical import JSON, canon, canon_json

WILDCARD = "*"

_PLACEHOLDER = re.compile(r"\{(\w+)\}")
_FORBIDDEN_OUTSIDE_PLACEHOLDERS = re.compile(r"[?\[\]{}]")
_FIELDS = ("reads", "writes", "volatile")
_REQUIRED = ("reads", "writes")


@dataclass(frozen=True)
class Effects:
    """The resources one call reads and writes, as patterns in which `*` is a glob."""

    reads: frozenset[str]
    writes: frozenset[str]


EVERYTHING = Effects(frozenset({WILDCARD}), frozenset({WILDCARD}))  # a tool without a contract


@dataclass(frozen=True)
class Contract:
    """Resource templates (`order:{id}`) read and written by a tool, and its volatile observation paths."""

    reads: tuple[str, ...]
    writes: tuple[str, ...]
    volatile: tuple[str, ...] = ()

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Contract:
        """Parse `{"reads": [...], "writes": [...], "volatile": [...]}` (D-015).

        `reads` and `writes` are required (`[]` declares that the tool reads or writes nothing, so a typo cannot
        silently declare a pure tool); `volatile` may be omitted. Patterns may use `{arg}` placeholders whose
        names are word characters and `*` as a glob; any other `{`, `}`, `?`, `[`, `]` is an error.
        """
        if not isinstance(data, Mapping):
            raise TypeError(f"a contract must be an object, got {type(data).__name__}")
        unknown = sorted(set(data) - set(_FIELDS))
        if unknown:
            raise ValueError(f"unknown contract fields {unknown}; expected {list(_FIELDS)}")
        missing = [name for name in _REQUIRED if name not in data]
        if missing:
            raise ValueError(f"a contract must declare {missing} (use [] for none)")
        fields = {}
        for name in _FIELDS:
            items = data.get(name, [])
            if not isinstance(items, (list, tuple)) or not all(isinstance(s, str) for s in items):
                raise TypeError(f"contract field {name!r} must be a list of strings, got {items!r}")
            fields[name] = tuple(items)
        for pattern in fields["reads"] + fields["writes"]:
            if _FORBIDDEN_OUTSIDE_PLACEHOLDERS.search(_PLACEHOLDER.sub("", pattern)):
                raise ValueError(
                    f"resource pattern {pattern!r} has a character that is not allowed outside an {{arg}} "
                    "placeholder ('?', '[', ']', '{', '}'); placeholder names are word characters"
                )
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

    def volatile(self, tool: str) -> tuple[str, ...]:
        """The volatile observation paths of `tool` (none without a contract)."""
        c = self._by_tool.get(tool)
        return c.volatile if c is not None else ()

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


def glob_match(pattern: str, s: str) -> bool:
    """Whether `pattern` matches all of `s`, where `*` matches any string (empty and newlines included) and every
    other character matches itself. The text before the first `*` must be a prefix and the text after the last `*`
    a suffix; the segments in between are found left to right with str.find (leftmost placement is optimal), so
    nothing backtracks and model-generated arguments full of `*` cannot blow up."""
    if WILDCARD not in pattern:
        return pattern == s
    first, *middle, last = pattern.split(WILDCARD)
    if len(first) + len(last) > len(s) or not s.startswith(first) or not s.endswith(last):
        return False
    pos, end = len(first), len(s) - len(last)
    for segment in middle:
        if segment:
            found = s.find(segment, pos, end)
            if found < 0:
                return False
            pos = found + len(segment)
    return True


def patterns_meet(p: str, q: str) -> bool:
    """Two resource patterns meet when either one glob-matches the other (SPEC §3)."""
    return p == q or (WILDCARD in p and glob_match(p, q)) or (WILDCARD in q and glob_match(q, p))


def _meet(a: frozenset[str], b: frozenset[str]) -> bool:
    return any(patterns_meet(p, q) for p in a for q in b)


def conflict(a: Effects, b: Effects) -> bool:
    """Bernstein condition: W(a) meets R(b) or W(b), or W(b) meets R(a) (SPEC §3). Symmetric."""
    return _meet(a.writes, b.reads | b.writes) or _meet(b.writes, a.reads)

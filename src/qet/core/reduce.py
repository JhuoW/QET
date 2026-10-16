"""Reduce: raw history -> ordered entry list H (SPEC §5.1).

Reduce never reads G. It is defined for any history, including calls never seen in training, tools without a
contract and error observations, and for every legal history of the same completed set it returns the same H.
"""

from __future__ import annotations

import heapq
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from qet.core.canonical import JSON, canon_call, canon_json, show_call
from qet.core.contracts import Contracts, as_contracts, conflict

EntryKey = tuple[str, str, int]  # (tool, canon_json(args), k): the Reduce ordering key


@dataclass(frozen=True)
class Entry:
    """One completed call of H."""

    tool: str
    args: dict[str, JSON]  # canonical
    obs: JSON  # canonical, volatile fields removed
    name: str  # show_call(tool, args), plus "#k" for the k-th identical call
    key: EntryKey


def name_calls(calls: Iterable[tuple[str, Mapping[str, JSON]]]) -> list[tuple[str, EntryKey]]:
    """Reduce step 2: each call's entry name and ordering key, in execution order.

    The name is show_call(tool, args), with `#k` appended to the k-th call of the same name (k >= 2); the key is
    (tool, canon_json(args), k). Names never mention step numbers.
    """
    seen: Counter[str] = Counter()
    out = []
    for tool, args in calls:
        base = show_call(tool, args)
        seen[base] += 1
        k = seen[base]
        out.append((base if k == 1 else f"{base}#{k}", (tool, canon_json(args), k)))
    return out


def canonical_entries(
    history: Sequence[Mapping[str, Any]], contracts: Contracts | Mapping[str, Any] | None
) -> list[Entry]:
    """Reduce steps 1-2: canonicalize every call and observation (volatile fields deleted) and name the entries.
    Execution order is kept (the chronological renderer uses this list as it is)."""
    c = as_contracts(contracts)
    calls = [canon_call(h.get("tool"), h.get("args")) for h in history]
    return [
        Entry(tool, args, c.canon_obs(tool, h.get("obs")), name, key)
        for (tool, args), h, (name, key) in zip(calls, history, name_calls(calls), strict=True)
    ]


def smallest_key_order(preds: Sequence[Iterable[int]], keys: Sequence[EntryKey]) -> list[int]:
    """The topological order that repeatedly places, among the unplaced items whose predecessors are all placed,
    the one with the smallest key. Raises ValueError on a cycle."""
    n = len(keys)
    if len(preds) != n:
        raise ValueError(f"expected predecessor lists for {n} items, got {len(preds)}")
    pending = [set(p) for p in preds]
    succ: list[list[int]] = [[] for _ in range(n)]
    for v, p in enumerate(pending):
        for u in p:
            succ[u].append(v)
    heap = [(keys[v], v) for v in range(n) if not pending[v]]
    heapq.heapify(heap)
    order = []
    while heap:
        _, u = heapq.heappop(heap)
        order.append(u)
        for v in succ[u]:
            pending[v].discard(u)
            if not pending[v]:
                heapq.heappush(heap, (keys[v], v))
    if len(order) != n:
        raise ValueError("the order constraints have a cycle")
    return order


def reduce(history: Sequence[Mapping[str, Any]], contracts: Contracts | Mapping[str, Any] | None) -> list[Entry]:
    """Reduce (SPEC §5.1): the smallest-key topological order of the entries under the conflict constraints only.

    Entries that conflict (contracts.conflict, the function Compile uses) keep their real order; everything else
    is sorted by key.
    """
    c = as_contracts(contracts)
    entries = canonical_entries(history, c)
    effects = [c.effects(e.tool, e.args) for e in entries]
    preds = [[i for i in range(j) if conflict(effects[i], effects[j])] for j in range(len(entries))]
    return [entries[i] for i in smallest_key_order(preds, [e.key for e in entries])]

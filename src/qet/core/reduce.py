"""Reduce: raw history -> ordered entry list H (SPEC §5.1)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping

from qet.core.canonical import JSON, canon_json, show_call

EntryKey = tuple[str, str, int]  # (tool, canon_json(args), k): the Reduce ordering key


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

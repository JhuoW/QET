"""Progress-state lattice, start distribution and completion probabilities (SPEC §6.1-§6.3, §6.6).

A state is a set of completed nodes containing all predecessors of its members (a downset of G), stored as an
int bitmask over node ids. Predecessors are bitmasks too.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence

S_MAX = 256  # default state cap before switching to the sub-lattice (SPEC §6.1)


def node_masks(n: int, preds: Sequence[Iterable[int]]) -> tuple[int, ...]:
    """Predecessor bitmasks of nodes 0..n-1. Raises ValueError for unknown ids, self-loops and cycles."""
    if len(preds) != n:
        raise ValueError(f"expected predecessor lists for {n} nodes, got {len(preds)}")
    masks = []
    for v, p in enumerate(preds):
        m = 0
        for u in p:
            if isinstance(u, bool) or not isinstance(u, int) or not 0 <= u < n or u == v:
                raise ValueError(f"node {v} has an invalid predecessor {u!r}")
            m |= 1 << u
        masks.append(m)
    done, progress = 0, True
    while progress:
        progress = False
        for v, m in enumerate(masks):
            if not done >> v & 1 and m & ~done == 0:
                done |= 1 << v
                progress = True
    if done != (1 << n) - 1:
        raise ValueError("the predecessor relation has a cycle")
    return tuple(masks)


def members(state: int) -> list[int]:
    """Node ids of a state, ascending."""
    return [v for v in range(state.bit_length()) if state >> v & 1]


def state_sort_key(state: int) -> tuple[int, list[int]]:
    """States sort by size, then by their sorted members."""
    return state.bit_count(), members(state)


def ready(state: int, preds: Sequence[int]) -> list[int]:
    """Nodes outside `state` whose predecessors are all inside it, ascending (SPEC §6.1)."""
    return [v for v, p in enumerate(preds) if not state >> v & 1 and p & ~state == 0]


def enumerate_states(preds: Sequence[int], cap: int | None = S_MAX) -> list[int] | None:
    """All states, breadth-first from the empty state and merging repeats, sorted by `state_sort_key`.

    Returns None as soon as the count exceeds `cap` (SPEC §6.1: switch to the sub-lattice).
    """
    seen, queue = {0}, deque([0])
    while queue:
        state = queue.popleft()
        for v in ready(state, preds):
            nxt = state | 1 << v
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
                if cap is not None and len(seen) > cap:
                    return None
    return sorted(seen, key=state_sort_key)

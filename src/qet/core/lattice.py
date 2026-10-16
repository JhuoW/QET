"""Progress-state lattice, start distribution and completion probabilities (SPEC §6.1-§6.3, §6.6).

A state is a set of completed nodes that contains every predecessor of its members (a downset of G), stored as an
int bitmask over node ids; predecessors are bitmasks too. An edge is (state, node) for a tool call and (V, END) for
the finish action. All quantities are float64; completion probabilities are computed in log space.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections import Counter, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

S_MAX = 256  # default state cap before switching to the sub-lattice (SPEC §6.1)
END = -1  # node id of the finish action in edge keys

EdgeKey = tuple[int, int]  # (state, node) or (V, END)
LogP = Mapping[EdgeKey, float]  # log pi(a_v | H_I): token log-prob sum of the target, end-of-turn token included


# ----------------------------------------------------------------------------- states
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


# ----------------------------------------------------------------------------- the lattice in use
@dataclass(frozen=True)
class Lattice:
    """The states S in use for a DAG with n nodes: the full lattice or a sub-lattice (SPEC §6.6).

    S must contain the empty state and V, consist of downsets, and put every state on a chain from the empty
    state to V inside S (true of the full lattice and of the prefix sets of complete orders).
    """

    n: int
    preds: tuple[int, ...]
    states: tuple[int, ...]  # sorted by state_sort_key
    full: int = field(init=False)
    exact: bool = field(init=False)  # S is the full lattice
    ready: dict[int, tuple[int, ...]] = field(init=False, repr=False, compare=False)
    succ: dict[int, tuple[int, ...]] = field(init=False, repr=False, compare=False)  # ready v with I + v in S
    outside: dict[int, tuple[int, ...]] = field(init=False, repr=False, compare=False)  # ready v with I + v not in S
    edges: tuple[EdgeKey, ...] = field(init=False, repr=False, compare=False)  # inside S, finish edge last

    def __post_init__(self) -> None:
        n, preds, states = self.n, self.preds, self.states
        if len(preds) != n:
            raise ValueError(f"expected {n} predecessor masks, got {len(preds)}")
        full = (1 << n) - 1
        in_s = set(states)
        if len(in_s) != len(states) or list(states) != sorted(states, key=state_sort_key):
            raise ValueError("states must be distinct and sorted by state_sort_key")
        if 0 not in in_s or full not in in_s:
            raise ValueError("the states must include the empty state and the full state V")
        for s in states:
            if s & ~full or any(preds[v] & ~s for v in members(s)):
                raise ValueError(f"state {members(s)} is not a downset of the graph")
        rdy = {s: tuple(ready(s, preds)) for s in states}
        succ = {s: tuple(v for v in rdy[s] if s | 1 << v in in_s) for s in states}
        out = {s: tuple(v for v in rdy[s] if s | 1 << v not in in_s) for s in states}
        for s in states:
            if s != full and not succ[s]:
                raise ValueError(f"state {members(s)} has no successor inside the states")
            if s != 0 and not any(s & ~(1 << v) in in_s for v in members(s)):
                raise ValueError(f"state {members(s)} has no predecessor inside the states")
        edges = tuple((s, v) for s in states if s != full for v in succ[s]) + ((full, END),)
        derived = {"full": full, "exact": not any(out.values()), "ready": rdy, "succ": succ, "outside": out}
        for name, value in {**derived, "edges": edges}.items():
            object.__setattr__(self, name, value)


def _lattice(n: int, masks: tuple[int, ...], states: Iterable[int]) -> Lattice:
    return Lattice(n, masks, tuple(sorted(set(states), key=state_sort_key)))


def full_lattice(n: int, preds: Sequence[Iterable[int]], cap: int | None = S_MAX) -> Lattice | None:
    """The full lattice, or None when it has more than `cap` states (default S_MAX; None = no cap)."""
    masks = node_masks(n, preds)
    states = enumerate_states(masks, cap)
    return None if states is None else Lattice(n, masks, tuple(states))


def lattice_from_states(n: int, preds: Sequence[Iterable[int]], states: Iterable[int]) -> Lattice:
    """A lattice on explicit states (bitmasks), validated by `Lattice`."""
    return _lattice(n, node_masks(n, preds), states)


def is_topological_order(order: Sequence[int], preds: Sequence[int]) -> bool:
    done = 0
    for v in order:
        if isinstance(v, bool) or not isinstance(v, int) or not 0 <= v < len(preds) or done >> v & 1:
            return False
        if preds[v] & ~done:
            return False
        done |= 1 << v
    return done == (1 << len(preds)) - 1


def prefix_states(order: Sequence[int]) -> list[int]:
    """The states 0, {o1}, {o1, o2}, ... visited by a complete order."""
    out, s = [0], 0
    for v in order:
        s |= 1 << v
        out.append(s)
    return out


def random_topological_order(preds: Sequence[int], rng: random.Random) -> list[int]:
    """Kahn's algorithm with a uniformly random choice among the ready nodes (SPEC §6.6)."""
    state, order = 0, []
    for _ in range(len(preds)):
        v = rng.choice(ready(state, preds))
        order.append(v)
        state |= 1 << v
    return order


def sublattice_seed(graph_id: str, epoch: int, run_seed: int) -> int:
    """Seed of the sub-lattice sample of one graph in one epoch (SPEC §6.6). It does not depend on the
    configuration, so configurations 3-6 and a4 share S (D-013), and it is stable across processes."""
    digest = hashlib.sha256(f"qet-sublattice\x1f{graph_id}\x1f{epoch}\x1f{run_seed}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def sample_orders(
    n: int, preds: Sequence[Iterable[int]], k: int, seed: int, demo_order: Sequence[int] | None = None
) -> list[list[int]]:
    """The demo order (node ids are demo positions, so by default 0..n-1) plus k-1 random topological orders."""
    if k < 1:
        raise ValueError(f"the sub-lattice needs k >= 1 orders, got {k}")
    masks = node_masks(n, preds)
    demo = list(range(n)) if demo_order is None else list(demo_order)
    if not is_topological_order(demo, masks):
        raise ValueError(f"the demo order {demo} is not a topological order of the graph")
    rng = random.Random(seed)
    return [demo] + [random_topological_order(masks, rng) for _ in range(k - 1)]


def sublattice(n: int, preds: Sequence[Iterable[int]], orders: Iterable[Sequence[int]]) -> Lattice:
    """S = all prefix sets of the given complete orders (SPEC §6.6)."""
    masks = node_masks(n, preds)
    states: set[int] = set()
    for o in orders:
        if not is_topological_order(o, masks):
            raise ValueError(f"{list(o)} is not a topological order of the graph")
        states.update(prefix_states(o))
    return _lattice(n, masks, states)


def build_lattice(
    n: int,
    preds: Sequence[Iterable[int]],
    *,
    k: int,
    seed: int,
    s_max: int = S_MAX,
    demo_order: Sequence[int] | None = None,
) -> Lattice:
    """The exact lattice when it has at most s_max states, otherwise the sub-lattice of k sampled orders."""
    lat = full_lattice(n, preds, cap=s_max)
    if lat is not None:
        return lat
    return sublattice(n, preds, sample_orders(n, preds, k, seed, demo_order))


# ----------------------------------------------------------------------------- start distribution (SPEC §6.3)
def start_weights(lat: Lattice, lam: float) -> dict[int, float]:
    """mu_G over the states in use: mu(empty) = 1 - lam, mu(I) = lam / (n - 1) / N_|I| for 1 <= |I| <= n - 1,
    mu(V) = 0, with N_k counted over the states in use; mu(empty) = 1 when n <= 1."""
    if not 0.0 <= lam < 1.0:
        raise ValueError(f"lambda must lie in [0, 1), got {lam!r}")
    mu = dict.fromkeys(lat.states, 0.0)
    n = lat.n
    if n <= 1:
        mu[0] = 1.0
        return mu
    layers = Counter(s.bit_count() for s in lat.states)
    for s in lat.states:
        k = s.bit_count()
        if k == 0:
            mu[s] = 1.0 - lam
        elif k < n:
            mu[s] = lam / (n - 1) / layers[k]
    return mu


# ----------------------------------------------------------------------------- completion probability (eq. 1)
def logsumexp(xs: Iterable[float]) -> float:
    xs = [x for x in xs if x != -math.inf]
    if not xs:
        return -math.inf
    m = max(xs)
    return m + math.log(math.fsum(math.exp(x - m) for x in xs))


def edge_logp(logp: LogP, edge: EdgeKey) -> float:
    try:
        x = float(logp[edge])
    except KeyError:
        state, v = edge
        raise KeyError(f"no log-probability for edge {edge_key(state, v)!r}") from None
    if math.isnan(x):
        raise ValueError(f"log-probability of edge {edge_key(*edge)!r} is NaN")
    return x


def log_completion(lat: Lattice, logp: LogP, *, plus: bool = False) -> dict[int, float]:
    """log B_S(I) for every state, from the largest down: B(V) = pi(end | H_V), B(I) = sum over ready v of
    pi(a_v | H_I) * B(I + v). Successors outside S count 0 (B_S <= B); with plus=True they count 1 (B+ >= B, the
    diagnostic of SPEC §6.6). On the full lattice both are B."""
    log_b: dict[int, float] = {}
    for s in reversed(lat.states):
        if s == lat.full:
            log_b[s] = edge_logp(logp, (s, END))
            continue
        terms = []
        for v in lat.ready[s]:
            t = s | 1 << v
            if t in log_b:
                terms.append(edge_logp(logp, (s, v)) + log_b[t])
            elif plus:
                terms.append(edge_logp(logp, (s, v)))
        log_b[s] = logsumexp(terms)
    return {s: log_b[s] for s in lat.states}


# ----------------------------------------------------------------------------- facade keys (SPEC §10)
def state_key(state: int) -> str:
    """Sorted node ids joined by commas; "" is the empty state."""
    return ",".join(str(v) for v in members(state))


def edge_key(state: int, v: int) -> str:
    return f"{state_key(state)}|{'end' if v == END else v}"


def state_of(nodes: Iterable[int], n: int) -> int:
    """The bitmask of a list of node ids."""
    state = 0
    for v in nodes:
        if isinstance(v, bool) or not isinstance(v, int) or not 0 <= v < n or state >> v & 1:
            raise ValueError(f"invalid node id {v!r} in a state of a graph with {n} nodes")
        state |= 1 << v
    return state


def parse_state_key(key: str) -> int:
    state = 0
    for part in key.split(",") if key else ():
        v = int(part)
        if v < 0 or state >> v & 1:
            raise ValueError(f"invalid state key {key!r}")
        state |= 1 << v
    return state


def parse_edge_key(key: str) -> EdgeKey:
    state, sep, node = key.partition("|")
    if not sep:
        raise ValueError(f"invalid edge key {key!r}")
    return parse_state_key(state), (END if node == "end" else int(node))

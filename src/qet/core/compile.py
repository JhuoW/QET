"""Compile: a demonstration -> the DAG of the orders its task requires, graph records, compile report (SPEC §4).

Compile is conservative. Five rules add u -> v (u earlier in the demo) whenever the order might matter, and
uncertain cases (no contract, a value nobody produced) add edges. Edges point forward in demo order, so the graph
is acyclic and the demo order is one of its topological orders. No model calls, no environment calls, and never a
benchmark's ground-truth DAG.
"""

from __future__ import annotations

import heapq
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from qet.core.canonical import JSON, action_string, canon, scalar_leaves, value_key
from qet.core.contracts import Contracts, as_contracts, conflict
from qet.core.lattice import S_MAX, enumerate_states, node_masks
from qet.core.reduce import EntryKey

LABELS = ("data", "effect", "decision", "dup", "task")
AUDIT = "audit"  # label of the chain fallback after a failed replay audit
VERSION = "qet-compile-1"

_REQUEST_NUMBER = re.compile(r"(?<![0-9A-Za-z_.])-?\d+(?:\.\d+)?(?![0-9A-Za-z_])")


# ----------------------------------------------------------------------------- values available from the start
class StartValues:
    """Values available from the start (SPEC §4, D-005), compared by typed key after canonicalization.

    Numbers that appear in the request as standalone numeric tokens; strings that appear in it as a whole word
    sequence (case-sensitive); every scalar leaf of the initial observation; adapter-declared start values.
    """

    def __init__(self, request: str, initial_observation: JSON = None, declared: Iterable[JSON] = ()):
        if not isinstance(request, str):
            raise TypeError(f"request must be a string, got {type(request).__name__}")
        self.request = request
        keys = {value_key(canon(float(t) if "." in t else int(t))) for t in _REQUEST_NUMBER.findall(request)}
        keys.update(value_key(canon(v)) for v in declared)
        if initial_observation is not None:
            keys.update(value_key(v) for v in scalar_leaves(canon(initial_observation)))
        self._keys = frozenset(keys)

    def __contains__(self, value: JSON) -> bool:
        key = value_key(value)
        return key in self._keys or (key[0] == "str" and _whole_word_in(key[1], self.request))


def _whole_word_in(s: str, text: str) -> bool:
    return s == "" or re.search(r"(?<!\w)" + re.escape(s) + r"(?!\w)", text) is not None


# ----------------------------------------------------------------------------- edges
@dataclass(frozen=True, order=True)
class Edge:
    """u -> v with u < v, labelled by every rule that produced it (sorted)."""

    u: int
    v: int
    labels: tuple[str, ...]


def _call(step: Mapping[str, Any]) -> tuple[str, dict[str, JSON]]:
    tool, args = step.get("tool"), step.get("args")
    if not isinstance(tool, str):
        raise TypeError(f"a step needs a string 'tool', got {tool!r}")
    if not isinstance(args, Mapping):
        raise TypeError(f"the args of a {tool!r} step must be an object, got {type(args).__name__}")
    return tool, canon(args)


def _pair(pair: Sequence[int], n: int) -> tuple[int, int]:
    u, v = pair
    if isinstance(u, bool) or isinstance(v, bool) or not isinstance(u, int) or not isinstance(v, int):
        raise ValueError(f"a task constraint must be a pair of step indices, got {pair!r}")
    if not 0 <= u < v < n:
        raise ValueError(f"a task constraint must point from an earlier to a later step, got {pair!r}")
    return u, v


def compile_edges(
    request: str,
    steps: Sequence[Mapping[str, Any]],
    contracts: Contracts | Mapping[str, Any] | None,
    *,
    initial_observation: JSON = None,
    task_order: Iterable[Sequence[int]] = (),
    start_values: Iterable[JSON] = (),
) -> list[Edge]:
    """The five ordering rules of SPEC §4 over the demo steps ({tool, args, obs}, in time order), sorted by (u, v)."""
    c = as_contracts(contracts)
    calls = [_call(s) for s in steps]
    n = len(calls)
    obs = [c.canon_obs(tool, s.get("obs")) for (tool, _), s in zip(calls, steps, strict=True)]
    available = StartValues(request, initial_observation, start_values)
    produced = [frozenset(value_key(x) for x in scalar_leaves(o)) for o in obs]
    actions = [action_string(tool, args) for tool, args in calls]
    effects = [c.effects(tool, args) for tool, args in calls]
    labels: defaultdict[tuple[int, int], set[str]] = defaultdict(set)
    for v, (_, args) in enumerate(calls):
        for value in scalar_leaves(args):  # rule 1, data: from every producer, or from every earlier call
            if value in available:
                continue
            key = value_key(value)
            for u in [u for u in range(v) if key in produced[u]] or range(v):
                labels[(u, v)].add("data")
        writes = bool(effects[v].writes)
        for u in range(v):
            if conflict(effects[u], effects[v]):  # rule 2, effect
                labels[(u, v)].add("effect")
            if writes:  # rule 3, decision: a write never moves ahead of anything
                labels[(u, v)].add("decision")
            if actions[u] == actions[v]:  # rule 4, dup
                labels[(u, v)].add("dup")
    for pair in task_order:  # rule 5, task
        labels[_pair(pair, n)].add("task")
    return [Edge(u, v, tuple(sorted(ls))) for (u, v), ls in sorted(labels.items())]


def chain_edges(n: int) -> list[Edge]:
    """The chain fallback 0 -> 1 -> ... -> n-1 after a failed replay audit (SPEC §4)."""
    return [Edge(i, i + 1, (AUDIT,)) for i in range(n - 1)]


def predecessor_sets(n: int, edges: Iterable[Edge]) -> list[frozenset[int]]:
    preds: list[set[int]] = [set() for _ in range(n)]
    for e in edges:
        preds[e.v].add(e.u)
    return [frozenset(p) for p in preds]


def canonical_order(preds: Sequence[Iterable[int]], keys: Sequence[EntryKey]) -> list[int]:
    """Topological order that always takes the ready node with the smallest Reduce key (SPEC §8, ablation a2)."""
    n = len(keys)
    if len(preds) != n:
        raise ValueError(f"expected predecessor lists for {n} nodes, got {len(preds)}")
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
        raise ValueError("the graph has a cycle")
    return order


# ----------------------------------------------------------------------------- graph shape
def ancestor_masks(n: int, preds: Sequence[Iterable[int]]) -> list[int]:
    """Bitmask of all (transitive) predecessors of each node."""
    masks = node_masks(n, preds)
    anc = [0] * n
    done = 0
    while done != (1 << n) - 1:
        for v, m in enumerate(masks):
            if not done >> v & 1 and m & ~done == 0:
                a = m
                for u in range(n):
                    if m >> u & 1:
                        a |= anc[u]
                anc[v] = a
                done |= 1 << v
    return anc


def width(n: int, preds: Sequence[Iterable[int]]) -> int:
    """Size of the largest antichain: n minus a maximum matching of the comparability graph (Dilworth)."""
    anc = ancestor_masks(n, preds)
    below = [[v for v in range(n) if anc[v] >> u & 1] for u in range(n)]
    match = [-1] * n

    def augment(u: int, seen: list[bool]) -> bool:
        for v in below[u]:
            if not seen[v]:
                seen[v] = True
                if match[v] < 0 or augment(match[v], seen):
                    match[v] = u
                    return True
        return False

    return n - sum(augment(u, [False] * n) for u in range(n))


def incomparable_pairs(n: int, preds: Sequence[Iterable[int]]) -> int:
    """Number of unordered node pairs that no path connects."""
    anc = ancestor_masks(n, preds)
    return sum(1 for b in range(n) for a in range(b) if not (anc[b] >> a & 1 or anc[a] >> b & 1))


# ----------------------------------------------------------------------------- graph records (SPEC §1)
@dataclass(frozen=True)
class Node:
    id: int
    tool: str
    args: dict[str, JSON]  # canonical
    obs: JSON  # canonical, volatile fields removed

    def to_json(self) -> dict[str, Any]:
        return {"id": self.id, "call": {"tool": self.tool, "args": self.args}, "obs": self.obs}


@dataclass(frozen=True)
class CompileInfo:
    version: str = VERSION
    fallback_to_chain: bool = False
    orders_replayed: int = 0
    mismatches: int = 0
    volatile_consumed: tuple[tuple[int, int, str], ...] = ()  # (u, v, path): v consumed a volatile field of u

    def to_json(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "fallback_to_chain": self.fallback_to_chain,
            "audit": {"orders_replayed": self.orders_replayed, "mismatches": self.mismatches},
            "volatile_consumed": [list(x) for x in self.volatile_consumed],
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> CompileInfo:
        audit = data.get("audit", {})
        return cls(
            version=data["version"],
            fallback_to_chain=bool(data["fallback_to_chain"]),
            orders_replayed=int(audit.get("orders_replayed", 0)),
            mismatches=int(audit.get("mismatches", 0)),
            volatile_consumed=tuple((int(u), int(v), str(p)) for u, v, p in data.get("volatile_consumed", [])),
        )


@dataclass(frozen=True)
class GraphRecord:
    """Compile output for one demonstration (SPEC §1). Node ids are demo positions, so edges point forward."""

    task_id: str
    request: str
    tools: list[dict[str, Any]]
    initial_observation: JSON
    snapshot: str | None
    nodes: tuple[Node, ...]
    edges: tuple[Edge, ...]
    final: dict[str, Any]  # {"tool": <finish tool>, "args": canonical args}
    compile: CompileInfo

    @property
    def n(self) -> int:
        return len(self.nodes)

    def preds(self) -> list[frozenset[int]]:
        return predecessor_sets(self.n, self.edges)

    def steps(self) -> list[dict[str, Any]]:
        """The demo steps {tool, args, obs} in demo order (canonical, volatile fields removed)."""
        return [{"tool": nd.tool, "args": nd.args, "obs": nd.obs} for nd in self.nodes]

    def to_json(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "request": self.request,
            "tools": self.tools,
            "initial_observation": self.initial_observation,
            "snapshot": self.snapshot,
            "nodes": [nd.to_json() for nd in self.nodes],
            "edges": [{"from": e.u, "to": e.v, "why": list(e.labels)} for e in self.edges],
            "final": self.final,
            "compile": self.compile.to_json(),
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> GraphRecord:
        nodes = tuple(Node(int(nd["id"]), nd["call"]["tool"], nd["call"]["args"], nd["obs"]) for nd in data["nodes"])
        if [nd.id for nd in nodes] != list(range(len(nodes))):
            raise ValueError("node ids must be the demo positions 0..n-1")
        edges = tuple(sorted(Edge(int(e["from"]), int(e["to"]), tuple(e["why"])) for e in data["edges"]))
        if any(not 0 <= e.u < e.v < len(nodes) for e in edges):
            raise ValueError("edges must point forward in demo order")
        return cls(
            task_id=data["task_id"],
            request=data["request"],
            tools=data["tools"],
            initial_observation=data.get("initial_observation"),
            snapshot=data.get("snapshot"),
            nodes=nodes,
            edges=edges,
            final=data["final"],
            compile=CompileInfo.from_json(data["compile"]),
        )


def _volatile_consumed(
    steps: Sequence[Mapping[str, Any]], calls: Sequence[tuple[str, dict]], c: Contracts, available: StartValues
) -> tuple[tuple[int, int, str], ...]:
    """(u, v, path) where a later call v uses, as an argument, a value found only in a volatile field of u."""
    found = []
    raw = [canon(s.get("obs")) for s in steps]
    for u, (tool, _) in enumerate(calls):
        contract = c.get(tool)
        for path in contract.volatile if contract else ():
            field: JSON = raw[u]
            for part in path.split("."):
                field = field.get(part) if isinstance(field, dict) else None
            keys = {value_key(x) for x in scalar_leaves(field)} if field is not None else set()
            for v in range(u + 1, len(calls)):
                if any(value_key(x) in keys and x not in available for x in scalar_leaves(calls[v][1])):
                    found.append((u, v, path))
    return tuple(found)


def compile_demo(
    task: Mapping[str, Any],
    demo: Sequence[Mapping[str, Any]],
    contracts: Contracts | Mapping[str, Any] | None,
    *,
    task_id: str,
    snapshot: str | None = None,
    task_order: Iterable[Sequence[int]] = (),
    start_values: Iterable[JSON] = (),
) -> GraphRecord:
    """Compile a successful demonstration (steps {tool, args, obs} in time order, then the finish call).

    `task` holds the public task x: request, tools, optional initial_observation and finish tool name (default
    "finish").
    """
    c = as_contracts(contracts)
    finish = task.get("finish", "finish")
    if not demo or demo[-1].get("tool") != finish:
        raise ValueError(f"a demonstration must end with a {finish!r} call")
    steps = demo[:-1]
    if any(s.get("tool") == finish for s in steps):
        raise ValueError(f"a demonstration has exactly one {finish!r} call, at the end")
    request, initial = task["request"], task.get("initial_observation")
    edges = compile_edges(
        request, steps, c, initial_observation=initial, task_order=task_order, start_values=start_values
    )
    calls = [_call(s) for s in steps]
    nodes = tuple(
        Node(i, tool, args, c.canon_obs(tool, s.get("obs")))
        for i, ((tool, args), s) in enumerate(zip(calls, steps, strict=True))
    )
    final_tool, final_args = _call(demo[-1])
    consumed = _volatile_consumed(steps, calls, c, StartValues(request, initial, start_values))
    return GraphRecord(
        task_id=task_id,
        request=request,
        tools=list(task.get("tools", [])),
        initial_observation=canon(initial),
        snapshot=snapshot,
        nodes=nodes,
        edges=tuple(edges),
        final={"tool": final_tool, "args": final_args},
        compile=CompileInfo(volatile_consumed=consumed),
    )


def fallback_to_chain(record: GraphRecord, *, orders_replayed: int, mismatches: int) -> GraphRecord:
    """The record after a failed replay audit: its edges become the chain, labelled `audit` (SPEC §4)."""
    info = replace(record.compile, fallback_to_chain=True, orders_replayed=orders_replayed, mismatches=mismatches)
    return replace(record, edges=tuple(chain_edges(record.n)), compile=info)


def write_records(path: str | Path, records: Iterable[GraphRecord]) -> None:
    """One graph record per line (JSONL), keys in SPEC §1 order."""
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec.to_json(), ensure_ascii=False, separators=(",", ":")) + "\n")


def read_records(path: str | Path) -> list[GraphRecord]:
    with open(path, encoding="utf-8") as f:
        return [GraphRecord.from_json(json.loads(line)) for line in f if line.strip()]


# ----------------------------------------------------------------------------- compile report (SPEC §4)
def compile_report(records: Sequence[GraphRecord], *, contract_source: str, s_max: int = S_MAX) -> dict[str, Any]:
    """Per-dataset compile statistics: fallback share, width distribution (largest antichain), share of
    incomparable pairs, states per graph (exact up to `s_max`), label counts, contract source, audit failures and
    volatile fields that later calls consumed."""
    widths: Counter[int] = Counter()
    states: Counter[int | str] = Counter()
    labels = dict.fromkeys(sorted((*LABELS, AUDIT)), 0)
    pairs = incomparable = 0
    consumed = []
    for rec in records:
        n, preds = rec.n, rec.preds()
        widths[width(n, preds)] += 1
        pairs += n * (n - 1) // 2
        incomparable += incomparable_pairs(n, preds)
        st = enumerate_states(node_masks(n, preds), cap=s_max)
        states[len(st) if st is not None else f">{s_max}"] += 1
        for e in rec.edges:
            for label in e.labels:
                labels[label] += 1
        consumed += [
            {"task_id": rec.task_id, "from": u, "to": v, "path": p} for u, v, p in rec.compile.volatile_consumed
        ]
    total = len(records)
    exact = sorted(k for k in states if isinstance(k, int))
    return {
        "tasks_compiled": total,
        "fallback_share": sum(r.compile.fallback_to_chain for r in records) / total if total else 0.0,
        "audit_failures": sum(1 for r in records if r.compile.mismatches > 0),
        "contract_source": contract_source,
        "label_counts": labels,
        "width": {str(k): widths[k] for k in sorted(widths)},
        "incomparable_pair_share": incomparable / pairs if pairs else 0.0,
        "states_per_graph": {
            **{str(k): states[k] for k in exact},
            **{k: v for k, v in states.items() if isinstance(k, str)},
        },
        "volatile_consumed": consumed,
    }

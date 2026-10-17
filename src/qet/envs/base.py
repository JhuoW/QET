"""What every environment adapter provides (EVAL_PLAN §3), and the audits built on it (SPEC §4).

An adapter gives the public task x (request, tools with the finish tool, initial observation), the tool contracts and
their source, the step budget, `step(tool, args) -> observation`, snapshot / restore, the native success check, and,
for audits only, the gold calls with the generator's true DAG. Compile never sees the true DAG (SPEC §4).
"""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

from qet.core.canonical import JSON, canon
from qet.core.compile import GraphRecord, ancestor_masks
from qet.core.contracts import as_contracts
from qet.core.lattice import node_masks, random_topological_order

FINISH_TOOL = {"name": "finish", "params": ["answer"], "description": "Submit the final answer"}


class Environment(Protocol):
    task_id: str
    contract_source: str

    def task(self) -> dict[str, Any]:
        """The public task x: request, tools (finish included), initial_observation, finish."""
        ...

    def contracts(self) -> dict[str, Any]: ...

    def step_budget(self) -> int:
        """Tool calls allowed before the episode ends (finish not counted)."""
        ...

    def step(self, tool: str, args: Mapping[str, JSON]) -> JSON: ...

    def snapshot(self) -> Any: ...

    def restore(self, snap: Any) -> None: ...

    def check(self, answer: JSON) -> bool:
        """The native success check, applied to the finish answer and the current state."""
        ...

    def final_state(self) -> JSON:
        """The state a replay audit compares (canonical JSON)."""
        ...


def run_calls(env: Environment, calls: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Execute calls in order and record the steps {tool, args, obs}."""
    return [{"tool": c["tool"], "args": dict(c["args"]), "obs": env.step(c["tool"], c["args"])} for c in calls]


def replay_audit(make_env: Callable[[], Environment], record: GraphRecord, *, k: int, seed: int) -> tuple[int, int]:
    """SPEC §4: from a fresh environment, execute k random topological orders of the record's graph and compare
    every node's canonical observation (volatile fields removed) and the final state with the demo's. Returns
    (orders replayed, mismatching orders)."""
    env = make_env()
    contracts = as_contracts(env.contracts())
    expected_obs = [nd.obs for nd in record.nodes]
    run_calls(env, [{"tool": nd.tool, "args": nd.args} for nd in record.nodes])
    expected_state = canon(env.final_state())
    rng = random.Random(seed)
    masks = node_masks(record.n, record.preds())
    mismatches = 0
    for _ in range(k):
        order = random_topological_order(masks, rng)
        env = make_env()
        got = {}
        for v in order:
            nd = record.nodes[v]
            got[v] = contracts.canon_obs(nd.tool, env.step(nd.tool, nd.args))
        if any(got[v] != expected_obs[v] for v in range(record.n)) or canon(env.final_state()) != expected_state:
            mismatches += 1
    return k, mismatches


def edge_audit(record: GraphRecord, true_preds: Sequence[Sequence[int]], *, orders: int = 200, seed: int = 0):
    """Compile against the generator's true DAG, both indexed by demo position (SPEC §4 audits): direct edge
    precision and recall, recall by reachability (a true order constraint implied by a path), and the share of
    sampled legal orders of the compiled graph that break a true constraint (the unsafe-order rate)."""
    n = record.n
    compiled = {(e.u, e.v) for e in record.edges}
    true = {(u, v) for v in range(n) for u in true_preds[v]}
    reach = ancestor_masks(n, record.preds())
    implied = {(u, v) for (u, v) in true if reach[v] >> u & 1}
    rng = random.Random(seed)
    masks = node_masks(n, record.preds())
    unsafe = 0
    for _ in range(orders if n else 0):
        pos = {v: i for i, v in enumerate(random_topological_order(masks, rng))}
        unsafe += any(pos[u] > pos[v] for u, v in true)
    return {
        "true_edges": len(true),
        "compiled_edges": len(compiled),
        "precision": len(compiled & true) / len(compiled) if compiled else 1.0,
        "recall": len(compiled & true) / len(true) if true else 1.0,
        "recall_by_reachability": len(implied) / len(true) if true else 1.0,
        "unsafe_order_rate": unsafe / orders if n else 0.0,
    }

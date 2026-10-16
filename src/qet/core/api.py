"""Thin JSON-in/JSON-out test facade used by reference/acceptance (SPEC §10).

Internals are free; these wrappers only convert between JSON and the core types.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from qet.core import canonical, compile, lattice, weights
from qet.core import render as rendering
from qet.core.canonical import JSON
from qet.core.reduce import name_calls
from qet.core.reduce import reduce as reduce_history


def canonical_json(value: JSON) -> str:
    return canonical.canon_json(value)


def action_string(tool: str, args: Mapping[str, JSON]) -> str:
    return canonical.action_string(tool, args)


def show_call(tool: str, args: Mapping[str, JSON]) -> str:
    return canonical.show_call(tool, args)


def _compile(case: Mapping[str, Any]) -> list[compile.Edge]:
    return compile.compile_edges(
        case["request"],
        case["steps"],
        case["contracts"],
        initial_observation=case.get("initial_observation"),
        task_order=case.get("task_order", ()),
        start_values=case.get("start_values", ()),
    )


def compile_edges(case: Mapping[str, Any]) -> list[list]:
    """`case`: request, steps, contracts, [initial_observation, task_order, start_values] -> [[u, v, labels]]."""
    return [[e.u, e.v, list(e.labels)] for e in _compile(case)]


def canonical_order(case: Mapping[str, Any]) -> list[int]:
    """The ablation-a2 order (SPEC §8) on the compiled graph of `case`."""
    steps = case["steps"]
    keys = [key for _, key in name_calls((s["tool"], s["args"]) for s in steps)]
    return compile.canonical_order(compile.predecessor_sets(len(steps), _compile(case)), keys)


def reduce(history: list[dict], contracts: Mapping[str, Any]) -> list[dict]:
    """Reduce (SPEC §5.1): [{"name": str, "obs": canonical obs}] in H order."""
    return [{"name": e.name, "obs": e.obs} for e in reduce_history(history, contracts)]


def render(
    task: Mapping[str, Any],
    history: list[dict],
    contracts: Mapping[str, Any],
    variant: str = "quotient",
    truncation: Mapping[str, Any] | None = None,
) -> dict:
    """{"system", "user"} for the quotient and chronological renderers; {"system", "messages"} (the chat turns after
    the system message) for native_multiturn."""
    if variant == "native_multiturn":
        msgs = rendering.render_messages(task, history, contracts, variant, truncation)
        return {"system": msgs[0]["content"], "messages": msgs[1:]}
    p = rendering.render(task, history, contracts, variant, truncation)
    return {"system": p.system, "user": p.user}


FULL_LATTICE_CAP = 1 << 16  # the facade enumerates the full lattice when no states are given; beyond this, pass states


def lattice_summary(
    n: int,
    preds: list[list[int]],
    logp: Mapping[str, float],
    method: str,
    lam: float,
    order: list[int] | None = None,
    paths: list[list[int]] | None = None,
    states: list[list[int]] | None = None,
    plus: bool = False,
) -> dict:
    """Lattice quantities of one configuration (SPEC §10). `logp` is keyed "<state>|<node>" / "<state>|end"; the
    lattice is the full one unless explicit `states` (node-id lists, a sub-lattice) are given. Returns logB and mu
    by state key, weights by edge key, qet_objective, surrogate, objective and logB_plus (when plus)."""
    if states is None:
        lat = lattice.full_lattice(n, preds, cap=FULL_LATTICE_CAP)
        if lat is None:
            raise ValueError(f"the lattice has more than {FULL_LATTICE_CAP} states; pass explicit states")
    else:
        lat = lattice.lattice_from_states(n, preds, [lattice.state_of(s, n) for s in states])
    table = {lattice.parse_edge_key(k): v for k, v in logp.items()}
    s = weights.summarize(method, lat, table, lam=lam, order=order, paths=paths, plus=plus)

    def by_state(d: dict[int, float]) -> dict[str, float]:
        return {lattice.state_key(st): x for st, x in d.items()}

    return {
        "logB": by_state(s.log_b),
        "mu": by_state(s.mu),
        "weights": {lattice.edge_key(*e): w for e, w in s.weights.items()},
        "qet_objective": s.qet_objective,
        "surrogate": s.surrogate,
        "objective": s.objective,
        "logB_plus": None if s.log_b_plus is None else by_state(s.log_b_plus),
    }

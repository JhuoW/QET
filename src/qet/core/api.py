"""Thin JSON-in/JSON-out test facade used by reference/acceptance (SPEC §10).

Internals are free; these wrappers only convert between JSON and the core types.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from qet.core import canonical, compile
from qet.core.canonical import JSON
from qet.core.reduce import name_calls


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

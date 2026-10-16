"""Thin JSON-in/JSON-out test facade used by reference/acceptance (SPEC §10).

Internals are free; these wrappers only convert between JSON and the core types.
"""

from __future__ import annotations

from collections.abc import Mapping

from qet.core import canonical
from qet.core.canonical import JSON


def canonical_json(value: JSON) -> str:
    return canonical.canon_json(value)


def action_string(tool: str, args: Mapping[str, JSON]) -> str:
    return canonical.action_string(tool, args)


def show_call(tool: str, args: Mapping[str, JSON]) -> str:
    return canonical.show_call(tool, args)

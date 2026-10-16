"""Render: (task, history) -> prompt, and the renderers of the baselines (SPEC §5.2-§5.7).

The quotient prompt is a function of (task, H) only, where H = Reduce(history): no step numbers, timestamps,
dates, turn counters or run ids, and no iteration over unsorted containers. It is re-rendered in full every step;
training and inference call the same functions.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from qet.core.canonical import JSON, action_string, canon_json
from qet.core.contracts import Contracts, as_contracts
from qet.core.reduce import Entry, canonical_entries, reduce

SYSTEM_TEXT = (
    "You are a tool-using agent. Read the task and the completed calls, then output the next action "
    'as one JSON object {"tool": ..., "args": {...}} and nothing else. '
    "Call finish when the task is complete."
)  # SPEC §5.2, D-003

VARIANTS = ("quotient", "chronological", "native_multiturn")  # SPEC §5.7, D-004
NONE_LINE = "  (none)"

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass(frozen=True)
class Prompt:
    system: str
    user: str

    def messages(self) -> list[dict[str, str]]:
        """Chat messages of the prompt (D-002): the system text, then one user message."""
        return [{"role": "system", "content": self.system}, {"role": "user", "content": self.user}]


# ----------------------------------------------------------------------------- truncation (SPEC §5.4)
@dataclass(frozen=True)
class Truncation:
    """Per-environment content limits (None = no limit), identical in training and inference."""

    max_list_items: int | None = None
    max_str_chars: int | None = None

    @classmethod
    def from_json(cls, data: Mapping[str, Any] | Truncation | None) -> Truncation:
        if data is None:
            return cls()
        if isinstance(data, Truncation):
            return data
        unknown = sorted(set(data) - {"max_list_items", "max_str_chars"})
        if unknown:
            raise ValueError(f"unknown truncation settings {unknown}")
        for name in ("max_list_items", "max_str_chars"):
            v = data.get(name)
            if v is not None and (isinstance(v, bool) or not isinstance(v, int)):
                raise TypeError(f"{name} must be an int or null, got {v!r}")
            if v is not None and v < 0:
                raise ValueError(f"{name} must be non-negative, got {v}")
        return cls(data.get("max_list_items"), data.get("max_str_chars"))


def truncate(x: JSON, t: Truncation) -> JSON:
    """Lists keep their first `max_list_items` items plus "<+N more>"; strings longer than `max_str_chars` keep that
    prefix plus "<+M chars>". Object keys are never truncated."""
    k, m = t.max_list_items, t.max_str_chars
    if isinstance(x, dict):
        return {key: truncate(v, t) for key, v in x.items()}
    if isinstance(x, list):
        items = [truncate(v, t) for v in (x if k is None else x[:k])]
        if k is not None and len(x) > k:
            items.append(f"<+{len(x) - k} more>")
        return items
    if isinstance(x, str) and m is not None and len(x) > m:
        return x[:m] + f"<+{len(x) - m} chars>"
    return x


# ----------------------------------------------------------------------------- sections
def value_lines(name: str, obs: JSON) -> list[str]:
    """[Available values] lines of one entry (SPEC §5.3): leaves in canonical order, `.key` or `["key"]` for object
    keys, `[i]` for list items; scalars and nested empty containers are leaves; an empty observation gives none."""
    lines: list[str] = []

    def walk(x: JSON, path: str) -> None:
        if isinstance(x, dict) and x:
            for key in sorted(x):
                seg = f".{key}" if _IDENTIFIER.fullmatch(key) else f"[{json.dumps(key, ensure_ascii=False)}]"
                walk(x[key], path + seg)
        elif isinstance(x, list) and x:
            for i, v in enumerate(x):
                walk(v, f"{path}[{i}]")
        else:
            lines.append(f"  {name}{path} = {canon_json(x)}")

    if not (isinstance(obs, (dict, list)) and not obs):
        walk(obs, "")
    return lines


def _tool_lines(tools: Sequence[Mapping[str, Any]]) -> list[str]:
    """One line per tool, sorted by name, parameters in schema order."""
    lines = []
    for t in sorted(tools, key=lambda t: t["name"]):
        params = t["params"]
        if not isinstance(params, (list, tuple)) or not all(isinstance(p, str) for p in params):
            raise TypeError(f"params of tool {t['name']!r} must be a list of names, got {params!r}")
        lines.append(f"  {t['name']}({', '.join(params)}): {t['description']}")
    return lines or [NONE_LINE]


def _task_lines(task: Mapping[str, Any]) -> list[str]:
    return [f"[Task] {task['request']}", "[Tools]", *_tool_lines(task["tools"])]


def _entries(
    history: Sequence[Mapping[str, Any]], contracts: Contracts, variant: str, t: Truncation
) -> tuple[list[Entry], list[JSON]]:
    if variant == "quotient":
        entries = reduce(history, contracts)
    elif variant in ("chronological", "native_multiturn"):
        entries = canonical_entries(history, contracts)
    else:
        raise ValueError(f"unknown renderer {variant!r}; expected one of {VARIANTS}")
    return entries, [truncate(e.obs, t) for e in entries]


# ----------------------------------------------------------------------------- renderers
def render(
    task: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    contracts: Contracts | Mapping[str, Any] | None,
    variant: str = "quotient",
    truncation: Mapping[str, Any] | Truncation | None = None,
) -> Prompt:
    """The prompt of the quotient interface (SPEC §5.2) or of the chronological baseline (§5.7, configuration 1).

    `history` is the raw history in execution order ({tool, args, obs}). The native_multiturn baseline is a list of
    chat messages: use `render_messages`.
    """
    if variant not in ("quotient", "chronological"):
        raise ValueError(f"render() builds the quotient and chronological prompts, not {variant!r}")
    entries, obs = _entries(history, as_contracts(contracts), variant, Truncation.from_json(truncation))
    lines = _task_lines(task)
    if variant == "quotient":
        lines.append("[Completed calls]")
        lines += [f"  {e.name} -> {canon_json(o)}" for e, o in zip(entries, obs, strict=True)] or [NONE_LINE]
    else:
        lines.append("[History]")
        calls = [f"  {i}. {e.name} -> {canon_json(o)}" for i, (e, o) in enumerate(zip(entries, obs, strict=True), 1)]
        lines += calls or [NONE_LINE]
    lines.append("[Available values]")
    lines += [ln for e, o in zip(entries, obs, strict=True) for ln in value_lines(e.name, o)] or [NONE_LINE]
    lines.append("[Next action]")
    return Prompt(SYSTEM_TEXT, "\n".join(lines))


def render_messages(
    task: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    contracts: Contracts | Mapping[str, Any] | None,
    variant: str = "quotient",
    truncation: Mapping[str, Any] | Truncation | None = None,
) -> list[dict[str, str]]:
    """Chat messages for any renderer (D-002, D-004).

    quotient / chronological: [system, user]. native_multiturn (ablation a1): the system text; one user message
    with the [Task] and [Tools] blocks; then per step an assistant message with the action string and a user message
    `[Observation] <canon_json(obs)>` (canonical, volatile fields deleted, truncated); no [Available values].
    """
    if variant != "native_multiturn":
        return render(task, history, contracts, variant, truncation).messages()
    entries, obs = _entries(history, as_contracts(contracts), variant, Truncation.from_json(truncation))
    msgs = [{"role": "system", "content": SYSTEM_TEXT}, {"role": "user", "content": "\n".join(_task_lines(task))}]
    for e, o in zip(entries, obs, strict=True):
        msgs.append({"role": "assistant", "content": action_string(e.tool, e.args)})
        msgs.append({"role": "user", "content": f"[Observation] {canon_json(o)}"})
    return msgs

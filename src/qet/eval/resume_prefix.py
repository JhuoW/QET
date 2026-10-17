"""Resume@prefix (EVAL_PLAN §6): finish an evaluation task from legal states of its compiled graph.

Start conditions, balanced across depths 1 .. n-1: on-path (a prefix of the demo order) and off-path (a legal set the
demo never visited), plus the empty start. A depth enters only when both an on-path and an off-path set exist there
(a chain has none). The environment is restored by executing a legal order of the prefix in a fresh episode, so every
observation the agent sees is real; the agent then continues with no graph, ready set or future output.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from qet.core.compile import GraphRecord
from qet.core.lattice import node_masks, random_topological_order
from qet.envs.base import Environment, run_calls
from qet.infer.agent import Policy, run_episode
from qet.train.chat import ChatFormat


@dataclass(frozen=True)
class Start:
    condition: str  # "empty", "on_path" or "off_path"
    depth: int
    order: tuple[int, ...]  # the nodes executed to restore the state, in a legal order


def starts(record: GraphRecord, seed: int, tries: int = 200) -> list[Start]:
    """The empty start, then for each depth with both: the demo's prefix and a legal set the demo never visited
    (the prefix of a random topological order, seeded by task and run)."""
    n = record.n
    masks = node_masks(n, record.preds())
    rng = random.Random(f"resume-prefix\x1f{record.task_id}\x1f{seed}")
    samples = [random_topological_order(masks, rng) for _ in range(tries)] if n else []
    out = [Start("empty", 0, ())]
    for d in range(1, n):
        on = tuple(range(d))  # node ids are demo positions, so the demo order is 0 .. n-1
        off = next((tuple(o[:d]) for o in samples if set(o[:d]) != set(on)), None)
        if off is not None:
            out += [Start("on_path", d, on), Start("off_path", d, off)]
    return out


def restore(env: Environment, record: GraphRecord, start: Start) -> list[dict[str, Any]]:
    """Execute the start's prefix in `env` and return the executed steps (real observations)."""
    return run_calls(env, [{"tool": record.nodes[v].tool, "args": record.nodes[v].args} for v in start.order])


def evaluate_resume(
    records: list[GraphRecord],
    tasks_by_id: Mapping[str, Any],
    make_env: Callable[[Any], Environment],
    policy: Policy,
    fmt: ChatFormat,
    *,
    decoding: str,
    renderer: str,
    truncation: Mapping[str, Any] | None,
    run_seed: int,
    on_episode: Callable[[dict[str, Any]], None],
) -> None:
    for record in records:
        for start in starts(record, run_seed):
            env = make_env(tasks_by_id[record.task_id])
            prefix = restore(env, record, start)
            ep = run_episode(env, policy, fmt, renderer=renderer, truncation=truncation, prefix=prefix,
                             run_seed=run_seed)  # fmt: skip
            on_episode({"decoding": decoding, "condition": start.condition, "depth": start.depth,
                        "prefix_order": list(start.order), **ep.to_json()})  # fmt: skip

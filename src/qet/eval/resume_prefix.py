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
from qet.infer.agent import EpisodeState, Policy, run_episode, run_rolling
from qet.train.chat import ChatFormat


@dataclass(frozen=True)
class Start:
    condition: str  # "empty", "on_path" or "off_path"
    depth: int
    order: tuple[int, ...]  # the nodes executed to restore the state, in a legal order


START_MODES = ("all_depths", "one_depth")


def starts(record: GraphRecord, seed: int, tries: int = 200, per_task: str = "all_depths") -> list[Start]:
    """The empty start, then for each depth with both: the demo's prefix and a legal set the demo never visited
    (the prefix of a random topological order, seeded by task and run). With per_task="one_depth", three starts per
    task (EVAL_PLAN §6.3 read literally): the empty start and the on-path and off-path starts of one of those depths,
    drawn uniformly per task, so depths balance over tasks."""
    if per_task not in START_MODES:
        raise ValueError(f"per_task must be one of {START_MODES}")
    n = record.n
    masks = node_masks(n, record.preds())
    rng = random.Random(f"resume-prefix\x1f{record.task_id}\x1f{seed}")
    samples = [random_topological_order(masks, rng) for _ in range(tries)] if n else []
    pairs = []
    for d in range(1, n):
        on = tuple(range(d))  # node ids are demo positions, so the demo order is 0 .. n-1
        off = next((tuple(o[:d]) for o in samples if set(o[:d]) != set(on)), None)
        if off is not None:
            pairs.append([Start("on_path", d, on), Start("off_path", d, off)])
    if per_task == "one_depth" and pairs:
        pairs = [random.Random(f"resume-depth\x1f{record.task_id}\x1f{seed}").choice(pairs)]
    return [Start("empty", 0, ())] + [s for pair in pairs for s in pair]


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
    start_seed: int = 0,
    width: int = 256,
    start_mode: str = "all_depths",
) -> None:
    """`start_seed` fixes the starts (the evaluation seed: the same for every training seed); `run_seed` decoding. A
    batch policy runs up to `width` episodes per round; the rows come out in the same order."""
    work = [(record, start) for record in records for start in starts(record, start_seed, per_task=start_mode)]

    def begin(record: GraphRecord, start: Start) -> EpisodeState:
        env = make_env(tasks_by_id[record.task_id])
        prefix = restore(env, record, start)
        return EpisodeState(env, fmt, renderer=renderer, truncation=truncation, prefix=prefix, run_seed=run_seed)

    def emit(i: int, ep: Any) -> None:
        start = work[i][1]
        on_episode({"decoding": decoding, "condition": start.condition, "depth": start.depth,
                    "prefix_order": list(start.order), **ep.to_json()})  # fmt: skip

    if hasattr(policy, "generate_batch"):
        run_rolling([lambda r=r, s=s: begin(r, s) for r, s in work], policy, width, emit)
        return
    for i, (record, start) in enumerate(work):
        env = make_env(tasks_by_id[record.task_id])
        prefix = restore(env, record, start)
        emit(i, run_episode(env, policy, fmt, renderer=renderer, truncation=truncation, prefix=prefix,
                            run_seed=run_seed))  # fmt: skip

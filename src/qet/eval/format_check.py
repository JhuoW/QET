"""The format-error rate of D-010: how often the policy's one action at a state is not a valid action (D-006: no
end-of-turn token within the token limit, or text that `parse_action` rejects, such as non-JSON or an unknown tool).

It is measured at sampled states along each task's demonstration: the empty state and random prefixes of the demo
order, rendered by the configuration's renderer exactly as training and inference render them. The warm-up length is
the number of warm-up steps after which this rate on dev falls below 1% (D-010).
"""

from __future__ import annotations

import random
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from qet.core.render import render_messages
from qet.infer.agent import Policy, step_seed
from qet.infer.decoding import Invalid, parse_action
from qet.train.chat import ChatFormat


def reason_kind(reason: str) -> str:
    """The kind of an invalid action without its details ("not one JSON object: Expecting value ..." -> "not one JSON
    object", "unknown tool 'x'" -> "unknown tool")."""
    return re.split(r": | '| \"", reason, maxsplit=1)[0]


def format_errors(
    tasks: Sequence[Any],
    records: Sequence[Any],
    policy: Policy,
    fmt: ChatFormat,
    contracts: Mapping[str, Any],
    *,
    renderer: str,
    truncation: Mapping[str, Any] | None = None,
    states_per_task: int = 2,
    seed: int = 0,
    run_seed: int = 0,
) -> dict[str, Any]:
    """One action at each of `states_per_task` states per task: the empty state, then demo-order prefixes of random
    depth 1..n (seeded by `seed` and the task id, so every checkpoint sees the same states)."""
    if states_per_task < 1:
        raise ValueError("states_per_task must be at least 1")
    by_id = {t.task_id: t for t in tasks}
    kinds: Counter[str] = Counter()
    checked = 0
    for rec in records:
        x = by_id[rec.task_id].public()
        tool_names = [t["name"] for t in x["tools"]]
        steps = rec.steps()
        rng = random.Random(f"format\x1f{seed}\x1f{rec.task_id}")
        depths = [0] + [rng.randint(1, rec.n) for _ in range(states_per_task - 1)] if rec.n else [0] * states_per_task
        for i, k in enumerate(depths):
            ids = fmt.prompt_ids(render_messages(x, steps[:k], contracts, renderer, truncation))
            gen = policy.generate(ids, step_seed(run_seed, rec.task_id, i, k))
            raw = fmt.decode(gen.ids[:-1] if gen.ended else gen.ids)
            parsed = parse_action(raw, tool_names) if gen.ended else Invalid("no end of turn within the token limit")
            checked += 1
            if isinstance(parsed, Invalid):
                kinds[reason_kind(parsed.reason)] += 1
    errors = sum(kinds.values())
    return {"states": checked, "errors": errors, "rate": errors / checked if checked else None,
            "kinds": dict(sorted(kinds.items()))}  # fmt: skip

"""Evaluation episodes (SPEC §7, EVAL_PLAN §7): every task under every decoding setting, greedy once and sampling with
independent rollouts (8 by default; never best-of-k). One JSON row per episode."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from qet.config import EvalSettings
from qet.envs.base import Environment
from qet.infer.agent import Policy, run_episode
from qet.infer.decoding import DecodeConfig
from qet.train.chat import ChatFormat


def decoding_settings(settings: EvalSettings) -> list[tuple[str, DecodeConfig, int]]:
    """(name, decoding, rollouts) for each configured setting."""
    out = []
    for d in settings.decoding:
        mode = d["mode"]
        rollouts = int(d.get("samples", 1 if mode == "greedy" else 8))
        if rollouts < 1 or (mode == "greedy" and rollouts != 1):
            raise ValueError("greedy decoding has one rollout; sampling at least one")
        cfg = DecodeConfig(mode, float(d.get("temperature", 1.0)), float(d.get("top_p", 1.0)), settings.max_new_tokens)
        name = "greedy" if mode == "greedy" else f"sample-t{cfg.temperature:g}-p{cfg.top_p:g}"
        out.append((name, cfg, rollouts))
    if len({n for n, _, _ in out}) != len(out):
        raise ValueError("two decoding settings with the same name")
    return out


def evaluate_tasks(
    tasks: Iterable[Any],
    make_env: Callable[[Any], Environment],
    policy_for: Callable[[DecodeConfig], Policy],
    fmt: ChatFormat,
    settings: EvalSettings,
    *,
    renderer: str,
    truncation: Mapping[str, Any] | None,
    run_seed: int,
    on_episode: Callable[[dict[str, Any]], None],
) -> None:
    """Run every task from the empty state under every decoding setting."""
    tasks = list(tasks)
    for name, cfg, rollouts in decoding_settings(settings):
        policy = policy_for(cfg)
        for task in tasks:
            for r in range(rollouts):
                ep = run_episode(make_env(task), policy, fmt, renderer=renderer, truncation=truncation, rollout=r,
                                 run_seed=run_seed)  # fmt: skip
                on_episode({"decoding": name, "condition": "standard", "depth": 0, **ep.to_json()})


def rows_by_task(rows: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        out.setdefault(r["task_id"], []).append(r)
    return out

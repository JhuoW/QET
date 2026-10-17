"""The inference agent (SPEC §7, Algorithm 2): re-render the real history every step, decode one action, execute it.

Inference never touches the graph, the lattice, B or the weights; it calls the same Reduce + Render code and the same
chat mapping as training. Each step logs the prompt's sha256, the raw output, the parsed action, the observation, an
error flag and token counts. An invalid action ends the episode (D-006); so does a tool call beyond the step budget.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from qet.core.canonical import JSON
from qet.core.render import Truncation, render_messages
from qet.envs.base import Environment
from qet.infer.decoding import Generation, Invalid, parse_action
from qet.train.chat import ChatFormat

ENDS = ("finish", "invalid_action", "budget_exhausted", "render_error")


def prompt_sha256(ids: Sequence[int]) -> str:
    """The hash logged per step: the exact token ids the model sees."""
    return hashlib.sha256(json.dumps(list(ids)).encode()).hexdigest()


def step_seed(run_seed: int, task_id: str, rollout: int, step: int) -> int:
    """Seed of one decoding step: rollouts are reproducible and independent of each other (never best-of-k)."""
    digest = hashlib.sha256(f"qet-decode\x1f{run_seed}\x1f{task_id}\x1f{rollout}\x1f{step}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


class Policy(Protocol):
    def generate(self, prompt_ids: Sequence[int], seed: int) -> Generation: ...


class ScriptedPolicy:
    """Emits the given action strings in turn, each followed by the end of turn (tests and audits)."""

    def __init__(self, fmt: ChatFormat, actions: Sequence[str]):
        self.fmt, self.actions, self.calls = fmt, list(actions), 0

    def generate(self, prompt_ids: Sequence[int], seed: int) -> Generation:
        action = self.actions[self.calls]
        self.calls += 1
        return Generation(self.fmt.encode(action) + [self.fmt.eot_id], True, 0.0)


@dataclass
class StepLog:
    prompt_sha256: str
    prompt_tokens: int
    output_tokens: int
    raw_output: str
    action: dict[str, Any] | None = None  # the parsed canonical action
    observation: JSON = None
    error: str | None = None  # why the action was invalid, "observation error", "budget exhausted", ...


@dataclass
class Episode:
    task_id: str
    rollout: int
    end: str  # one of ENDS
    success: bool
    answer: JSON
    calls: int  # tool calls the agent executed (the prefix not included)
    prefix: int  # steps executed before the agent started (Resume@prefix)
    steps: list[StepLog] = field(default_factory=list)
    env_failures: list[dict[str, Any]] = field(default_factory=list)  # the environment's own error log

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _close(episode: Episode, env: Environment) -> Episode:
    """Attach the environment's own error log (e.g. FuncBenchGen's failure reasons) for the error taxonomy."""
    episode.env_failures = [dict(f) for f in getattr(env, "failures", None) or []]
    return episode


def run_episode(
    env: Environment,
    policy: Policy,
    fmt: ChatFormat,
    *,
    renderer: str = "quotient",
    truncation: Mapping[str, Any] | Truncation | None = None,
    prefix: Sequence[Mapping[str, Any]] = (),
    budget: int | None = None,
    rollout: int = 0,
    run_seed: int = 0,
) -> Episode:
    """Algorithm 2 on one task. `prefix` holds steps {tool, args, obs} already executed in `env` (Resume@prefix
    restores the state by executing a legal order of the prefix); the budget then counts the agent's own calls and
    defaults to what remains of the environment's."""
    x = env.task()
    contracts = env.contracts()
    tool_names = [t["name"] for t in x["tools"]]
    finish = x.get("finish", "finish")
    history = [dict(s) for s in prefix]
    budget = max(0, env.step_budget() - len(prefix)) if budget is None else budget
    if budget < 0:
        raise ValueError(f"the step budget must be >= 0, got {budget}")
    episode = Episode(env.task_id, rollout, "", False, None, 0, len(prefix))
    for t in range(budget + 1):
        try:
            ids = fmt.prompt_ids(render_messages(x, history, contracts, renderer, truncation))
        except ValueError as e:  # an observation the prompt cannot carry (D-022d): adapters should prevent it
            episode.end = "render_error"
            episode.steps.append(StepLog("", 0, 0, "", error=str(e)))
            return _close(episode, env)
        gen = policy.generate(ids, step_seed(run_seed, env.task_id, rollout, t))
        raw = fmt.decode(gen.ids[:-1] if gen.ended else gen.ids)
        log = StepLog(prompt_sha256(ids), len(ids), len(gen.ids), raw)
        episode.steps.append(log)
        parsed = parse_action(raw, tool_names) if gen.ended else Invalid("no end of turn within the token limit")
        if isinstance(parsed, Invalid):
            log.error = parsed.reason
            episode.end = "invalid_action"
            return _close(episode, env)
        log.action = {"tool": parsed.tool, "args": parsed.args}
        if parsed.tool == finish:
            episode.answer = parsed.args.get("answer")
            episode.success = bool(env.check(episode.answer))
            episode.end = "finish"
            return _close(episode, env)
        if episode.calls >= budget:
            break
        obs = env.step(parsed.tool, parsed.args)
        episode.calls += 1
        log.observation = obs
        if isinstance(obs, dict) and "error" in obs:
            log.error = "observation error"
        history.append({"tool": parsed.tool, "args": parsed.args, "obs": obs})
    episode.steps[-1].error = "budget exhausted"
    episode.end = "budget_exhausted"
    return _close(episode, env)

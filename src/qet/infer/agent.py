"""The inference agent (SPEC §7, Algorithm 2): re-render the real history every step, decode one action, execute it.

Inference never touches the graph, the lattice, B or the weights; it calls the same Reduce + Render code and the same
chat mapping as training. Each step logs the prompt's sha256, the raw output, the parsed action, the observation, an
error flag and token counts. An invalid action ends the episode (D-006); so does a tool call beyond the step budget.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
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


class EpisodeState:
    """Algorithm 2 on one task as a state machine, so that one driver (`run_episode`) or a lockstep batch of them
    (`run_episodes`) applies exactly the same rules. `prompt()` renders the real history (None once the episode is
    over), `seed()` is the current step's decoding seed, `advance()` applies one generation. `prefix` holds steps
    {tool, args, obs} already executed in `env` (Resume@prefix restores the state by executing a legal order of the
    prefix); the budget then counts the agent's own calls and defaults to what remains of the environment's."""

    def __init__(
        self,
        env: Environment,
        fmt: ChatFormat,
        *,
        renderer: str = "quotient",
        truncation: Mapping[str, Any] | Truncation | None = None,
        prefix: Sequence[Mapping[str, Any]] = (),
        budget: int | None = None,
        rollout: int = 0,
        run_seed: int = 0,
        show_prefix: bool = True,
    ):
        """`show_prefix=False` is Resume@prefix's no-history condition for chronological baselines (EVAL_PLAN §6.5):
        the environment has executed the prefix, but the prompt starts without those calls."""
        self.env, self.fmt, self.renderer, self.truncation = env, fmt, renderer, truncation
        self.rollout, self.run_seed = rollout, run_seed
        self.x = env.task()
        self.contracts = env.contracts()
        self.tool_names = [t["name"] for t in self.x["tools"]]
        self.finish = self.x.get("finish", "finish")
        self.history = [dict(s) for s in prefix] if show_prefix else []
        self.budget = max(0, env.step_budget() - len(prefix)) if budget is None else budget
        if self.budget < 0:
            raise ValueError(f"the step budget must be >= 0, got {self.budget}")
        self.episode = Episode(env.task_id, rollout, "", False, None, 0, len(prefix))
        self.t = 0
        self.done = False
        self._ids: list[int] | None = None

    def prompt(self) -> list[int] | None:
        if self.done:
            return None
        if self._ids is None:
            try:
                self._ids = self.fmt.prompt_ids(
                    render_messages(self.x, self.history, self.contracts, self.renderer, self.truncation)
                )
            except ValueError as e:  # an observation the prompt cannot carry (D-022d): adapters should prevent it
                self.episode.end = "render_error"
                self.episode.steps.append(StepLog("", 0, 0, "", error=str(e)))
                self._end()
                return None
        return self._ids

    def seed(self) -> int:
        return step_seed(self.run_seed, self.env.task_id, self.rollout, self.t)

    def advance(self, gen: Generation) -> None:
        ids, episode = self._ids, self.episode
        if self.done or ids is None:
            raise RuntimeError("advance() needs the prompt of a running episode")
        self._ids = None
        raw = self.fmt.decode(gen.ids[:-1] if gen.ended else gen.ids)
        log = StepLog(prompt_sha256(ids), len(ids), len(gen.ids), raw)
        episode.steps.append(log)
        parsed = parse_action(raw, self.tool_names) if gen.ended else Invalid("no end of turn within the token limit")
        if isinstance(parsed, Invalid):
            log.error = parsed.reason
            episode.end = "invalid_action"
            return self._end()
        log.action = {"tool": parsed.tool, "args": parsed.args}
        if parsed.tool == self.finish:
            episode.answer = parsed.args.get("answer")
            episode.success = bool(self.env.check(episode.answer))
            episode.end = "finish"
            return self._end()
        if episode.calls >= self.budget:  # a tool call beyond the budget ends the episode
            episode.steps[-1].error = "budget exhausted"
            episode.end = "budget_exhausted"
            return self._end()
        obs = self.env.step(parsed.tool, parsed.args)
        episode.calls += 1
        log.observation = obs
        if isinstance(obs, dict) and "error" in obs:
            log.error = "observation error"
        self.history.append({"tool": parsed.tool, "args": parsed.args, "obs": obs})
        self.t += 1

    def _end(self) -> None:
        self.done = True
        _close(self.episode, self.env)


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
    """Algorithm 2 on one task (see `EpisodeState` for the arguments)."""
    state = EpisodeState(env, fmt, renderer=renderer, truncation=truncation, prefix=prefix, budget=budget,
                         rollout=rollout, run_seed=run_seed)  # fmt: skip
    while (ids := state.prompt()) is not None:
        state.advance(policy.generate(ids, state.seed()))
    return state.episode


class BatchPolicy(Protocol):
    def generate_batch(self, prompts: Sequence[Sequence[int]], seeds: Sequence[int]) -> list[Generation]: ...


def run_episodes(states: Sequence[EpisodeState], policy: Policy | BatchPolicy) -> list[Episode]:
    """Many episodes in lockstep: each round decodes the next action of every running episode in one batch. With
    per-step seeds, each episode's outputs do not depend on which episodes share its batches, up to the backend's
    numerics (D-026b)."""
    batch = getattr(policy, "generate_batch", None)
    while live := [(s, ids) for s in states if (ids := s.prompt()) is not None]:
        prompts, seeds = [ids for _, ids in live], [s.seed() for s, _ in live]
        gens = (
            batch(prompts, seeds) if batch else [policy.generate(i, sd) for i, sd in zip(prompts, seeds, strict=True)]
        )
        for (s, _), gen in zip(live, gens, strict=True):
            s.advance(gen)
    return [s.episode for s in states]


def run_rolling(
    jobs: Sequence[Callable[[], EpisodeState]],
    policy: Policy | BatchPolicy,
    width: int,
    on_done: Callable[[int, Episode], None],
) -> None:
    """Episodes in lockstep rounds with at most `width` running: a finished episode's place goes to the next job, so
    long episodes do not leave the batch idle. `jobs[i]()` starts job i (its environment is made then); `on_done(i,
    episode)` is called in job order. Same rules and outputs as `run_episodes`."""
    if width < 1:
        raise ValueError("width must be at least 1")
    batch = getattr(policy, "generate_batch", None)
    queue = iter(enumerate(jobs))
    live: list[tuple[int, EpisodeState]] = []
    finished: dict[int, Episode] = {}
    emitted = 0
    more = True
    while more or live:
        while more and len(live) < width:
            nxt = next(queue, None)
            if nxt is None:
                more = False
            else:
                live.append((nxt[0], nxt[1]()))
        round_ = [(i, s, ids) for i, s in live if (ids := s.prompt()) is not None]
        if round_:
            prompts, seeds = [ids for _, _, ids in round_], [s.seed() for _, s, _ in round_]
            gens = (
                batch(prompts, seeds)
                if batch
                else [policy.generate(p, sd) for p, sd in zip(prompts, seeds, strict=True)]
            )
            for (_, s, _), gen in zip(round_, gens, strict=True):
                s.advance(gen)
        for i, s in live:
            if s.done:
                finished[i] = s.episode
        live = [(i, s) for i, s in live if not s.done]
        while emitted in finished:
            on_done(emitted, finished.pop(emitted))
            emitted += 1

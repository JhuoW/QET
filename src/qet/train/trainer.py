"""The two-pass training step and loop (SPEC §6.5, Algorithm 1), shared by every configuration (SPEC §8).

A step on a batch of graph records: lattice packs (exact, or the sub-lattice above S_max); pass 1 without gradients,
in eval mode, for the configurations whose weights depend on the model; the configuration's edge weights (production
code in `qet.core.weights`); pass 2, weighted SFT with the weights detached, summed over targets and averaged over the
graphs of the batch (D-014), micro-batched and accumulated; one optimizer step. No parameter changes between pass 1
and pass 2. The logged objective is the configuration's own (SPEC §8), never the pass-2 loss.
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import torch

from qet.core.canonical import action_string
from qet.core.compile import GraphRecord, canonical_order
from qet.core.contracts import Contracts, as_contracts
from qet.core.lattice import (
    END,
    S_MAX,
    EdgeKey,
    Lattice,
    build_lattice,
    full_lattice,
    lattice_from_states,
    log_completion,
    members,
    prefix_states,
    sample_orders,
    start_weights,
    sublattice,
    sublattice_seed,
)
from qet.core.reduce import name_calls
from qet.core.render import Truncation, render_messages
from qet.core.weights import EdgeWeights, edge_weights, logged_objective
from qet.train.chat import ChatFormat
from qet.train.methods import Method
from qet.train.scoring import Pair, TokenCounts, count_tokens, forward_logp, micro_batches, score

Item = tuple[int, EdgeKey]  # (index of the graph in the batch, edge)


# ----------------------------------------------------------------------------- lattice packs (SPEC §1)
@dataclass
class LatticePack:
    """The states, edges, prompts and targets of one graph in one epoch; independent of the model."""

    task_id: str
    lat: Lattice
    order: list[int] | None = None  # path_sft: the order trained on
    paths: list[list[int]] | None = None  # sampled_paths: the paths marginalized over
    prompt_ids: dict[int, tuple[int, ...]] = field(default_factory=dict)  # per state
    target_ids: dict[int, tuple[int, ...]] = field(default_factory=dict)  # per node; END is the finish action

    def pair(self, edge: EdgeKey) -> Pair:
        state, v = edge
        return Pair(self.prompt_ids[state], self.target_ids[v])


def path_edges(order: Sequence[int]) -> list[EdgeKey]:
    states = prefix_states(order)
    return [(s, v) for s, v in zip(states, order, strict=False)] + [(states[-1], END)]


class PackBuilder:
    """Lattice packs of one dataset under one configuration. A prompt is rendered and tokenized once per graph and
    state and reused in later epochs, whose sub-lattices are resampled (SPEC §6.6)."""

    def __init__(
        self,
        fmt: ChatFormat,
        contracts: Contracts | Mapping[str, Any] | None,
        method: Method,
        *,
        run_seed: int,
        truncation: Mapping[str, Any] | Truncation | None = None,
        k: int | None = None,
        s_max: int = S_MAX,
    ):
        """`run_seed` seeds the sub-lattices with (graph id, epoch); `train` requires it to be the run's seed. `k` is
        the number of sampled orders: of the sub-lattice above S_max (configurations 3-6), of every graph's
        sub-lattice (a4), or of the marginalized paths (a3, default K = 4)."""
        if method.force_sublattice and k is None:
            raise ValueError(f"configuration {method.config_id} needs the number of sub-lattice orders K")
        self.fmt = fmt
        self.contracts = as_contracts(contracts)
        self.method = method
        self.truncation = Truncation.from_json(truncation)
        self.k, self.s_max, self.run_seed = k, s_max, run_seed
        self._prompts: dict[tuple[str, int], tuple[int, ...]] = {}
        self._targets: dict[tuple[str, int], tuple[int, ...]] = {}

    def lattice(self, rec: GraphRecord, epoch: int) -> tuple[Lattice, list[int] | None, list[list[int]] | None]:
        n, preds = rec.n, [sorted(p) for p in rec.preds()]
        m = self.method
        if m.lattice == "path":
            if m.order == "demo":
                order = list(range(n))
            else:  # a2: repeatedly the ready node with the smallest Reduce key
                order = canonical_order(preds, [key for _, key in name_calls((nd.tool, nd.args) for nd in rec.nodes)])
            return lattice_from_states(n, preds, prefix_states(order)), order, None
        seed = sublattice_seed(rec.task_id, epoch, self.run_seed)  # shared by configurations 3-6 and a4 (D-013)
        if m.lattice == "paths":  # a3: K from the run config like a4, so equal K gives a4's orders (D-023b)
            paths = sample_orders(n, preds, self.k if self.k is not None else m.paths, seed)
            return sublattice(n, preds, paths), None, paths
        if self.k is None:
            lat = full_lattice(n, preds, cap=self.s_max)
            if lat is None:
                raise ValueError(f"graph {rec.task_id} has more than {self.s_max} states: set K in the run config")
            return lat, None, None
        s_max = 0 if m.force_sublattice else self.s_max
        return build_lattice(n, preds, k=self.k, seed=seed, s_max=s_max), None, None

    def build(self, rec: GraphRecord, epoch: int = 0) -> LatticePack:
        lat, order, paths = self.lattice(rec, epoch)
        pack = LatticePack(rec.task_id, lat, order, paths)
        task = {"request": rec.request, "tools": rec.tools}
        steps = rec.steps()
        for s in lat.states:
            key = (rec.task_id, s)
            if key not in self._prompts:
                history = [steps[i] for i in members(s)]  # the demo order restricted to the state: a legal history
                msgs = render_messages(task, history, self.contracts, self.method.renderer, self.truncation)
                self._prompts[key] = tuple(self.fmt.prompt_ids(msgs))
            pack.prompt_ids[s] = self._prompts[key]
        # Every node lies on an edge inside S (S holds complete orders), so these targets also cover the B+ diagnostic.
        for v in sorted({v for _, v in lat.edges}):
            key = (rec.task_id, v)
            if key not in self._targets:
                tool, args = (
                    (rec.final["tool"], rec.final["args"]) if v == END else (rec.nodes[v].tool, rec.nodes[v].args)
                )
                self._targets[key] = tuple(self.fmt.target_ids(action_string(tool, args)))
            pack.target_ids[v] = self._targets[key]
        return pack


# ----------------------------------------------------------------------------- scorers
class Scorer(Protocol):
    """The policy as seen by the training step: log pi of pack edges, with or without gradients."""

    def parameters(self) -> list[torch.nn.Parameter]: ...

    def logp(
        self, packs: Sequence[LatticePack], items: Sequence[Item], *, grad: bool
    ) -> tuple[torch.Tensor, TokenCounts]: ...

    def micro_batches(self, packs: Sequence[LatticePack], items: Sequence[Item]) -> list[list[int]]: ...


class LMScorer:
    """A causal language model, optionally LoRA-wrapped, scoring the packs' (prompt, target) pairs."""

    def __init__(
        self, model: torch.nn.Module, pad_id: int, max_tokens: int = 16384, max_target_tokens: int | None = None
    ):
        self.model, self.pad_id = model, pad_id
        self.max_tokens, self.max_target_tokens = max_tokens, max_target_tokens

    def parameters(self) -> list[torch.nn.Parameter]:
        return [p for p in self.model.parameters() if p.requires_grad]

    def logp(
        self, packs: Sequence[LatticePack], items: Sequence[Item], *, grad: bool
    ) -> tuple[torch.Tensor, TokenCounts]:
        pairs = [packs[g].pair(e) for g, e in items]
        if grad:  # one micro-batch of pass 2; the caller runs backward and times the whole pass
            with torch.enable_grad():
                return forward_logp(self.model, pairs, self.pad_id), count_tokens(pairs, seconds=0.0)
        values, counts = score(
            self.model, pairs, pad_id=self.pad_id, max_tokens=self.max_tokens, max_target_tokens=self.max_target_tokens
        )
        return torch.tensor(values, dtype=torch.float64), counts

    def micro_batches(self, packs: Sequence[LatticePack], items: Sequence[Item]) -> list[list[int]]:
        pairs = [packs[g].pair(e) for g, e in items]
        return micro_batches(pairs, self.max_tokens, self.max_target_tokens)


def add_lora(model: torch.nn.Module, *, rank: int = 64, alpha: int = 128, seed: int = 0) -> torch.nn.Module:
    """LoRA on all linear layers with dropout 0 (D-009), returned in eval mode: both passes run in eval mode."""
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(seed)
    cfg = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.0, target_modules="all-linear", task_type="CAUSAL_LM")
    return get_peft_model(model, cfg).eval()


# ----------------------------------------------------------------------------- the training step
@dataclass
class StepResult:
    objective: float  # mean over the batch's graphs of the logged objective (SPEC §8)
    objectives: list[float]  # per graph
    scored: TokenCounts  # pass 1
    gradient: TokenCounts  # pass 2
    diagnostic: TokenCounts  # the B+ diagnostic's extra scoring; not part of the training budget
    edges_scored: int
    edges_trained: int
    seconds: dict[str, float]
    grad_norm: float | None
    bplus_gap: list[float] | None  # per graph, qet with plus=True: sum of mu * (log B+ - log B_S), the error bound
    pass2_loss: float  # for tests only: never logged, never used for early stopping or selection (SPEC §6.5)


def pass1_edges(pack: LatticePack, method: Method) -> list[EdgeKey]:
    """The edges whose log pi the weights need: the paths for sampled_paths, else every edge inside S."""
    lat = pack.lat
    if method.weights == "sampled_paths":
        on_paths = {e for p in pack.paths or () for e in path_edges(p)}
        return [e for e in lat.edges if e in on_paths]
    return list(lat.edges)


def outside_edges(pack: LatticePack) -> list[EdgeKey]:
    """The ready actions that leave S, scored only for the B+ diagnostic (SPEC §6.6)."""
    return [(s, v) for s in pack.lat.states for v in pack.lat.outside[s]]


def _param_state(params: Sequence[torch.Tensor]) -> tuple[list[tuple[int, int]], list[float]]:
    """Storage and in-place version counter of each trainable parameter, and a float64 checksum of its values: the
    versions catch in-place updates, the checksums also catch fused optimizers and writes through `.data`."""
    pointers = [(p.data_ptr(), p._version) for p in params]
    sums = torch.stack([p.detach().double().sum() for p in params]).tolist() if params else []
    return pointers, sums


def _no_step_between_passes(optimizer: torch.optim.Optimizer, args: Any, kwargs: Any) -> None:
    raise RuntimeError("optimizer step between pass 1 and pass 2 (SPEC §6.5)")


def train_step(
    scorer: Scorer,
    packs: Sequence[LatticePack],
    method: Method,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    plus: bool = False,
    max_grad_norm: float | None = None,
) -> StepResult:
    """One step of Algorithm 1 on a batch of packs. Gradients start from zero; without an optimizer they stay in
    `.grad`. With plus=True, QET steps also score the actions that leave S and log the B+ bound (not budgeted)."""
    if not packs:
        raise ValueError("a step needs at least one graph")
    n_graphs = len(packs)
    params = scorer.parameters()
    for p in params:
        p.grad = None  # no gradient left over from an earlier call
    guard = optimizer.register_step_pre_hook(_no_step_between_passes) if optimizer is not None else None
    try:
        before = _param_state(params)
        if not all(math.isfinite(x) for x in before[1]):  # NaN would also defeat the change check below
            raise RuntimeError("non-finite trainable parameters before pass 1")
        t0 = time.perf_counter()

        # Pass 1: score without gradients (eval mode is asserted by the scorer).
        logp1: list[dict[EdgeKey, float] | None] = [None] * n_graphs
        scored, diagnostic = TokenCounts(), TokenCounts()
        items1: list[Item] = []
        if method.pass1:
            items1 = [(g, e) for g, pk in enumerate(packs) for e in pass1_edges(pk, method)]
            values, scored = scorer.logp(packs, items1, grad=False)
            logp1 = [{} for _ in packs]
            for (g, e), x in zip(items1, values.tolist(), strict=True):
                logp1[g][e] = x
        t1 = time.perf_counter()
        if plus and method.weights == "qet":  # the B+ diagnostic, timed and counted apart from pass 1
            extra = [(g, e) for g, pk in enumerate(packs) for e in outside_edges(pk)]
            if extra:
                values, diagnostic = scorer.logp(packs, extra, grad=False)
                for (g, e), x in zip(extra, values.tolist(), strict=True):
                    logp1[g][e] = x
        t_diag = time.perf_counter()

        # Weights (the production code of every configuration), then the guard: no update since pass 1.
        ews: list[EdgeWeights] = [
            edge_weights(method.weights, pk.lat, logp1[g], lam=method.lam, order=pk.order, paths=pk.paths)
            for g, pk in enumerate(packs)
        ]
        if _param_state(params) != before:
            raise RuntimeError("parameters changed between pass 1 and pass 2 (SPEC §6.5)")
        t2 = time.perf_counter()

        # Pass 2: weighted SFT on the edges with non-zero weight, weights detached, averaged over graphs (D-014).
        items2 = [
            (g, e) for g, (pk, ew) in enumerate(zip(packs, ews, strict=True)) for e in pk.lat.edges if ew.weights[e]
        ]
        w2 = torch.tensor([ews[g].weights[e] for g, e in items2], dtype=torch.float64)
        logp2: list[dict[EdgeKey, float]] = [{} for _ in packs]
        gradient = TokenCounts()
        loss_value = 0.0
        for mb in scorer.micro_batches(packs, items2):
            batch = [items2[i] for i in mb]
            lp, counts = scorer.logp(packs, batch, grad=True)
            loss = -(w2[mb].to(lp.device) * lp).sum() / n_graphs
            loss.backward()
            loss_value += loss.item()
            gradient += counts
            for (g, e), x in zip(batch, lp.detach().tolist(), strict=True):
                logp2[g][e] = x
        if _param_state(params)[1] != before[1]:
            raise RuntimeError("parameters changed during pass 2 (SPEC §6.5)")
        t3 = time.perf_counter()
        gradient.seconds = t3 - t2
    finally:
        if guard is not None:
            guard.remove()

    # The configuration's logged objective: from pass 1 when the weights used it, else from pass 2 (SPEC §8).
    objectives = [
        logged_objective(ew, pk.lat, logp1[g] if method.pass1 else logp2[g], paths=pk.paths)
        for g, (pk, ew) in enumerate(zip(packs, ews, strict=True))
    ]
    gap = None
    if plus and method.weights == "qet":
        gap = []
        for g, pk in enumerate(packs):
            mu = start_weights(pk.lat, method.lam)
            lb, lbp = log_completion(pk.lat, logp1[g]), log_completion(pk.lat, logp1[g], plus=True)
            gap.append(math.fsum(mu[s] * (lbp[s] - lb[s]) for s in pk.lat.states if mu[s] > 0.0))

    grad_norm = None
    if optimizer is not None:
        clip = max_grad_norm if max_grad_norm is not None else math.inf
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, clip))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    t4 = time.perf_counter()
    return StepResult(
        objective=math.fsum(objectives) / n_graphs,
        objectives=objectives,
        scored=scored,
        gradient=gradient,
        diagnostic=diagnostic,
        edges_scored=len(items1),
        edges_trained=len(items2),
        seconds={
            "pass1": t1 - t0,
            "diagnostic": t_diag - t1,
            "weights": t2 - t_diag,
            "pass2": t3 - t2,
            "update": t4 - t3,
        },
        grad_norm=grad_norm,
        bplus_gap=gap,
        pass2_loss=loss_value,
    )


# ----------------------------------------------------------------------------- the training loop
def lr_at(progress: float, peak: float, warmup: float) -> float:
    """Linear warm-up over the first `warmup` of the run's own tokens, then cosine decay to 0 at the budget (D-009,
    D-023e). The schedule follows tokens, not steps, because configurations without pass 1 take more steps; it spans
    the run's own tokens because a run that starts from a warm-up checkpoint starts with a fresh optimizer."""
    if warmup > 0.0 and progress < warmup:
        return peak * progress / warmup
    rest = min(1.0, (progress - warmup) / (1.0 - warmup)) if warmup < 1.0 else 1.0
    return peak * 0.5 * (1.0 + math.cos(math.pi * rest))


@dataclass(frozen=True)
class TrainConfig:
    token_budget: int  # scored + gradient tokens of the configuration, warm-up included (EVAL_PLAN §8)
    batch_graphs: int = 8
    lr: float = 1e-4
    warmup: float = 0.03  # fraction of the run's own tokens
    max_grad_norm: float | None = 1.0
    seed: int = 0  # the epoch order and, through the pack builder, the sub-lattices
    plus: bool = False  # the B+ diagnostic of QET runs (scored outside the budget)
    checkpoint_fractions: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
    tokens_at_start: int = 0  # tokens already spent, e.g. by the format warm-up of the parent checkpoint (D-010)
    max_steps: int | None = None  # a hard stop (tests, smoke runs)
    max_minutes: float | None = None  # a wall-clock stop (throughput pilots that must fit a job's walltime)


def train(
    scorer: Scorer,
    builder: PackBuilder,
    records: Sequence[GraphRecord],
    cfg: TrainConfig,
    optimizer: torch.optim.Optimizer,
    *,
    on_step: Callable[[dict[str, Any]], None] | None = None,
    on_checkpoint: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Epochs over the records in a seeded order until the token budget is spent. `on_step` receives one log row per
    step; `on_checkpoint` is called at each budget fraction ("frac-0.25", ...) and, after an early stop, "final".
    The epoch-0 packs are built first, so a graph that cannot be trained on fails before any compute is spent."""
    ids = [r.task_id for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError("task ids must be unique: they key the prompt cache and the sub-lattice seed")
    if not records or cfg.batch_graphs < 1 or not 0 <= cfg.tokens_at_start < cfg.token_budget:
        raise ValueError("training needs records, a positive batch size and budget left to spend")
    if builder.run_seed != cfg.seed:
        raise ValueError(f"the pack builder's run seed {builder.run_seed} is not the run's seed {cfg.seed}")
    for r in records:
        builder.build(r, 0)
    rng = random.Random(cfg.seed)
    used, step, epoch = cfg.tokens_at_start, 0, 0
    span = cfg.token_budget - cfg.tokens_at_start
    pending = sorted(f for f in cfg.checkpoint_fractions if f * cfg.token_budget > used)
    totals = {"scored": TokenCounts(), "gradient": TokenCounts(), "diagnostic": TokenCounts()}
    row: dict[str, Any] = {}

    started = time.perf_counter()

    def done() -> bool:
        late = cfg.max_minutes is not None and time.perf_counter() - started >= 60 * cfg.max_minutes
        return used >= cfg.token_budget or (cfg.max_steps is not None and step >= cfg.max_steps) or late

    while not done():
        order = list(range(len(records)))
        rng.shuffle(order)
        for start in range(0, len(order), cfg.batch_graphs):
            packs = [builder.build(records[j], epoch) for j in order[start : start + cfg.batch_graphs]]
            lr = lr_at((used - cfg.tokens_at_start) / span, cfg.lr, cfg.warmup)
            for group in optimizer.param_groups:
                group["lr"] = lr
            res = train_step(scorer, packs, builder.method, optimizer, plus=cfg.plus, max_grad_norm=cfg.max_grad_norm)
            used += res.scored.tokens + res.gradient.tokens  # the diagnostic is not part of the budget
            step += 1
            for key in totals:
                totals[key] += getattr(res, key)
            row = {
                "step": step,
                "epoch": epoch,
                "graphs": [pk.task_id for pk in packs],
                "lr": lr,
                "objective": res.objective,
                "objectives": res.objectives,
                "scored": res.scored.to_json(),
                "gradient": res.gradient.to_json(),
                "diagnostic": res.diagnostic.to_json(),
                "tokens_used": used,
                "budget_fraction": used / cfg.token_budget,
                "edges_scored": res.edges_scored,
                "edges_trained": res.edges_trained,
                "states": [len(pk.lat.states) for pk in packs],
                "exact": [pk.lat.exact for pk in packs],
                "grad_norm": res.grad_norm,
                "bplus_gap": res.bplus_gap,
                "seconds": res.seconds,
            }
            if on_step:
                on_step(row)
            while pending and used >= pending[0] * cfg.token_budget:
                frac = pending.pop(0)
                if on_checkpoint:
                    on_checkpoint(f"frac-{frac:g}", row)
            if done():
                break
        epoch += 1
    if pending and step and on_checkpoint:
        on_checkpoint("final", row)
    return {
        "steps": step,
        "epochs_started": epoch,
        "tokens_used": used,
        **{key: counts.to_json() for key, counts in totals.items()},
        "last": row,
    }

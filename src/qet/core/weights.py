"""Edge weights and logged objectives of the training configurations (SPEC §6.4, §8).

Every configuration is weighted SFT on edges of the same lattice; they differ only in the edge weights w(I, v)
and their start states. The flow-based rows set w(I, v) = m(I) * s(v | I), where the step probability s sums to 1
over the successors of I inside S and m(I) = mu(I) + incoming weight is the occupancy. The finish action has
weight 1 in every configuration.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from qet.core.lattice import (
    END,
    EdgeKey,
    Lattice,
    LogP,
    edge_logp,
    is_topological_order,
    log_completion,
    logsumexp,
    members,
    prefix_states,
    start_weights,
)

METHODS = ("path_sft", "uniform_aug", "frontier", "qet", "sampled_paths")
FLOW_METHODS = ("uniform_aug", "frontier", "qet")


@dataclass(frozen=True)
class EdgeWeights:
    method: str
    weights: dict[EdgeKey, float]  # every edge of the lattice, finish edge (V, END) included; 0 = not trained
    occupancy: dict[int, float] | None = None  # m(I), flow-based methods only
    lam: float | None = None  # lambda of the start distribution mu the weights were computed with (flow-based)


def flow(
    lat: Lattice, mu: dict[int, float], step_prob: Callable[[int, int], float]
) -> tuple[dict[EdgeKey, float], dict[int, float]]:
    """w(I, v) = m(I) * step_prob(I, v) from the small states up; w(V, end) = m(V). States without mass send none,
    so step_prob is never evaluated where it may be undefined."""
    m = {s: mu.get(s, 0.0) for s in lat.states}
    w: dict[EdgeKey, float] = {}
    for s in lat.states:
        if s == lat.full:
            continue
        for v in lat.succ[s]:
            x = m[s] * step_prob(s, v) if m[s] != 0.0 else 0.0
            w[(s, v)] = x
            m[s | 1 << v] += x
    w[(lat.full, END)] = m[lat.full]
    return w, m


def completion_counts(lat: Lattice) -> dict[int, int]:
    """c(V) = 1 and c(I) = sum of c(I + v) over successors inside S: the legal completions from I (exact ints)."""
    c: dict[int, int] = {}
    for s in reversed(lat.states):
        c[s] = 1 if s == lat.full else sum(c[s | 1 << v] for v in lat.succ[s])
    return c


def _check_order(lat: Lattice, order: Sequence[int]) -> list[int]:
    order = list(order)
    if not is_topological_order(order, lat.preds):
        raise ValueError(f"{order} is not a topological order of the graph")
    outside = [members(s) for s in prefix_states(order) if s not in lat.succ]
    if outside:
        raise ValueError(f"the order {order} leaves the states in use at {outside[0]}")
    return order


def _path_edges(order: Sequence[int]) -> list[EdgeKey]:
    states = prefix_states(order)
    return [(s, v) for s, v in zip(states, order, strict=False)] + [(states[-1], END)]


def _path_logp(order: Sequence[int], logp: LogP) -> float:
    return math.fsum(edge_logp(logp, e) for e in _path_edges(order))


def unique_paths(lat: Lattice, paths: Sequence[Sequence[int]]) -> list[list[int]]:
    """Validated paths with repeats removed, first occurrence kept (D-011)."""
    out: list[list[int]] = []
    for p in paths:
        p = _check_order(lat, p)
        if p not in out:
            out.append(p)
    if not out:
        raise ValueError("sampled_paths needs at least one path")
    return out


def edge_weights(
    method: str,
    lat: Lattice,
    logp: LogP | None = None,
    *,
    lam: float = 0.5,
    order: Sequence[int] | None = None,
    paths: Sequence[Sequence[int]] | None = None,
) -> EdgeWeights:
    """Edge weights of one configuration (SPEC §8).

    path_sft (configurations 1, 2; ablation a2 with the canonical order): 1 on the edges of `order`.
    uniform_aug (configuration 3, D-012): m(I) * c(I + v) / c(I), starts mu.
    frontier (configuration 4, D-008): m(I) * pi(a_v) / sum of pi(a_u) over u ready in S, starts mu.
    qet (configurations 5, 6): rho(I, v) = m(I) * q(v | I), q = pi(a_v | H_I) B(I + v) / B(I), starts mu.
    sampled_paths (ablation a3, D-011): the posterior of each of the given paths, root start only.
    """
    zeros = dict.fromkeys(lat.edges, 0.0)
    if method == "path_sft":
        if order is None:
            raise ValueError("path_sft needs an order")
        return EdgeWeights(method, {**zeros, **dict.fromkeys(_path_edges(_check_order(lat, order)), 1.0)})
    if method == "sampled_paths":
        if logp is None or paths is None:
            raise ValueError("sampled_paths needs logp and paths")
        uniq = unique_paths(lat, paths)
        lps = [_path_logp(p, logp) for p in uniq]
        z = logsumexp(lps)
        if z == -math.inf:
            raise ValueError("every sampled path has probability 0")
        w = dict(zeros)
        for p, lp in zip(uniq, lps, strict=True):
            post = math.exp(lp - z)
            for e in _path_edges(p):
                w[e] += post
        return EdgeWeights(method, w)
    if method not in FLOW_METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    mu = start_weights(lat, lam)
    if method == "uniform_aug":
        c = completion_counts(lat)

        def step(s: int, v: int) -> float:
            return c[s | 1 << v] / c[s]

    elif logp is None:
        raise ValueError(f"{method} needs logp (pass 1)")
    elif method == "qet":
        log_b = log_completion(lat, logp)

        def step(s: int, v: int) -> float:
            if log_b[s] == -math.inf:
                raise ValueError(f"B({members(s)}) = 0: the objective is infinite")
            return math.exp(edge_logp(logp, (s, v)) + log_b[s | 1 << v] - log_b[s])

    else:  # frontier: the model renormalized over the ready actions inside S

        def step(s: int, v: int) -> float:
            z = logsumexp(edge_logp(logp, (s, u)) for u in lat.succ[s])
            if z == -math.inf:
                raise ValueError(f"every ready action of {members(s)} has probability 0")
            return math.exp(edge_logp(logp, (s, v)) - z)

    w, m = flow(lat, mu, step)
    return EdgeWeights(method, {**zeros, **w}, m, lam)


# ----------------------------------------------------------------------------- objectives
def qet_objective(lat: Lattice, log_b: dict[int, float], mu: dict[int, float]) -> float:
    """sum over states of mu(I) * (-log B(I)) (eq. 2), over the states with mu > 0."""
    return math.fsum(mu[s] * -log_b[s] for s in lat.states if mu[s] > 0.0)


def surrogate(weights: dict[EdgeKey, float], logp: LogP) -> float:
    """The weighted SFT loss sum of w * (-log pi), finish included, over edges with non-zero weight."""
    return math.fsum(w * -edge_logp(logp, e) for e, w in weights.items() if w != 0.0)


def logged_objective(
    ew: EdgeWeights,
    lat: Lattice,
    logp: LogP,
    *,
    log_b: dict[int, float] | None = None,
    mu: dict[int, float] | None = None,
    paths: Sequence[Sequence[int]] | None = None,
) -> float:
    """The objective each configuration logs (SPEC §8): never the pass-2 loss of the flow-based methods.

    Only qet needs log B and mu; mu defaults to the start distribution of the lambda the weights were computed with,
    so configuration 5 (lambda = 0) logs -log B(empty). path_sft and uniform_aug read log pi on their weighted edges
    only, so configurations 1-3 can log from pass-2 log-probabilities alone.
    """
    if ew.method == "qet":
        if mu is None and ew.lam is None:
            raise ValueError("qet weights must record their lambda (use edge_weights) or mu must be given")
        log_b = log_completion(lat, logp) if log_b is None else log_b
        return qet_objective(lat, log_b, start_weights(lat, ew.lam) if mu is None else mu)
    if ew.method == "frontier":
        m = ew.occupancy or {}
        local = [
            m[s] * -logsumexp(edge_logp(logp, (s, u)) for u in lat.succ[s])
            for s in lat.states
            if s != lat.full and m[s] != 0.0
        ]
        return math.fsum(local) + m[lat.full] * -edge_logp(logp, (lat.full, END))
    if ew.method == "sampled_paths":
        return -logsumexp(_path_logp(p, logp) for p in unique_paths(lat, paths or ()))
    return surrogate(ew.weights, logp)  # path_sft, uniform_aug: the weighted negative log-likelihood


@dataclass(frozen=True)
class Summary:
    log_b: dict[int, float]
    mu: dict[int, float]
    weights: dict[EdgeKey, float]
    qet_objective: float
    surrogate: float
    objective: float
    log_b_plus: dict[int, float] | None


def summarize(
    method: str,
    lat: Lattice,
    logp: LogP,
    *,
    lam: float = 0.5,
    order: Sequence[int] | None = None,
    paths: Sequence[Sequence[int]] | None = None,
    plus: bool = False,
) -> Summary:
    """Every lattice quantity of one configuration on one graph (SPEC §10): log B over S, mu with this lambda,
    the edge weights, Σ mu (-log B), the surrogate Σ w (-log pi), the logged objective and, with plus, log B+."""
    ew = edge_weights(method, lat, logp, lam=lam, order=order, paths=paths)
    log_b = log_completion(lat, logp)
    mu = start_weights(lat, lam)
    return Summary(
        log_b=log_b,
        mu=mu,
        weights=ew.weights,
        qet_objective=qet_objective(lat, log_b, mu),
        surrogate=surrogate(ew.weights, logp),
        objective=logged_objective(ew, lat, logp, log_b=log_b, mu=mu, paths=paths),
        log_b_plus=log_completion(lat, logp, plus=True) if plus else None,
    )

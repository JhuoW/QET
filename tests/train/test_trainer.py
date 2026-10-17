"""E1.3: the two-pass trainer (SPEC §6.5, §8) on a tiny random-weight model in float64 on CPU."""

import hashlib
import json
import math
from pathlib import Path

import pytest
import torch
from tiny_models import tiny_lm

from qet.core.compile import compile_demo
from qet.core.lattice import END, full_lattice, log_completion, members, start_weights, state_of
from qet.core.weights import qet_objective
from qet.train.chat import ChatFormat
from qet.train.methods import METHODS, method
from qet.train.scoring import TokenCounts, forward_logp, score
from qet.train.trainer import (
    LatticePack,
    LMScorer,
    PackBuilder,
    TrainConfig,
    add_lora,
    lr_at,
    path_edges,
    train,
    train_step,
)

FIXTURES = Path(__file__).resolve().parents[2] / "reference" / "fixtures"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
VOCAB = 151936  # Qwen2.5's embedding rows (>= every token id of its tokenizer)

TOOLS = [
    {"name": "book", "params": ["to", "price"], "description": "Book a flight."},
    {"name": "get_orders", "params": ["user"], "description": "Orders of a user."},
    {"name": "get_price", "params": ["from", "to"], "description": "Price of a flight."},
    {"name": "get_user", "params": ["name"], "description": "Look up a user."},
    {"name": "get_weather", "params": ["city"], "description": "Weather in a city."},
]
CONTRACTS = {
    "book": {"reads": [], "writes": ["booking"]},
    "get_orders": {"reads": [], "writes": []},
    "get_price": {"reads": [], "writes": []},
    "get_user": {"reads": [], "writes": []},
    "get_weather": {"reads": [], "writes": []},
}
CITIES = ["SG", "LA", "SF", "NY", "DC", "TO", "HK", "BJ", "RM"]


def step(tool, args, obs):
    return {"tool": tool, "args": args, "obs": obs}


def finish(answer):
    return {"tool": "finish", "args": {"answer": answer}}


def records():
    """Fan-in (three independent prices, then a booking: 9 states), a chain beside an independent call (6 states),
    a task with no calls, nine independent calls (512 states > S_max: the sub-lattice) and a single call."""
    fan_in = [
        step("get_price", {"from": "NYC", "to": "SG"}, {"price": 1200}),
        step("get_price", {"from": "NYC", "to": "LA"}, {"price": 300}),
        step("get_price", {"from": "NYC", "to": "SF"}, {"price": 350}),
        step("book", {"to": "LA", "price": 300}, {"booking": "B-1"}),
        finish("B-1"),
    ]
    chain = [
        step("get_user", {"name": "ann"}, {"id": 17}),
        step("get_orders", {"user": 17}, {"orders": [5, 9]}),
        step("get_weather", {"city": "Paris"}, {"temp": 21}),
        finish(21),
    ]
    wide = [step("get_price", {"from": "NYC", "to": c}, {"price": 100 + i}) for i, c in enumerate(CITIES)]
    cases = [
        ("fan-in", "Compare flights from NYC to SG, LA and SF and book the cheapest.", fan_in),
        ("chain", "Find the orders of ann and the weather in Paris.", chain),
        ("empty", "Say hello.", [finish("hello")]),
        ("wide", "Get prices from NYC to " + ", ".join(CITIES) + ".", wide + [finish(108)]),
        ("single", "Weather in Paris.", [step("get_weather", {"city": "Paris"}, {"temp": 21}), finish(21)]),
    ]
    return [compile_demo({"request": req, "tools": TOOLS}, demo, CONTRACTS, task_id=tid) for tid, req, demo in cases]


@pytest.fixture(scope="module")
def fmt():
    return ChatFormat.load(MODEL_ID)


@pytest.fixture(scope="module")
def recs():
    return records()


def test_the_test_graphs_have_the_intended_shapes(recs):
    sizes = [None if (lat := full_lattice(r.n, r.preds(), cap=256)) is None else len(lat.states) for r in recs]
    assert sizes == [9, 6, 1, None, 2]


def lm(seed=0):
    return tiny_lm("qwen2", vocab_size=VOCAB, seed=seed)


def builder(fmt, config_id, k=3, run_seed=0):
    return PackBuilder(fmt, CONTRACTS, method(config_id), k=k, run_seed=run_seed)


# ----------------------------------------------------------------------------- (a) gradients
def torch_log_b(lat, lp):
    """log B_S by the recursion of SPEC §6.2 in torch, so autograd differentiates through it."""
    lb = {}
    for s in reversed(lat.states):
        if s == lat.full:
            lb[s] = lp[(s, END)]
        else:
            lb[s] = torch.logsumexp(torch.stack([lp[(s, v)] + lb[s | 1 << v] for v in lat.succ[s]]), 0)
    return lb


def autograd_objective(config_id, packs, lps, ews):
    """The single-pass form (SPEC §6.5): configurations 5 and 6 differentiate sum mu (-log B) through the lattice;
    configurations 3 and 4 differentiate their logged objective with the state weights held constant."""
    m = method(config_id)
    total = 0
    for pk, lp, ew in zip(packs, lps, ews, strict=True):
        lat = pk.lat
        if m.weights == "qet":
            mu, lb = start_weights(lat, m.lam), torch_log_b(lat, lp)
            total = total + sum(mu[s] * -lb[s] for s in lat.states if mu[s] > 0)
        elif m.weights == "frontier":
            occ = ew.occupancy
            for s in lat.states:
                if s != lat.full and occ[s] != 0:
                    total = total - occ[s] * torch.logsumexp(torch.stack([lp[(s, u)] for u in lat.succ[s]]), 0)
            total = total - occ[lat.full] * lp[(lat.full, END)]
        else:  # uniform_aug: sum w (-log pi), weights independent of the model
            total = total + sum(w * -lp[e] for e, w in ew.weights.items() if w)
    return total / len(packs)


@pytest.mark.parametrize("config_id", ["3", "4", "5", "6", "a4"])
def test_two_pass_gradient_equals_the_autograd_gradient(fmt, recs, config_id):
    model = lm()
    scorer = LMScorer(model, pad_id=fmt.eot_id, max_tokens=4000)
    b = builder(fmt, config_id)
    packs = [b.build(r) for r in recs]
    assert not packs[3].lat.exact  # the wide graph trains on the sub-lattice
    params = scorer.parameters()
    model.zero_grad(set_to_none=True)
    train_step(scorer, packs, method(config_id))  # no optimizer: the gradient stays in .grad
    two_pass = [p.grad.clone() for p in params]

    from qet.core.weights import edge_weights

    items = [(g, e) for g, pk in enumerate(packs) for e in pk.lat.edges]
    with torch.enable_grad():
        values = forward_logp(model, [packs[g].pair(e) for g, e in items], pad_id=fmt.eot_id)
    lps = [{} for _ in packs]
    for (g, e), x in zip(items, values, strict=True):
        lps[g][e] = x
    m = method(config_id)
    ews = [
        edge_weights(m.weights, pk.lat, {e: float(x.detach()) for e, x in lp.items()}, lam=m.lam)
        for pk, lp in zip(packs, lps, strict=True)
    ]
    auto = torch.autograd.grad(autograd_objective(config_id, packs, lps, ews), params)
    worst = 0.0
    for a, t in zip(two_pass, auto, strict=True):  # relative to each tensor's largest entry; all-zero tensors absolute
        scale = t.abs().max()
        worst = max(worst, ((a - t).abs().max() / scale).item() if scale > 0 else a.abs().max().item())
    assert worst < 1e-8, worst


# ----------------------------------------------------------------------------- (b) toy training fixture
class TableScorer:
    """One softmax per state over its targets plus a 'wrong' bucket (the fixture's table policy)."""

    def __init__(self, lat, init_pi):
        self.targets = {s: [END] if s == lat.full else list(lat.succ[s]) for s in lat.states}
        self.z = {}
        for s in lat.states:
            ps = [init_pi[(s, v)] for v in self.targets[s]]
            logits = [math.log(p) for p in ps] + [math.log(1.0 - sum(ps))]
            self.z[s] = torch.nn.Parameter(torch.tensor(logits, dtype=torch.float64))

    def parameters(self):
        return list(self.z.values())

    def logp(self, packs, items, *, grad):
        with torch.set_grad_enabled(grad):
            out = [torch.log_softmax(self.z[s], 0)[self.targets[s].index(v)] for _, (s, v) in items]
            return torch.stack(out), TokenCounts(sequences=len(items))

    def micro_batches(self, packs, items):
        return [list(range(len(items)))]

    def all_logp(self):
        with torch.no_grad():
            return {(s, v): torch.log_softmax(z, 0)[i].item() for s, z in self.z.items() for i, v in
                    enumerate(self.targets[s])}  # fmt: skip


def parse_edge(key, n):
    st, v = key.split("|")
    return state_of([int(x) for x in st.split(",")] if st else [], n), END if v == "end" else int(v)


def test_table_policy_reproduces_the_toy_training_fixture():
    fx = json.loads((FIXTURES / "toy_training.json").read_text())
    n, preds = len(fx["preds"]), fx["preds"]
    lat = full_lattice(n, preds)
    for run in fx["runs"]:
        init = {parse_edge(k, n): p for k, p in run["init_pi"].items()}
        table = TableScorer(lat, init)
        cfg = {"qet": "6", "path_sft": "2"}[run["method"]]
        assert method(cfg).lam == run["lam"] or run["method"] == "path_sft"
        pack = LatticePack("toy", lat, order=run.get("order"))
        opt = torch.optim.SGD(table.parameters(), lr=run["lr"])
        recorded = {h["step"]: h for h in run["history"]}
        for t in range(run["steps"] + 1):
            if t in recorded:
                lp = table.all_logp()
                lb = log_completion(lat, lp)
                h = recorded[t]
                for key, p in h["pi"].items():
                    assert abs(math.exp(lp[parse_edge(key, n)]) - p) <= 1e-6, (run["method"], t, key)
                for key, b in h["B"].items():
                    assert abs(math.exp(lb[state_of([int(x) for x in key.split(",")] if key else [], n)]) - b) <= 1e-6
                obj = qet_objective(lat, lb, start_weights(lat, run["lam"]))
                assert abs(obj - h["qet_objective"]) <= 1e-6
            if t < run["steps"]:
                train_step(table, [pack], method(cfg), opt)
    # SPEC §6.5's toy check: QET raises pi(A | H_{B}) from 0.10 to 0.956; demo SFT leaves it at 0.100
    last = {(r["init"], r["method"]): r["history"][-1]["pi"]["1|0"] for r in fx["runs"]}
    assert (
        round(last[("demo_path_model", "qet")], 3) == 0.956 and round(last[("demo_path_model", "path_sft")], 3) == 0.1
    )


# ----------------------------------------------------------------------------- (c) the logged objective
@pytest.mark.parametrize("config_id", ["5", "6"])
def test_the_logged_objective_is_sum_mu_minus_log_b_not_the_pass2_loss(fmt, recs, config_id):
    model = lm(1)
    scorer = LMScorer(model, pad_id=fmt.eot_id)
    packs = [builder(fmt, config_id).build(r) for r in recs[:2]]
    res = train_step(scorer, packs, method(config_id))
    expected = []
    for pk in packs:
        edges = list(pk.lat.edges)
        values, _ = score(model, [pk.pair(e) for e in edges], pad_id=fmt.eot_id)
        lp = dict(zip(edges, values, strict=True))
        expected.append(qet_objective(pk.lat, log_completion(pk.lat, lp), start_weights(pk.lat, method(config_id).lam)))
    assert res.objectives == pytest.approx(expected, rel=1e-12)
    assert res.objective == pytest.approx(sum(expected) / 2, rel=1e-12)
    if config_id == "5":  # lambda = 0: -log B(empty)
        assert res.objectives[0] == pytest.approx(-log_completion(packs[0].lat, lp_of(model, fmt, packs[0]))[0])
    assert res.pass2_loss > res.objective + 1e-6  # the pass-2 loss adds the posterior entropy (SPEC §6.5 item 5)


def lp_of(model, fmt, pk):
    edges = list(pk.lat.edges)
    values, _ = score(model, [pk.pair(e) for e in edges], pad_id=fmt.eot_id)
    return dict(zip(edges, values, strict=True))


# ----------------------------------------------------------------------------- (d) no update between the passes
def param_hash(params):
    h = hashlib.sha256()
    for p in params:
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


class Recording(LMScorer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = []

    def logp(self, packs, items, *, grad):
        self.calls.append((grad, param_hash(self.parameters())))
        return super().logp(packs, items, grad=grad)


@pytest.mark.parametrize("fused", [False, True])
def test_no_parameter_update_between_the_passes(fmt, recs, fused):
    model = lm(2)
    scorer = Recording(model, pad_id=fmt.eot_id, max_tokens=1500)  # several pass-2 micro-batches
    packs = [builder(fmt, "6").build(r) for r in recs[:2]]
    opt = torch.optim.AdamW(scorer.parameters(), lr=1e-2, fused=fused, foreach=None if fused else True)
    for _ in range(3):  # from the second step on, an optimizer step between the passes would have state to apply
        scorer.calls.clear()
        start = param_hash(scorer.parameters())
        train_step(scorer, packs, method("6"), opt)
        grads = [g for g, _ in scorer.calls]
        assert grads[0] is False and grads.count(False) == 1 and grads.count(True) >= 2
        assert {h for _, h in scorer.calls} == {start}  # pass 1 and every pass-2 micro-batch saw the same parameters
        assert param_hash(scorer.parameters()) != start  # the one update comes after pass 2


class Meddling(LMScorer):
    """Changes a parameter right after pass 1: in place, through .data (no version bump), or by an optimizer step."""

    how = "inplace"
    optimizer = None

    def logp(self, packs, items, *, grad):
        out = super().logp(packs, items, grad=grad)
        if not grad:
            p = self.parameters()[0]
            if self.how == "inplace":
                with torch.no_grad():
                    p.add_(1e-3)
            elif self.how == "data":
                p.data.add_(1e-3)
            else:
                self.optimizer.step()
        return out


@pytest.mark.parametrize("how", ["inplace", "data", "optimizer"])
def test_an_update_between_the_passes_is_refused(fmt, recs, how):
    scorer = Meddling(lm(3), pad_id=fmt.eot_id)
    scorer.how = how
    scorer.optimizer = opt = torch.optim.AdamW(scorer.parameters(), lr=1e-2, fused=True)
    packs = [builder(fmt, "6").build(recs[0])]
    with pytest.raises(RuntimeError, match="between pass 1 and pass 2"):
        train_step(scorer, packs, method("6"), opt)


def test_gradients_start_from_zero_each_step(fmt, recs):
    scorer = LMScorer(lm(8), pad_id=fmt.eot_id)
    packs = [builder(fmt, "2").build(recs[0])]
    train_step(scorer, packs, method("2"))
    first = [p.grad.clone() for p in scorer.parameters()]
    train_step(scorer, packs, method("2"))
    assert all(torch.equal(a, p.grad) for a, p in zip(first, scorer.parameters(), strict=True))


def test_the_bplus_diagnostic_is_scored_for_qet_only_and_outside_the_budget(fmt, recs):
    model = lm(7)
    wide6, wide4 = builder(fmt, "6").build(recs[3]), builder(fmt, "4").build(recs[3])
    plain = train_step(LMScorer(model, pad_id=fmt.eot_id), [wide6], method("6"))
    diag = train_step(LMScorer(model, pad_id=fmt.eot_id), [wide6], method("6"), plus=True)
    assert diag.scored.tokens == plain.scored.tokens and plain.diagnostic.tokens == 0 < diag.diagnostic.tokens
    assert diag.objectives == pytest.approx(plain.objectives, rel=1e-12)
    assert plain.bplus_gap is None and diag.bplus_gap[0] >= 0.0
    four = train_step(LMScorer(model, pad_id=fmt.eot_id), [wide4], method("4"), plus=True)
    assert four.diagnostic.tokens == 0 and four.bplus_gap is None


def test_train_requires_the_builder_to_use_the_run_seed(fmt, recs):
    scorer = LMScorer(lm(9), pad_id=fmt.eot_id)
    opt = torch.optim.SGD(scorer.parameters(), lr=0.1)
    with pytest.raises(ValueError, match="run seed"):
        train(scorer, builder(fmt, "6", run_seed=0), recs, TrainConfig(token_budget=1000, seed=1), opt)


@pytest.mark.slow
@pytest.mark.parametrize("config_id", sorted(METHODS))
def test_every_configuration_logs_the_oracles_objective(fmt, recs, config_id):
    """train_step against reference/qet_ref.summary for all ten configurations (SPEC §8's logged objective), epochs 0
    and 1, the sub-lattice, n = 0 and n = 1 included, and the B+ bound of the QET configurations."""
    import qet_ref as R

    model = lm(10)
    m = method(config_id)
    b = builder(fmt, config_id, k=3)
    for epoch in (0, 1):
        packs = [b.build(r, epoch) for r in recs]
        res = train_step(LMScorer(model, pad_id=fmt.eot_id), packs, m, plus=True)
        for g, (pk, r) in enumerate(zip(packs, recs, strict=True)):
            lat = pk.lat
            edges = list(lat.edges) + [(s, v) for s in lat.states for v in lat.outside[s]]
            values, _ = score(model, [pk.pair(e) for e in edges], pad_id=fmt.eot_id)
            olp = {
                (frozenset(members(s)), "end" if v == END else v): x for (s, v), x in zip(edges, values, strict=True)
            }
            states = [frozenset(members(s)) for s in lat.states]
            kw = {"path_sft": {"order": pk.order}, "sampled_paths": {"paths": pk.paths}}.get(m.weights, {})
            preds = [frozenset(p) for p in r.preds()]
            exp = R.summary(m.weights, states=states, preds=preds, n=r.n, logp=olp, lam=m.lam, plus=True, **kw)
            assert res.objectives[g] == pytest.approx(exp["objective"], rel=1e-9, abs=1e-12), (config_id, epoch, g)
            if res.bplus_gap is not None:
                mu = R.start_weights(states, r.n, m.lam)
                gap = sum(mu[st] * (exp["logB_plus"][st] - exp["logB"][st]) for st in states if mu[st] > 0)
                assert res.bplus_gap[g] == pytest.approx(gap, rel=1e-9, abs=1e-12)


# ----------------------------------------------------------------------------- configurations share everything else
def test_lattice_configurations_share_states_edges_prompts_and_targets(fmt, recs):
    for epoch in (0, 1):
        built = {c: [builder(fmt, c).build(r, epoch) for r in recs] for c in ("3", "4", "5", "6")}
        for c in ("4", "5", "6"):
            for a, b in zip(built["3"], built[c], strict=True):
                assert (a.lat.states, a.lat.edges, a.prompt_ids, a.target_ids) == (
                    b.lat.states, b.lat.edges, b.prompt_ids, b.target_ids
                )  # fmt: skip
    wide = [builder(fmt, "6").build(recs[3], e).lat.states for e in range(4)]
    assert len(set(wide)) > 1  # the sub-lattice is resampled every epoch (SPEC §6.6)


def test_configuration_1_differs_from_2_only_in_its_renderer(fmt, recs):
    for r in recs:
        one, two = builder(fmt, "1").build(r), builder(fmt, "2").build(r)
        assert (one.lat.states, one.lat.edges, one.order, one.target_ids) == (
            two.lat.states, two.lat.edges, two.order, two.target_ids
        )  # fmt: skip
        assert r.n < 2 or one.prompt_ids != two.prompt_ids
        assert one.order == list(range(r.n)) and one.lat.edges == tuple(path_edges(one.order))


def test_path_sft_trains_the_demo_path_without_pass_1(fmt, recs):
    model = lm(4)
    scorer = LMScorer(model, pad_id=fmt.eot_id)
    packs = [builder(fmt, "2").build(r) for r in recs]
    res = train_step(scorer, packs, method("2"))
    assert res.scored.sequences == 0 and res.edges_scored == 0
    assert res.edges_trained == sum(r.n + 1 for r in recs)
    nll = []
    for pk in packs:
        values, _ = score(model, [pk.pair(e) for e in pk.lat.edges], pad_id=fmt.eot_id)
        nll.append(-sum(values))
    assert res.objectives == pytest.approx(nll, rel=1e-12)
    assert res.pass2_loss == pytest.approx(sum(nll) / len(nll), rel=1e-12)  # here they coincide (SPEC §8)


def test_ablations_build_their_states(fmt, recs):
    a2 = builder(fmt, "a2").build(recs[0])
    assert a2.order != list(range(4)) and a2.order[-1] == 3  # the canonical order sorts the three prices by key
    a3 = builder(fmt, "a3", k=None).build(recs[3])  # without a K from the run config, D-011's default K = 4
    assert len(a3.paths) == 4 and a3.paths[0] == list(range(9))
    assert len(builder(fmt, "a3", k=8).build(recs[3]).paths) == 8  # a3 takes K from the run config (D-023b)
    res = train_step(LMScorer(lm(5), pad_id=fmt.eot_id), [a3], method("a3"))
    assert res.edges_scored == len({e for p in a3.paths for e in path_edges(p)})
    a4 = builder(fmt, "a4", k=2).build(recs[0])
    assert not a4.lat.exact  # a4 samples a sub-lattice even below S_max
    with pytest.raises(ValueError, match="K"):
        PackBuilder(fmt, CONTRACTS, method("a4"), run_seed=0)


def test_configurations_are_valid():
    assert sorted(METHODS) == ["1", "2", "3", "4", "5", "6", "a1", "a2", "a3", "a4"]
    assert [method(c).pass1 for c in ("1", "2", "3", "4", "5", "6")] == [False, False, False, True, True, True]
    with pytest.raises(ValueError):
        method("7")


# ----------------------------------------------------------------------------- the loop
def test_lr_schedule():
    assert lr_at(0.0, 1e-4, 0.03) == 0.0
    assert lr_at(0.015, 1e-4, 0.03) == pytest.approx(5e-5)
    assert lr_at(0.03, 1e-4, 0.03) == pytest.approx(1e-4)
    assert lr_at(0.515, 1e-4, 0.03) == pytest.approx(5e-5)
    assert lr_at(1.0, 1e-4, 0.03) == pytest.approx(0.0, abs=1e-20)


@pytest.mark.slow  # imports peft: 10-30 s on the shared login node
def test_train_spends_the_token_budget_and_checkpoints_at_fractions(fmt, recs):
    model = add_lora(lm(6), rank=4, alpha=8)
    dropouts = [m.lora_dropout["default"] for m in model.modules() if hasattr(m, "lora_dropout")]
    assert dropouts and all(isinstance(d, torch.nn.Identity) for d in dropouts)  # PEFT's form of dropout 0
    scorer = LMScorer(model, pad_id=fmt.eot_id)
    b = builder(fmt, "6")
    cfg = TrainConfig(token_budget=60_000, batch_graphs=2, lr=1e-3, seed=0)
    rows, ckpts = [], []
    summary = train(
        scorer, b, recs, cfg, torch.optim.AdamW(scorer.parameters(), lr=cfg.lr),
        on_step=rows.append, on_checkpoint=lambda label, row: ckpts.append((label, row["tokens_used"])),
    )  # fmt: skip
    assert summary["tokens_used"] >= cfg.token_budget and rows[-2]["tokens_used"] < cfg.token_budget
    assert [label for label, _ in ckpts] == ["frac-0.25", "frac-0.5", "frac-0.75", "frac-1"]
    for (_, used), frac in zip(ckpts, (0.25, 0.5, 0.75, 1.0), strict=True):
        assert used >= frac * cfg.token_budget
    assert all(set(r["graphs"]) <= {"fan-in", "chain", "empty", "wide", "single"} for r in rows)
    assert rows[0]["lr"] == 0.0 and max(r["lr"] for r in rows) <= cfg.lr
    assert sum(r["scored"]["tokens"] + r["gradient"]["tokens"] for r in rows) == summary["tokens_used"]
    assert all(math.isfinite(r["objective"]) and r["objective"] > 0 for r in rows)
    epochs = {r["epoch"] for r in rows}
    assert len(epochs) >= 2  # several epochs, each in a fresh seeded order
    stop = train(
        scorer, b, recs, TrainConfig(token_budget=10**9, batch_graphs=2, max_steps=2),
        torch.optim.AdamW(scorer.parameters()), on_checkpoint=lambda label, row: ckpts.append((label, 0)),
    )  # fmt: skip
    assert stop["steps"] == 2 and ckpts[-1][0] == "final"


def test_members_of_a_state_give_the_demo_order_history(fmt, recs):
    """Each state's prompt renders the demo steps of the state in demo order (a legal history)."""
    pk = builder(fmt, "6").build(recs[0])
    assert all(members(s) == sorted(members(s)) for s in pk.lat.states)

"""E1.2: the scorer (SPEC §6.2, §6.7): log pi = sum of target-token log-probabilities, end of turn included."""

import math
import random

import pytest
import torch
from tiny_models import MODEL_TYPES, tiny_lm

from qet.train.scoring import Pair, forward_logp, micro_batches, score

VOCAB = 101


def random_pairs(rng, n, vocab=VOCAB):
    return [
        Pair(
            tuple(rng.randrange(vocab) for _ in range(rng.randint(1, 40))),
            tuple(rng.randrange(vocab) for _ in range(rng.randint(1, 9))),
        )
        for _ in range(n)
    ]


def manual_logp(model, pair):
    """The textbook computation: full logits of the unpadded sequence, log-softmax, sum over the target tokens."""
    ids = torch.tensor([pair.prompt_ids + pair.target_ids])
    logits = model(ids).logits[0].to(torch.float64)
    lp = logits.log_softmax(-1)
    start = len(pair.prompt_ids) - 1
    return sum(lp[start + j, t] for j, t in enumerate(pair.target_ids))


@pytest.fixture(scope="module", params=MODEL_TYPES)
def model(request):
    return tiny_lm(request.param)


def test_log_pi_equals_a_manual_per_token_computation(model):
    pairs = random_pairs(random.Random(0), 40)
    got, _ = score(model, pairs, pad_id=0)
    with torch.no_grad():
        want = [manual_logp(model, p).item() for p in pairs]
    for g, w in zip(got, want, strict=True):
        assert w < 0 and abs(g - w) <= 1e-12 * abs(w)


def test_batch_and_padding_invariance_in_float64(model):
    rng = random.Random(1)
    pairs = random_pairs(rng, 60)
    alone, _ = score(model, pairs, pad_id=0, max_tokens=1)  # every pair in a batch of its own, no padding
    together, _ = score(model, pairs, pad_id=0, max_tokens=10**6)  # one batch, padded to the longest pair
    other_pad, _ = score(model, pairs, pad_id=VOCAB - 1, max_tokens=200)
    idx = rng.sample(range(len(pairs)), 25)
    subset, _ = score(model, [pairs[i] for i in idx], pad_id=7, max_tokens=300)
    for i in range(len(pairs)):
        assert abs(alone[i] - together[i]) <= 1e-9 and abs(alone[i] - other_pad[i]) <= 1e-9
    for j, i in enumerate(idx):
        assert abs(subset[j] - alone[i]) <= 1e-9


def test_gradients_equal_the_manual_computation(model):
    """Pass 2 differentiates the scorer: padding and the selected output rows must not change the gradient."""
    pairs = random_pairs(random.Random(2), 12)
    weights = torch.tensor([random.Random(3).uniform(0.1, 2.0) for _ in pairs], dtype=torch.float64)
    params = [p for p in model.parameters() if p.requires_grad]
    with torch.enable_grad():
        g1 = torch.autograd.grad(-(weights * forward_logp(model, pairs, pad_id=0)).sum(), params)
        manual = torch.stack([manual_logp(model, p) for p in pairs])
        g2 = torch.autograd.grad(-(weights * manual).sum(), params)
    for a, b in zip(g1, g2, strict=True):
        assert torch.allclose(a, b, rtol=1e-10, atol=1e-12)


def test_eval_mode_is_required():
    model = tiny_lm("qwen2")
    pairs = random_pairs(random.Random(4), 3)
    model.train()
    with pytest.raises(RuntimeError, match="eval mode"):
        score(model, pairs, pad_id=0)
    model.eval()
    model.model.layers[1].mlp.train()  # one submodule in training mode is enough to refuse
    with pytest.raises(RuntimeError, match="eval mode"):
        forward_logp(model, pairs, pad_id=0)
    model.eval()
    score(model, pairs, pad_id=0)


def test_token_counts():
    model = tiny_lm("llama")
    pairs = random_pairs(random.Random(5), 30)
    _, counts = score(model, pairs, pad_id=0, max_tokens=120)
    batches = micro_batches(pairs, 120)
    assert counts.sequences == 30
    assert counts.prompt_tokens == sum(len(p.prompt_ids) for p in pairs)
    assert counts.target_tokens == sum(len(p.target_ids) for p in pairs)
    assert counts.tokens == sum(len(p) for p in pairs)
    assert counts.padded_tokens == sum(len(b) * max(len(pairs[i]) for i in b) for b in batches)
    assert counts.seconds > 0
    assert counts.to_json()["tokens"] == counts.tokens


def test_micro_batches_cover_every_pair_once_within_the_budget():
    rng = random.Random(6)
    for _ in range(50):
        pairs = random_pairs(rng, rng.randint(1, 40))
        budget = rng.choice([1, 30, 64, 200, 10**6])
        batches = micro_batches(pairs, budget)
        assert sorted(i for b in batches for i in b) == list(range(len(pairs)))
        for b in batches:
            assert len(b) == 1 or len(b) * max(len(pairs[i]) for i in b) <= budget


def test_lora_wrapped_models_score_through_the_adapters():
    from peft import LoraConfig, get_peft_model

    model = get_peft_model(
        tiny_lm("qwen2"), LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules="all-linear")
    ).eval()
    torch.manual_seed(7)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.normal_(0.0, 0.3)  # B starts at 0, which would hide a path that skips the adapters
    pairs = random_pairs(random.Random(8), 10)
    got, _ = score(model, pairs, pad_id=0)
    with torch.no_grad():
        want = [manual_logp(model, p).item() for p in pairs]
        base = [manual_logp(model.get_base_model(), p).item() for p in pairs]
    for g, w in zip(got, want, strict=True):
        assert abs(g - w) <= 1e-12 * abs(w)
    with model.disable_adapter(), torch.no_grad():
        plain = [manual_logp(model, p).item() for p in pairs]
    assert max(abs(a - b) for a, b in zip(want, plain, strict=True)) > 1e-3  # the adapters change the scores
    assert base == want  # get_base_model() keeps the injected adapters


def test_pass_1_runs_without_gradients():
    """SPEC §6.5: pass 1 scores under no_grad; pass 2 (forward_logp with grad enabled) builds a graph."""
    model = tiny_lm("qwen2")
    seen = []
    handle = model.model.register_forward_pre_hook(lambda module, args: seen.append(torch.is_grad_enabled()))
    pairs = random_pairs(random.Random(11), 5)
    with torch.enable_grad():
        score(model, pairs, pad_id=0)
        assert seen and not any(seen)
        seen.clear()
        out = forward_logp(model, pairs, pad_id=0)
    handle.remove()
    assert seen == [True] and out.requires_grad and out.dtype == torch.float64


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_log_softmax_in_float32_and_sums_in_float64(dtype):
    """SPEC §6.7 on a half-precision model: the reference takes the same output-layer logits, upcasts them to float32
    for the log-softmax and sums in float64; a log-softmax in the model's dtype would differ."""
    model = tiny_lm("llama", dtype=dtype)
    pairs = random_pairs(random.Random(12), 8)
    width = max(len(p) for p in pairs)
    ids = torch.zeros((len(pairs), width), dtype=torch.long)
    rows, cols, targets = [], [], []
    for i, p in enumerate(pairs):
        ids[i, : len(p)] = torch.tensor(p.prompt_ids + p.target_ids)
        rows += [i] * len(p.target_ids)
        cols += range(len(p.prompt_ids) - 1, len(p) - 1)
        targets += p.target_ids
    with torch.no_grad():
        hidden = model.model(input_ids=ids, attention_mask=torch.ones_like(ids)).last_hidden_state
        logits = model.lm_head(hidden[torch.tensor(rows), torch.tensor(cols)])
        tok = torch.tensor(targets)[:, None]
        exact = logits.float().log_softmax(-1).gather(1, tok).squeeze(1).double()
        native = logits.log_softmax(-1).gather(1, tok).squeeze(1).double()
        want = torch.zeros(len(pairs), dtype=torch.float64).index_add(0, torch.tensor(rows), exact)
        low = torch.zeros(len(pairs), dtype=torch.float64).index_add(0, torch.tensor(rows), native)
        got = forward_logp(model, pairs, pad_id=0)
    assert got.dtype == torch.float64
    assert torch.allclose(got, want, rtol=1e-12, atol=0)
    if dtype != torch.float32:
        assert (got - low).abs().max() > 1e-3  # the test would notice a log-softmax in the model's dtype


def test_adapters_other_than_lora_and_dynamic_rope_are_refused():
    from peft import PrefixTuningConfig, PromptTuningConfig, get_peft_model

    pairs = random_pairs(random.Random(13), 2)
    for cfg in (
        PromptTuningConfig(task_type="CAUSAL_LM", num_virtual_tokens=4),
        PrefixTuningConfig(task_type="CAUSAL_LM", num_virtual_tokens=4),
    ):
        model = get_peft_model(tiny_lm("llama"), cfg).eval()
        with pytest.raises(ValueError, match="plain LoRA"):
            score(model, pairs, pad_id=0)
    model = tiny_lm("llama", rope_scaling={"rope_type": "dynamic", "factor": 2.0})
    with pytest.raises(ValueError, match="dynamic RoPE"):
        score(model, pairs, pad_id=0)


def test_micro_batches_cap_target_tokens():
    pairs = random_pairs(random.Random(14), 40)
    for cap in (1, 9, 20, 50):
        batches = micro_batches(pairs, 10**6, max_target_tokens=cap)
        assert sorted(i for b in batches for i in b) == list(range(40))
        for b in batches:
            assert len(b) == 1 or sum(len(pairs[i].target_ids) for i in b) <= cap


def test_unsupported_model_types_are_refused():
    from transformers import GPT2Config, GPT2LMHeadModel

    model = GPT2LMHeadModel(GPT2Config(vocab_size=50, n_embd=16, n_layer=1, n_head=2)).eval()
    with pytest.raises(ValueError, match="unsupported model type"):
        score(model, random_pairs(random.Random(9), 2, vocab=50), pad_id=0)


@pytest.mark.gpu
def test_batch_and_padding_invariance_in_bf16_on_gpu():
    """The real backbone in bf16 on one GPU: scores do not depend on the batch within 1e-4."""
    from transformers import AutoModelForCausalLM

    from qet.core.canonical import action_string
    from qet.core.render import render
    from qet.train.chat import ChatFormat, backbone

    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    fmt = ChatFormat.load(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=backbone(model_id).revision, dtype=torch.bfloat16
    ).to("cuda").eval()  # fmt: skip
    task = {
        "request": "Compute f3 for x = 412 and y = 907.",
        "tools": [{"name": f"f{i}", "params": ["a", "b"], "description": f"Function {i}."} for i in range(6)],
    }
    contracts = {f"f{i}": {"reads": [], "writes": []} for i in range(6)}
    rng = random.Random(10)
    pairs = []
    for k in range(24):
        history = [
            {"tool": f"f{rng.randrange(6)}", "args": {"a": rng.randrange(1000), "b": rng.randrange(1000)},
             "obs": {"value": rng.randrange(1000)}}
            for _ in range(k % 6)
        ]  # fmt: skip
        action = action_string(f"f{rng.randrange(6)}", {"a": rng.randrange(1000), "b": rng.randrange(1000)})
        prompt = fmt.prompt_ids(render(task, history, contracts).messages())
        pairs.append(Pair(tuple(prompt), tuple(fmt.target_ids(action))))
    alone, _ = score(model, pairs, pad_id=fmt.eot_id, max_tokens=1)
    together, _ = score(model, pairs, pad_id=fmt.eot_id, max_tokens=10**6)
    mixed, _ = score(model, pairs, pad_id=fmt.eot_id, max_tokens=4096)
    worst = max(max(abs(a - b), abs(a - c)) for a, b, c in zip(alone, together, mixed, strict=True))
    print(f"bf16 batch invariance: worst |delta log pi| = {worst:.3g} over {len(pairs)} pairs")
    assert all(math.isfinite(x) and x < 0 for x in alone)
    assert worst <= 1e-4
    full, _ = score(model.float(), pairs, pad_id=fmt.eot_id, max_tokens=4096)  # the same weights in float32
    gap = max(abs(a - b) for a, b in zip(alone, full, strict=True))
    print(f"bf16 vs float32 scores: worst |delta log pi| = {gap:.3g}, mean log pi = {sum(full) / len(full):.3f}")
    assert gap < 1.0  # a gross error (a wrong dtype path, a different function) shows up as nats

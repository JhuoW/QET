"""Scoring: log pi(target | prompt) for (prompt ids, target ids) pairs (SPEC §6.2, §6.7).

log pi is the sum of the target tokens' log-probabilities, the end-of-turn token included: no length normalization and
no renormalization over the legal actions. Log-softmax runs in float32 (in float64 for a float64 model) and the sums in
float64. The model must be in eval mode (dropout inactive) in pass 1, which scores without gradients, and in pass 2,
which differentiates the same function for the weighted SFT loss.

Sequences are right-padded and run with an all-ones attention mask, so only the causal mask applies: a real token never
attends to the padding after it and keeps its position 0..len-1, so a pair's log pi does not depend on the batch it is
in (exactly with sdpa in float64; eager attention computes its softmax in float32), and SDPA may use its causal
kernels. Only the hidden states at target positions go through the output layer. Plain LoRA is the only adapter
accepted: prompt and prefix tuning act inside `PeftModel.forward`, which the scorer bypasses. Dynamic RoPE scaling is
refused because its frequencies depend on the padded width (D-024).
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch

SUPPORTED_MODEL_TYPES = ("qwen2", "llama")  # decoder + output layer give the logits, with no post-processing


@dataclass(frozen=True)
class Pair:
    """One (prompt, target) to score: the target is an action string's tokens plus the end-of-turn id."""

    prompt_ids: tuple[int, ...]
    target_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.prompt_ids or not self.target_ids:
            raise ValueError("a pair needs a non-empty prompt and a non-empty target")

    def __len__(self) -> int:
        return len(self.prompt_ids) + len(self.target_ids)


@dataclass
class TokenCounts:
    """Token accounting of forward passes (SPEC §6.7; EVAL_PLAN §8 matches compute on these)."""

    sequences: int = 0
    prompt_tokens: int = 0
    target_tokens: int = 0
    padded_tokens: int = 0  # positions computed, padding included
    seconds: float = 0.0

    @property
    def tokens(self) -> int:
        """Real tokens processed: prompts plus targets."""
        return self.prompt_tokens + self.target_tokens

    def __iadd__(self, other: TokenCounts) -> TokenCounts:
        self.sequences += other.sequences
        self.prompt_tokens += other.prompt_tokens
        self.target_tokens += other.target_tokens
        self.padded_tokens += other.padded_tokens
        self.seconds += other.seconds
        return self

    def to_json(self) -> dict[str, Any]:
        return {**asdict(self), "tokens": self.tokens}


def count_tokens(pairs: Sequence[Pair], *, seconds: float) -> TokenCounts:
    return TokenCounts(
        sequences=len(pairs),
        prompt_tokens=sum(len(p.prompt_ids) for p in pairs),
        target_tokens=sum(len(p.target_ids) for p in pairs),
        padded_tokens=len(pairs) * max((len(p) for p in pairs), default=0),
        seconds=seconds,
    )


def assert_eval_mode(model: torch.nn.Module) -> None:
    """Pass 1 and pass 2 both run in eval mode (SPEC §6.5): no module may be in training mode."""
    training = [name or "<model>" for name, m in model.named_modules() if m.training]
    if training:
        raise RuntimeError(f"scoring needs the model in eval mode; in training mode: {training[:5]}")


def _decoder_and_head(model: torch.nn.Module) -> tuple[torch.nn.Module, torch.nn.Module]:
    peft_config = getattr(model, "peft_config", None)
    if peft_config is not None:  # a PEFT model wraps the original, with LoRA injected into its linear layers
        for name, cfg in peft_config.items():
            kind = getattr(cfg.peft_type, "value", cfg.peft_type)
            if kind != "LORA" or getattr(cfg, "alora_invocation_tokens", None):
                raise ValueError(f"adapter {name!r} is {kind}: the scorer supports plain LoRA only")
        base = model.get_base_model()
    else:
        base = model
    model_type = getattr(base.config, "model_type", None)
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(f"unsupported model type {model_type!r}; supported: {SUPPORTED_MODEL_TYPES}")
    rope = getattr(base.config, "rope_scaling", None) or {}
    if (rope.get("rope_type") or rope.get("type")) == "dynamic":
        raise ValueError("dynamic RoPE scaling depends on the padded width; scores would depend on the batch")
    return base.get_decoder(), base.get_output_embeddings()


def forward_logp(model: torch.nn.Module, pairs: Sequence[Pair], pad_id: int) -> torch.Tensor:
    """log pi of each pair from one forward pass, as a float64 tensor; differentiable when grad is enabled."""
    if not pairs:
        raise ValueError("no pairs to score")
    assert_eval_mode(model)
    decoder, head = _decoder_and_head(model)
    device = head.weight.device
    width = max(len(p) for p in pairs)
    ids = torch.full((len(pairs), width), pad_id, dtype=torch.long)
    rows: list[int] = []
    cols: list[int] = []
    targets: list[int] = []
    for i, p in enumerate(pairs):
        seq = p.prompt_ids + p.target_ids
        ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        start = len(p.prompt_ids) - 1  # the logits at position t predict token t + 1
        rows += [i] * len(p.target_ids)
        cols += range(start, start + len(p.target_ids))
        targets += p.target_ids
    ids = ids.to(device)
    hidden = decoder(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).last_hidden_state
    rows_t = torch.tensor(rows, device=device)
    logits = head(hidden[rows_t, torch.tensor(cols, device=device)])
    if logits.dtype in (torch.float16, torch.bfloat16):
        logits = logits.float()
    token_logp = logits.log_softmax(-1).gather(1, torch.tensor(targets, device=device)[:, None]).squeeze(1)
    zeros = torch.zeros(len(pairs), dtype=torch.float64, device=device)
    return zeros.index_add(0, rows_t, token_logp.to(torch.float64))


def micro_batches(pairs: Sequence[Pair], max_tokens: int, max_target_tokens: int | None = None) -> list[list[int]]:
    """Indices of the pairs in groups of at most `max_tokens` padded positions and, optionally, `max_target_tokens`
    target tokens (each needs vocabulary-sized logits), longest pairs first so that padding stays small. A pair over
    either limit goes alone."""
    order = sorted(range(len(pairs)), key=lambda i: (-len(pairs[i]), i))
    batches: list[list[int]] = []
    current: list[int] = []
    longest = targets = 0
    for i in order:
        n, t = len(pairs[i]), len(pairs[i].target_ids)
        too_wide = max(longest, n) * (len(current) + 1) > max_tokens
        too_many = max_target_tokens is not None and targets + t > max_target_tokens
        if current and (too_wide or too_many):
            batches.append(current)
            current, longest, targets = [], 0, 0
        current.append(i)
        longest, targets = max(longest, n), targets + t
    if current:
        batches.append(current)
    return batches


@torch.no_grad()
def score(
    model: torch.nn.Module,
    pairs: Sequence[Pair],
    *,
    pad_id: int,
    max_tokens: int = 16384,
    max_target_tokens: int | None = None,
) -> tuple[list[float], TokenCounts]:
    """Pass 1: log pi of every pair without gradients (float64), and the tokens it took."""
    out = [0.0] * len(pairs)
    counts = TokenCounts()
    for idx in micro_batches(pairs, max_tokens, max_target_tokens):
        t0 = time.perf_counter()
        batch = [pairs[i] for i in idx]
        values = forward_logp(model, batch, pad_id).tolist()  # .tolist() waits for the device
        for i, x in zip(idx, values, strict=True):
            out[i] = x
        counts += count_tokens(batch, seconds=time.perf_counter() - t0)
    return out, counts

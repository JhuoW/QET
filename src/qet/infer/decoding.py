"""Decoding one action per step and parsing it (SPEC §7, D-006, D-020b, D-022d, D-022f).

`parse_action` is lenient about spacing and key order but requires one JSON object with exactly `tool` and `args`, a
tool from the task's list, an object for args that has a canonical form, no duplicate keys and no special-token text.

`HFSampler` decodes with an explicit loop rather than `generate`, which would fill unset fields from the model's own
generation_config (Qwen2.5 samples at T = 0.7, top-p 0.8, top-k 20, repetition penalty 1.05 by default). Greedy takes
the argmax; sampling draws from softmax(logits / T) restricted to the top-p nucleus, which at T = 1.0 and top-p 1.0 is
the model's pi itself. Each step has its own seeded generator, so rollouts are reproducible and independent.
"""

from __future__ import annotations

import json
import math
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from qet.core.canonical import JSON, canon_call, canon_json
from qet.train.chat import special_text
from qet.train.scoring import assert_eval_mode


@dataclass(frozen=True)
class Action:
    tool: str
    args: dict[str, JSON]  # canonical


@dataclass(frozen=True)
class Invalid:
    reason: str


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [k for k, _ in pairs]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate key")
    return dict(pairs)


def _no_constants(name: str) -> Any:
    raise ValueError(f"{name} is not JSON")


def _valid_text(x: Any) -> bool:
    """Whether every string in x, object keys included, is valid Unicode (JSON escapes can spell lone surrogates)."""
    if isinstance(x, str):
        try:
            x.encode("utf-8")
        except UnicodeEncodeError:
            return False
        return True
    if isinstance(x, dict):
        return all(_valid_text(k) and _valid_text(v) for k, v in x.items())
    if isinstance(x, list):
        return all(_valid_text(v) for v in x)
    return True


def parse_action(text: str, tool_names: Collection[str]) -> Action | Invalid:
    """The model output as an action, or why it is invalid (an invalid action ends the episode, D-006)."""
    token = special_text(text)
    if token:
        return Invalid(f"special token {token} in the output")
    try:
        obj = json.loads(text, object_pairs_hook=_no_duplicate_keys, parse_constant=_no_constants)
    except RecursionError:
        return Invalid("nested too deeply")
    except ValueError as e:  # not JSON (json.JSONDecodeError is a ValueError), duplicate keys, NaN
        return Invalid(f"not one JSON object: {e}")
    if not isinstance(obj, dict) or set(obj) != {"tool", "args"}:
        return Invalid("not an object with exactly the keys tool and args")
    if not isinstance(obj["tool"], str) or obj["tool"] not in tool_names:
        return Invalid(f"unknown tool {obj['tool']!r}")
    if not isinstance(obj["args"], dict):
        return Invalid("args is not an object")
    try:
        if not _valid_text(obj["args"]):
            return Invalid("args contain text that is not valid Unicode (a lone surrogate)")
        tool, args = canon_call(obj["tool"], obj["args"])
    except RecursionError:
        return Invalid("nested too deeply")
    except (TypeError, ValueError) as e:  # an infinite float, keys that collide after NFC (D-020b)
        return Invalid(f"args have no canonical form: {e}")
    token = special_text(canon_json(args))  # escapes such as < hide special-token text from the raw check
    if token:
        return Invalid(f"special token {token} in the arguments")
    return Action(tool, args)


@dataclass(frozen=True)
class DecodeConfig:
    mode: str = "greedy"  # "greedy" or "sample"
    temperature: float = 1.0
    top_p: float = 1.0
    max_new_tokens: int = 256

    def __post_init__(self) -> None:
        if self.mode not in ("greedy", "sample"):
            raise ValueError(f"unknown decoding mode {self.mode!r}")
        if self.temperature <= 0 or not 0 < self.top_p <= 1 or self.max_new_tokens < 1:
            raise ValueError("need temperature > 0, 0 < top_p <= 1 and max_new_tokens >= 1")


@dataclass(frozen=True)
class Generation:
    ids: list[int]  # the generated ids, the end-of-turn id included when the model ended its turn
    ended: bool  # the end-of-turn id was generated within max_new_tokens
    logprob: float  # sum of the model's log-probabilities of `ids` (before temperature and top-p), float64


class HFSampler:
    """One action per call from a Hugging Face causal LM (optionally LoRA-wrapped), for tests and small runs."""

    def __init__(self, model: torch.nn.Module, eot_id: int, decode: DecodeConfig):
        self.model, self.eot_id, self.decode = model, eot_id, decode

    @torch.no_grad()
    def generate(self, prompt_ids: Sequence[int], seed: int) -> Generation:
        assert_eval_mode(self.model)
        device = next(self.model.parameters()).device
        gen = torch.Generator(device=device).manual_seed(seed % (1 << 63))
        out = self.model(input_ids=torch.tensor([list(prompt_ids)], device=device), use_cache=True, logits_to_keep=1)
        ids: list[int] = []
        total = 0.0
        d = self.decode
        for _ in range(d.max_new_tokens):
            logits = out.logits[0, -1]
            logits = logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits
            logp = logits.log_softmax(-1)
            if d.mode == "greedy":
                tok = int(logp.argmax())
            else:
                probs = (logits / d.temperature).softmax(-1)
                if d.top_p < 1.0:
                    sorted_p, order = probs.sort(descending=True)
                    keep = sorted_p.cumsum(-1) - sorted_p < d.top_p  # the smallest prefix with mass >= top_p
                    probs = torch.zeros_like(probs).scatter(0, order[keep], sorted_p[keep])
                tok = int(torch.multinomial(probs, 1, generator=gen))
            ids.append(tok)
            total += float(logp[tok].double())
            if tok == self.eot_id:
                return Generation(ids, True, total)
            out = self.model(
                input_ids=torch.tensor([[tok]], device=device), past_key_values=out.past_key_values, use_cache=True
            )
        return Generation(ids, False, total if math.isfinite(total) else -math.inf)

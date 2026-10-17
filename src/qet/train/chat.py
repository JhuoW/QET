"""Chat mapping and tokenization (SPEC §6.7, D-002, D-022).

A prompt is a list of chat messages: [system, user] for the quotient and chronological renderers, and the native
multi-turn list for ablation a1 (`qet.core.render.render_messages`). Prompt ids are the backbone's chat template with
add_generation_prompt=True; target ids are the action string's tokens (no special tokens) followed by the end-of-turn
id, so log pi includes the end of the turn. Each backbone is pinned to a Hugging Face revision, and a template that
would insert today's date (Llama 3.1) gets a fixed date string, so training, scoring and inference produce the same
ids on any day and in any process.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

LLAMA31_DATE = "26 Jul 2024"  # the Llama 3.1 template's own fallback when no clock is available (D-022)


@dataclass(frozen=True)
class Backbone:
    model_id: str
    revision: str  # Hugging Face commit of the model repository: weights, tokenizer and chat template
    family: str
    end_of_turn: str  # the token that closes an assistant turn (D-002)
    template_kwargs: tuple[tuple[str, str], ...] = ()  # fixed variables for the chat template


BACKBONES: dict[str, Backbone] = {
    b.model_id: b
    for b in (
        Backbone("Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775", "qwen2.5", "<|im_end|>"),
        Backbone("Qwen/Qwen2.5-7B-Instruct", "a09a35458c702b33eeacc393d103063234e8bc28", "qwen2.5", "<|im_end|>"),
        Backbone(
            "meta-llama/Llama-3.1-8B-Instruct",
            "0e9e39f249a16976918f6564b8830bc894c89659",
            "llama3.1",
            "<|eot_id|>",
            (("date_string", LLAMA31_DATE),),
        ),
    )
}


def backbone(model_id: str) -> Backbone:
    if model_id not in BACKBONES:
        raise ValueError(f"unsupported backbone {model_id!r}; supported: {sorted(BACKBONES)}")
    return BACKBONES[model_id]


def load_tokenizer(model_id: str) -> Any:
    """The backbone's tokenizer at its pinned revision (download it once where there is network access)."""
    from transformers import AutoTokenizer

    b = backbone(model_id)
    return AutoTokenizer.from_pretrained(b.model_id, revision=b.revision)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def check_messages(messages: Sequence[Mapping[str, Any]]) -> None:
    """An explicit system message first (Qwen inserts a default one otherwise, D-002), then user and assistant turns
    alternating and ending with a user turn: the model writes the next assistant turn."""
    if len(messages) < 2:
        raise ValueError("a prompt needs a system message and at least one user message")
    for i, m in enumerate(messages):
        expected = "system" if i == 0 else ("user" if i % 2 == 1 else "assistant")
        if not isinstance(m, Mapping) or set(m) != {"role", "content"}:
            raise ValueError(f"message {i} must have exactly the keys 'role' and 'content', got {m!r}")
        if m["role"] != expected:
            raise ValueError(f"message {i} must have role {expected!r}, got {m['role']!r}")
        if not isinstance(m["content"], str):
            raise TypeError(f"message {i} content must be a string, got {type(m['content']).__name__}")
    if messages[-1]["role"] != "user":
        raise ValueError("the last message must be a user message")


class ChatFormat:
    """Token ids of prompts and targets for one backbone; the same object serves training, scoring and inference."""

    def __init__(self, tokenizer: Any, bb: Backbone):
        eot = tokenizer.convert_tokens_to_ids(bb.end_of_turn)
        if not isinstance(eot, int) or tokenizer.convert_ids_to_tokens(eot) != bb.end_of_turn:
            raise ValueError(f"{bb.end_of_turn!r} is not a token of the {bb.model_id} tokenizer")
        self.tokenizer = tokenizer
        self.backbone = bb
        self.eot_id: int = eot
        self.template_kwargs: dict[str, str] = dict(bb.template_kwargs)
        special = sorted({t.content for t in tokenizer.added_tokens_decoder.values() if t.special}, key=len)
        self._special = re.compile("|".join(re.escape(s) for s in reversed(special))) if special else None

    @classmethod
    def load(cls, model_id: str) -> ChatFormat:
        return cls(load_tokenizer(model_id), backbone(model_id))

    def _check_text(self, text: str, what: str) -> None:
        """Text that spells a special token would be read as that control token (D-022): refuse it."""
        m = self._special.search(text) if self._special else None
        if m:
            raise ValueError(f"{what} contains the special token {m.group()!r} of {self.backbone.model_id}")

    def prompt_text(self, messages: Sequence[Mapping[str, Any]]) -> str:
        check_messages(messages)
        for i, m in enumerate(messages):
            self._check_text(m["content"], f"message {i}")
        return self.tokenizer.apply_chat_template(
            [dict(m) for m in messages], tokenize=False, add_generation_prompt=True, **self.template_kwargs
        )

    def prompt_ids(self, messages: Sequence[Mapping[str, Any]]) -> list[int]:
        """The template text already contains its special tokens (Llama's begin-of-text included)."""
        return self.encode(self.prompt_text(messages))

    def target_ids(self, action: str) -> list[int]:
        """tokenizer(action, no special tokens) + [end-of-turn id] (SPEC §6.7)."""
        self._check_text(action, "the action string")
        return self.encode(action) + [self.eot_id]

    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def decode(self, ids: Iterable[int]) -> str:
        """Exact inverse of `encode` on canonical (NFC) text: no clean-up of spaces before punctuation (Llama 3.1's
        tokenizer would do it by default)."""
        return self.tokenizer.decode(list(ids), skip_special_tokens=False, clean_up_tokenization_spaces=False)

    def fingerprint(self) -> dict[str, Any]:
        """What a run manifest records about the tokenizer and template (EVAL_PLAN §10)."""
        return {
            "backbone": self.backbone.model_id,
            "revision": self.backbone.revision,
            "tokenizer_sha256": sha256_text(self.tokenizer.backend_tokenizer.to_str()),
            "chat_template_sha256": sha256_text(self.tokenizer.chat_template),
            "template_kwargs": dict(sorted(self.template_kwargs.items())),
            "end_of_turn": self.backbone.end_of_turn,
        }

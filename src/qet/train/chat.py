"""Chat mapping and tokenization (SPEC §6.7, D-002, D-022).

A prompt is a list of chat messages: [system, user] for the quotient and chronological renderers, and the native
multi-turn list for ablation a1 (`qet.core.render.render_messages`). Prompt ids are the backbone's chat template with
add_generation_prompt=True; target ids are the action string's tokens (no special tokens) followed by the end-of-turn
id, so log pi includes the end of the turn. Each backbone is pinned to a Hugging Face revision, and the template's date
variable is fixed (Llama 3.1), so training, scoring and inference produce the same ids on any day and in any process.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import version
from typing import Any

LLAMA31_DATE = "26 Jul 2024"  # the date the Llama 3.1 template itself falls back to (D-022)

# Special-token text of every supported backbone has this form (<|im_end|>, <|eot_id|>, <|begin_of_text|>, ...).
# Refusing it everywhere keeps the data that is accepted the same for every backbone (D-022d).
SPECIAL_TEXT = re.compile(r"<\|[A-Za-z0-9_]+\|>")
ACTION_PREFIX = '{"tool":'  # every canonical action string starts so (SPEC §2, D-001)


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

# The probe whose ids the fingerprint hashes: library drift that changes ids changes the fingerprint (D-022a).
_PROBE_MESSAGES = (
    {"role": "system", "content": "You are a tool-using agent."},
    {"role": "user", "content": '[Task] Book NYC to SG for 1200.\n[Completed calls]\n  f(a="x , y") -> {"v":1}'},
)
_PROBE_ACTIONS = ('{"tool":"f","args":{"a":"x , y","b":[1,2.5,null]}}', '{"tool":"finish","args":{"answer":"é"}}')


def backbone(model_id: str) -> Backbone:
    if model_id not in BACKBONES:
        raise ValueError(f"unsupported backbone {model_id!r}; supported: {sorted(BACKBONES)}")
    return BACKBONES[model_id]


def load_tokenizer(model_id: str) -> Any:
    """The backbone's tokenizer at its pinned revision (`make fetch` downloads it once where there is network)."""
    from transformers import AutoTokenizer

    b = backbone(model_id)
    return AutoTokenizer.from_pretrained(b.model_id, revision=b.revision)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def special_text(text: str) -> str | None:
    """The first special-token string in `text`, for any supported backbone (adapters sanitize observations with
    this before Compile and at inference alike, D-022d), or None."""
    m = SPECIAL_TEXT.search(text)
    return m.group() if m else None


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
    """Token ids of prompts and targets for one backbone; the same object serves training, scoring and inference.
    Inference passes `prompt_ids` to the model (re-tokenizing `prompt_text` with default settings would add a second
    begin-of-text on Llama) and decodes outputs with `decode` (D-022f)."""

    def __init__(self, tokenizer: Any, bb: Backbone):
        if tokenizer.name_or_path != bb.model_id:
            raise ValueError(f"tokenizer {tokenizer.name_or_path!r} does not belong to backbone {bb.model_id!r}")
        eot = tokenizer.convert_tokens_to_ids(bb.end_of_turn)
        if not isinstance(eot, int) or tokenizer.convert_ids_to_tokens(eot) != bb.end_of_turn:
            raise ValueError(f"{bb.end_of_turn!r} is not a token of the {bb.model_id} tokenizer")
        self.tokenizer = tokenizer
        self.backbone = bb
        self.eot_id: int = eot
        self.template_kwargs: dict[str, str] = dict(bb.template_kwargs)
        special = sorted(
            {t.content for t in tokenizer.added_tokens_decoder.values() if t.special}, key=lambda s: (len(s), s)
        )
        self._special = re.compile("|".join(re.escape(s) for s in reversed(special))) if special else None

    @classmethod
    def load(cls, model_id: str) -> ChatFormat:
        return cls(load_tokenizer(model_id), backbone(model_id))

    def _check_text(self, text: str, what: str) -> None:
        """Text that spells a special token would be read as that control token (D-022d); lone surrogates cannot
        be tokenized at all."""
        try:
            text.encode("utf-8")
        except UnicodeEncodeError as e:
            raise ValueError(f"{what} is not valid Unicode text ({e.reason})") from None
        found = special_text(text) or (m.group() if self._special and (m := self._special.search(text)) else None)
        if found:
            raise ValueError(f"{what} contains the special token {found!r}")

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
        """tokenizer(action, no special tokens) + [end-of-turn id] (SPEC §6.7), for canonical action strings that
        decode back exactly: a string that is not NFC as a whole (an escape such as \\n before a combining mark) would
        come back composed from Qwen's normalizer, which is not the trained string (D-022c, Q16)."""
        if not action.startswith(ACTION_PREFIX) or action != action.strip():
            raise ValueError(f"not a canonical action string: {action[:60]!r}")
        self._check_text(action, "the action string")
        if not unicodedata.is_normalized("NFC", action):
            raise ValueError(f"the action string is not NFC as a whole: {action[:60]!r}")
        ids = self.encode(action)
        if self.decode(ids) != action:
            raise ValueError(f"the action string does not survive tokenization: {action[:60]!r}")
        return ids + [self.eot_id]

    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def decode(self, ids: Iterable[int]) -> str:
        """Exact inverse of `encode` on NFC text: no clean-up of spaces before punctuation (Llama 3.1's tokenizer
        would do it by default)."""
        return self.tokenizer.decode(list(ids), skip_special_tokens=False, clean_up_tokenization_spaces=False)

    def fingerprint(self) -> dict[str, Any]:
        """What a run manifest records about the tokenizer and template (EVAL_PLAN §10). The tokenizer hash leaves out
        the backend's transient truncation and padding settings; the probe hash covers the ids themselves."""
        state = json.loads(self.tokenizer.backend_tokenizer.to_str())
        state.pop("truncation", None)
        state.pop("padding", None)
        probe = {"prompt": self.prompt_ids(_PROBE_MESSAGES), "targets": [self.target_ids(a) for a in _PROBE_ACTIONS]}
        return {
            "backbone": self.backbone.model_id,
            "revision": self.backbone.revision,
            "tokenizer_sha256": sha256_text(json.dumps(state, sort_keys=True, ensure_ascii=False)),
            "chat_template_sha256": sha256_text(self.tokenizer.chat_template),
            "template_kwargs": dict(sorted(self.template_kwargs.items())),
            "end_of_turn": self.backbone.end_of_turn,
            "probe_ids_sha256": sha256_text(json.dumps(probe)),
            "libraries": {lib: version(lib) for lib in ("transformers", "tokenizers", "jinja2")},
        }

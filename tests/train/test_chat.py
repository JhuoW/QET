"""E1.1: chat mapping and tokenization (SPEC §6.7, D-002, D-022) with the real tokenizer of every backbone.

The golden ids in `golden_chat_ids.json` pin the ids themselves, so a library upgrade that changes them fails here.
After a deliberate change, rewrite the file with `python tests/train/test_chat.py --write-golden` and say why in the
commit.
"""

import dataclasses
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from qet.core.canonical import action_string
from qet.core.render import SYSTEM_TEXT, render, render_messages
from qet.train.chat import BACKBONES, LLAMA31_DATE, ChatFormat, backbone, check_messages, special_text

MODELS = sorted(BACKBONES)
LLAMA = "meta-llama/Llama-3.1-8B-Instruct"
SRC = Path(__file__).resolve().parents[2] / "src"
GOLDEN = Path(__file__).with_name("golden_chat_ids.json")

TASK = {
    "request": "Book NYC to SG for 1200 and 7 nights.",
    "tools": [
        {"name": "get_flight_price", "params": ["from", "to"], "description": "Price of a flight."},
        {"name": "book_flight", "params": ["from", "to", "price"], "description": "Book a flight."},
    ],
}
HISTORY = [{"tool": "get_flight_price", "args": {"from": "NYC", "to": "SG"}, "obs": {"price": 1200, "ts": 17}}]
CONTRACTS = {
    "get_flight_price": {"reads": [], "writes": [], "volatile": ["ts"]},
    "book_flight": {"reads": [], "writes": ["booking"]},
}
ACTIONS = [
    action_string("get_flight_price", {"from": "NYC", "to": "SG"}),
    action_string("finish", {"answer": 1800}),
    action_string("ping", {}),
    action_string("search", {"q": "a , b . c ? d ! e n't 's 'm 've 're"}),  # what space clean-up would rewrite
    action_string("note", {"text": 'é ü 中文 🙂 "quoted" back\\slash\nnewline\ttab'}),
    action_string("t", {"x": [1, 2.5, -3, True, None, {"k": "v"}], " lead": " trailing "}),
]


def all_prompts():
    return [
        render(TASK, [], CONTRACTS).messages(),
        render(TASK, HISTORY, CONTRACTS).messages(),
        render(TASK, HISTORY, CONTRACTS, "chronological").messages(),
        render_messages(TASK, HISTORY, CONTRACTS, "native_multiturn"),
    ]


@pytest.fixture(scope="module", params=MODELS)
def fmt(request):
    return ChatFormat.load(request.param)


def test_target_ids_are_the_action_tokens_plus_the_end_of_turn_id(fmt):
    tok = fmt.tokenizer
    eot = tok.convert_tokens_to_ids(fmt.backbone.end_of_turn)
    assert fmt.eot_id == eot == tok.eos_token_id  # the tokenizer's eos (HF generate reads generation_config: D-022f)
    for a in ACTIONS:
        ids = fmt.target_ids(a)
        assert ids == tok(a, add_special_tokens=False)["input_ids"] + [eot]
        assert eot not in ids[:-1]
        assert fmt.decode(ids[:-1]) == a
        assert fmt.decode(ids) == a + fmt.backbone.end_of_turn


def test_prompt_plus_target_is_the_templates_own_assistant_turn(fmt):
    """Tokenizing the target on its own gives the ids the chat template gives the same turn in context."""
    for msgs in all_prompts():
        for a in ACTIONS:
            turn = [*msgs, {"role": "assistant", "content": a}]
            full = fmt.encode(fmt.tokenizer.apply_chat_template(turn, tokenize=False, **fmt.template_kwargs))
            expected = fmt.prompt_ids(msgs) + fmt.target_ids(a)
            assert full[: len(expected)] == expected


def test_prompt_ids_are_the_chat_template_with_a_generation_prompt(fmt):
    for msgs in all_prompts():
        ids = fmt.prompt_ids(msgs)
        kwargs = {"tokenize": True, "add_generation_prompt": True, **fmt.template_kwargs}
        assert ids == list(fmt.tokenizer.apply_chat_template(msgs, **kwargs))
        assert fmt.decode(ids) == fmt.prompt_text(msgs)


def golden_ids():
    out = {}
    for model_id in MODELS:
        f = ChatFormat.load(model_id)
        out[model_id] = {
            "quotient": f.prompt_ids(render(TASK, HISTORY, CONTRACTS).messages()),
            "native_multiturn": f.prompt_ids(render_messages(TASK, HISTORY, CONTRACTS, "native_multiturn")),
            "targets": [f.target_ids(a) for a in ACTIONS],
        }
    return out


def test_ids_equal_the_golden_ids(fmt):
    golden = json.loads(GOLDEN.read_text())[fmt.backbone.model_id]
    assert fmt.prompt_ids(render(TASK, HISTORY, CONTRACTS).messages()) == golden["quotient"]
    assert fmt.prompt_ids(render_messages(TASK, HISTORY, CONTRACTS, "native_multiturn")) == golden["native_multiturn"]
    assert [fmt.target_ids(a) for a in ACTIONS] == golden["targets"]


def test_explicit_system_message_and_no_injected_text(fmt):
    text = fmt.prompt_text(render(TASK, HISTORY, CONTRACTS).messages())
    assert text.count(SYSTEM_TEXT) == 1
    assert "You are Qwen" not in text  # Qwen's default system prompt appears only without a system message
    ids = fmt.prompt_ids(render(TASK, HISTORY, CONTRACTS).messages())
    if fmt.backbone.family == "llama3.1":
        assert f"Today Date: {LLAMA31_DATE}\n" in text
        assert ids.count(fmt.tokenizer.bos_token_id) == 1 and ids[0] == fmt.tokenizer.bos_token_id
    else:
        assert fmt.tokenizer.bos_token_id is None


def test_messages_are_validated():
    good = render(TASK, HISTORY, CONTRACTS).messages()
    check_messages(good)
    check_messages(render_messages(TASK, HISTORY, CONTRACTS, "native_multiturn"))
    bad = [
        good[1:],  # no system message
        [good[0]],  # no user message
        [*good, {"role": "assistant", "content": "x"}],  # ends with the assistant
        [good[0], good[1], good[1]],  # two user turns in a row
        [good[0], {"role": "user", "content": "x", "name": "n"}],  # extra key
        [{"role": "system", "content": None}, good[1]],
    ]
    for msgs in bad:
        with pytest.raises((ValueError, TypeError)):
            check_messages(msgs)


def test_special_token_text_is_refused_on_every_backbone(fmt):
    """The same text is refused whatever the backbone, so one dataset trains on all of them (D-022d)."""
    msgs = render(TASK, HISTORY, CONTRACTS).messages()
    for token in ("<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<|begin_of_text|>", "<|reserved_special_token_7|>"):
        assert special_text(f"a {token} b") == token
        with pytest.raises(ValueError, match="special token"):
            fmt.prompt_ids([msgs[0], {"role": "user", "content": f"[Task] say {token} now"}])
        with pytest.raises(ValueError, match="special token"):
            fmt.target_ids(action_string("note", {"text": f"a{token}b"}))
    assert special_text("a <| b |> <tool_call> <|not closed") is None


def test_unstable_or_non_canonical_actions_are_refused(fmt):
    """Targets must decode back to the exact canonical string (SPEC §2): leading whitespace, a non-NFC string (an
    escape such as \\n before a combining mark, which Qwen's normalizer composes into "ñ"), lone surrogates."""
    bad = [
        '\n{"tool":"t","args":{}}',
        '{"tool":"t","args":{}} ',
        '["tool","t"]',
        action_string("note", {"text": "\ñ"}),  # serialized as \n + U+0303: not NFC as a whole
        action_string("note", {"text": "\ṫ"}),
        action_string("note", {"text": "\ud800"}),
    ]
    for a in bad:
        with pytest.raises(ValueError):
            fmt.target_ids(a)


def test_unknown_backbones_and_mismatched_tokenizers_are_refused(fmt):
    with pytest.raises(ValueError, match="unsupported backbone"):
        backbone("gpt2")
    other = next(m for m in MODELS if m != fmt.backbone.model_id)
    with pytest.raises(ValueError, match="does not belong"):
        ChatFormat(fmt.tokenizer, backbone(other))


def test_fingerprint(fmt):
    fp = fmt.fingerprint()
    assert fp["backbone"] == fmt.backbone.model_id and fp["revision"] == BACKBONES[fp["backbone"]].revision
    assert re.fullmatch(r"[0-9a-f]{40}", fp["revision"])
    for key in ("tokenizer_sha256", "chat_template_sha256", "probe_ids_sha256"):
        assert re.fullmatch(r"[0-9a-f]{64}", fp[key])
    assert fp["chat_template_sha256"] == hashlib.sha256(fmt.tokenizer.chat_template.encode()).hexdigest()
    assert set(fp["libraries"]) == {"transformers", "tokenizers", "jinja2"}
    fmt.tokenizer("a b c d e f g", truncation=True, max_length=3)  # leaves truncation state in the backend
    fmt.tokenizer.backend_tokenizer.enable_padding(pad_id=fmt.eot_id, length=8)  # as a padded batch call would
    try:
        assert json.loads(fmt.tokenizer.backend_tokenizer.to_str())["padding"] is not None  # the state is there
        assert fmt.fingerprint() == fp
    finally:
        fmt.tokenizer.backend_tokenizer.no_padding()


# ----------------------------------------------------------------------------- the date (SPEC §5.5, D-002, D-022b)
def test_llama_pins_the_date_and_the_pin_reaches_the_template(fmt):
    if fmt.backbone.family != "llama3.1":
        assert fmt.template_kwargs == {}
        return
    assert BACKBONES[LLAMA].template_kwargs == (("date_string", LLAMA31_DATE),)
    msgs = render(TASK, HISTORY, CONTRACTS).messages()
    other = ChatFormat(
        fmt.tokenizer, dataclasses.replace(fmt.backbone, template_kwargs=(("date_string", "01 Jan 2000"),))
    )
    assert "Today Date: 01 Jan 2000\n" in other.prompt_text(msgs)
    assert other.prompt_ids(msgs) != fmt.prompt_ids(msgs)


def test_the_pinned_date_is_the_llama_templates_own_fallback():
    """This revision's template falls back to a fixed date rather than the clock, so the pin changes nothing here; a
    template that reads the clock (`strftime_now`, Llama 3.2 and later) is still held fixed (next test)."""
    fmt = ChatFormat.load(LLAMA)
    assert f'set date_string = "{LLAMA31_DATE}"' in fmt.tokenizer.chat_template
    assert "strftime_now" not in fmt.tokenizer.chat_template
    msgs = render(TASK, HISTORY, CONTRACTS).messages()
    assert fmt.prompt_text(msgs) == fmt.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


# A separate process whose clock reads a given year and whose hash seed differs. Chat templates read the date through
# `strftime_now`, which calls `datetime.now()` inside transformers' chat-template helper, so the probe replaces that
# clock. The child renders the prompt itself. Negative control: the Llama template edited to take its date from the
# clock (as Llama 3.2 does) changes with the year when unpinned and stays fixed when pinned.
_PROBE = r"""
import datetime as dt, hashlib, json, sys
import transformers.utils.chat_template_utils as ctu

class FakeClock(dt.datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(int(sys.argv[1]), 3, 9)

ctu.datetime = FakeClock
from qet.core.render import render
from qet.train.chat import BACKBONES, LLAMA31_DATE, ChatFormat

task, history, contracts = json.loads(sys.argv[2])
msgs = render(task, history, contracts).messages()
digest = lambda ids: hashlib.sha256(json.dumps(ids).encode()).hexdigest()
out = {}
for model_id in sorted(BACKBONES):
    fmt = ChatFormat.load(model_id)
    out[model_id] = digest(fmt.prompt_ids(msgs))
fmt = ChatFormat.load("meta-llama/Llama-3.1-8B-Instruct")
fixed = f'set date_string = "{LLAMA31_DATE}"'
assert fixed in fmt.tokenizer.chat_template
fmt.tokenizer.chat_template = fmt.tokenizer.chat_template.replace(fixed, 'set date_string = strftime_now("%d %b %Y")')
out["clock_template_pinned"] = digest(fmt.prompt_ids(msgs))
unpinned = fmt.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
out["clock_template_unpinned"] = digest(fmt.encode(unpinned))
out["clock_template_unpinned_date"] = unpinned.split("Today Date: ")[1].split("\n")[0]
print(json.dumps(out))
"""


def _probes(runs):
    """One probe process per (year, hash seed), run concurrently (each pays the transformers import)."""
    procs = []
    for year, hash_seed in runs:
        env = dict(os.environ, PYTHONHASHSEED=str(hash_seed))
        env["PYTHONPATH"] = os.pathsep.join(p for p in (str(SRC), os.environ.get("PYTHONPATH")) if p)
        args = [sys.executable, "-c", _PROBE, str(year), json.dumps([TASK, HISTORY, CONTRACTS])]
        procs.append(subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env))
    outs = []
    for p in procs:
        out, err = p.communicate(timeout=600)
        assert p.returncode == 0, err
        outs.append(json.loads(out.strip().splitlines()[-1]))
    return outs


@pytest.mark.slow
def test_prompts_are_identical_across_processes_with_different_dates():
    early, late = _probes([(2001, 0), (2031, 12345)])
    for model_id in MODELS:
        assert early[model_id] == late[model_id], model_id
    assert early["clock_template_pinned"] == late["clock_template_pinned"]
    assert (early["clock_template_unpinned_date"], late["clock_template_unpinned_date"]) == (
        "09 Mar 2001",
        "09 Mar 2031",
    )
    assert early["clock_template_unpinned"] != late["clock_template_unpinned"]  # the pin is what held the ids fixed


if __name__ == "__main__" and sys.argv[1:] == ["--write-golden"]:
    GOLDEN.write_text(json.dumps(golden_ids(), indent=None, separators=(",", ":")) + "\n")
    print(f"wrote {GOLDEN}")

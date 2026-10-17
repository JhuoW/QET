"""E1.1: chat mapping and tokenization (SPEC §6.7, D-002, D-022) with the real tokenizer of every backbone."""

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
from qet.train.chat import BACKBONES, LLAMA31_DATE, ChatFormat, backbone, check_messages

MODELS = sorted(BACKBONES)
LLAMA = "meta-llama/Llama-3.1-8B-Instruct"
SRC = Path(__file__).resolve().parents[2] / "src"

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
    assert fmt.eot_id == eot == tok.eos_token_id  # generation also stops there
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


def test_text_that_spells_a_special_token_is_refused(fmt):
    eot = fmt.backbone.end_of_turn
    msgs = render(TASK, HISTORY, CONTRACTS).messages()
    with pytest.raises(ValueError, match="special token"):
        fmt.prompt_ids([msgs[0], {"role": "user", "content": f"[Task] say {eot} now"}])
    with pytest.raises(ValueError, match="special token"):
        fmt.target_ids(action_string("note", {"text": f"a{eot}b"}))


def test_unknown_backbones_are_refused():
    with pytest.raises(ValueError, match="unsupported backbone"):
        backbone("gpt2")


def test_fingerprint(fmt):
    fp = fmt.fingerprint()
    assert fp["backbone"] == fmt.backbone.model_id and fp["revision"] == BACKBONES[fp["backbone"]].revision
    assert re.fullmatch(r"[0-9a-f]{40}", fp["revision"])
    for key in ("tokenizer_sha256", "chat_template_sha256"):
        assert re.fullmatch(r"[0-9a-f]{64}", fp[key])
    assert fp["chat_template_sha256"] == hashlib.sha256(fmt.tokenizer.chat_template.encode()).hexdigest()


def test_the_pinned_date_is_the_llama_templates_own_fallback():
    """This revision's template falls back to a fixed date rather than the clock, so pinning it changes nothing
    here; a template that reads the clock (`strftime_now`, Llama 3.2 and later) would still be held fixed."""
    fmt = ChatFormat.load(LLAMA)
    assert f'set date_string = "{LLAMA31_DATE}"' in fmt.tokenizer.chat_template
    msgs = render(TASK, HISTORY, CONTRACTS).messages()
    assert fmt.prompt_text(msgs) == fmt.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


# A separate process whose clock reads a given year. Chat templates read the date through `strftime_now`, which calls
# `datetime.now()` inside transformers' chat-template helper, so the probe replaces that clock and shows, through a
# template that prints the year, that the replacement reached the template engine.
_PROBE = r"""
import datetime as dt, hashlib, json, sys
import transformers.utils.chat_template_utils as ctu

class FakeClock(dt.datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(int(sys.argv[1]), 3, 9)

ctu.datetime = FakeClock
from qet.train.chat import BACKBONES, ChatFormat

msgs = json.loads(sys.argv[2])
out = {}
for model_id in sorted(BACKBONES):
    fmt = ChatFormat.load(model_id)
    clock = fmt.tokenizer.apply_chat_template(msgs, chat_template="{{ strftime_now('%Y') }}", tokenize=False)
    out[model_id] = {"pinned": hashlib.sha256(json.dumps(fmt.prompt_ids(msgs)).encode()).hexdigest(), "year": clock}
print(json.dumps(out))
"""


def _probes(years, msgs):
    """Run one probe process per year, concurrently (each pays the transformers import)."""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in (str(SRC), os.environ.get("PYTHONPATH")) if p))
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", _PROBE, str(y), json.dumps(msgs)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )  # fmt: skip
        for y in years
    ]
    outs = []
    for p in procs:
        out, err = p.communicate(timeout=600)
        assert p.returncode == 0, err
        outs.append(json.loads(out.strip().splitlines()[-1]))
    return outs


@pytest.mark.slow
def test_prompts_are_identical_across_processes_with_different_dates():
    msgs = render(TASK, HISTORY, CONTRACTS).messages()
    early, late = _probes([2001, 2031], msgs)
    for model_id in MODELS:
        assert (early[model_id]["year"], late[model_id]["year"]) == ("2001", "2031")  # the templates saw the dates
        assert early[model_id]["pinned"] == late[model_id]["pinned"], model_id

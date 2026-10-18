"""D-026b: the vLLM sampler's reading of the engine's outputs (no GPU: a fake engine stands in for vLLM)."""

import sys
import types
from types import SimpleNamespace

import pytest

from qet.infer.decoding import DecodeConfig


@pytest.fixture
def fake_vllm(monkeypatch):
    """Just enough of the vllm package for VLLMSampler: SamplingParams records its arguments, TokensPrompt is a dict."""
    vllm = types.ModuleType("vllm")
    vllm.SamplingParams = lambda **kw: SimpleNamespace(**kw)
    inputs = types.ModuleType("vllm.inputs")
    inputs.TokensPrompt = dict
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.inputs", inputs)


class Engine:
    def __init__(self, results):
        self.results, self.params = results, None

    def generate(self, prompts, params, lora_request=None, use_tqdm=False):
        self.params = params
        return [SimpleNamespace(outputs=[SimpleNamespace(token_ids=ids, finish_reason=why, cumulative_logprob=-1.0)])
                for ids, why in self.results]  # fmt: skip


def test_the_end_of_turn_ends_an_output_even_at_the_token_limit(fake_vllm):
    from qet.infer.vllm_sampler import VLLMSampler

    eot = 7
    engine = Engine([([1, 2, eot], "stop"), ([1, 2, 3], "length"), ([1, 2, eot], "length"), ([1, 2], "stop")])
    gens = VLLMSampler(engine, eot, DecodeConfig("greedy", max_new_tokens=3)).generate_batch([[0]] * 4, [1, 2, 3, 4])
    assert [(g.ids, g.ended) for g in gens] == [([1, 2, eot], True), ([1, 2, 3], False), ([1, 2, eot], True),
                                                 ([1, 2, eot], True)]  # fmt: skip
    p = engine.params[0]
    assert (p.temperature, p.top_k, p.stop_token_ids, p.ignore_eos, p.max_tokens) == (0.0, 0, [eot], True, 3)
    assert len({q.seed for q in engine.params}) == 4  # a seed per step


def test_sampling_keeps_temperature_and_top_p(fake_vllm):
    from qet.infer.vllm_sampler import VLLMSampler

    engine = Engine([([5], "stop")])
    VLLMSampler(engine, 5, DecodeConfig("sample", 0.7, 0.9, 16)).generate_batch([[0]], [2**64 - 1])
    p = engine.params[0]
    assert (p.temperature, p.top_p, p.top_k, p.min_p, p.repetition_penalty) == (0.7, 0.9, 0, 0.0, 1.0)
    assert 0 <= p.seed < 2**63

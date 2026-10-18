"""Batched decoding with vLLM, for evaluation at scale (D-026b; runs in the ~/venvs/qet-vllm environment).

It keeps the rules of `HFSampler`: the prompt is the same token ids (rendered and mapped by the same code); decoding
stops only at the end-of-turn token (`ignore_eos`, so the model's other end tokens do not stop it) or at the token
limit; sampling uses the given temperature and top-p with top-k off, min-p 0 and no penalties; each step has its own
seed, so an episode's outputs do not depend on the other episodes in its batch. Greedy is temperature 0. The log
probability is the raw one (before temperature and top-p). A LoRA adapter, a trained checkpoint, applies to every
request. Numerics differ from the Hugging Face loop (bf16 kernels, batching, the adapter in the model's dtype), so every
system in a comparison is evaluated with the same backend.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from qet.infer.decoding import DecodeConfig, Generation


def make_llm(model_id: str, revision: str, *, lora: bool, max_model_len: int = 16384, memory: float = 0.85,
             seed: int = 0) -> Any:  # fmt: skip
    """The engine for one backbone at its pinned revision, with prefix caching (prompts of one task share their
    system message and tool list) and room for one rank-64 adapter."""
    from vllm import LLM

    return LLM(model=model_id, revision=revision, tokenizer=model_id, tokenizer_revision=revision, dtype="bfloat16",
               enable_lora=lora, max_lora_rank=64, max_loras=1, enable_prefix_caching=True,
               max_model_len=max_model_len, gpu_memory_utilization=memory, seed=seed,
               logprobs_mode="raw_logprobs")  # fmt: skip


class VLLMSampler:
    """`generate_batch` decodes one action for each prompt; `generate` is the batch of one."""

    def __init__(self, llm: Any, eot_id: int, decode: DecodeConfig, lora_path: str | None = None):
        self.llm, self.eot_id, self.decode = llm, eot_id, decode
        self.lora = None
        if lora_path:
            from vllm.lora.request import LoRARequest

            self.lora = LoRARequest("checkpoint", 1, lora_path)

    def params(self, seed: int) -> Any:
        from vllm import SamplingParams

        d = self.decode
        greedy = d.mode == "greedy"
        return SamplingParams(temperature=0.0 if greedy else d.temperature, top_p=1.0 if greedy else d.top_p,
                              top_k=0, min_p=0.0, repetition_penalty=1.0, presence_penalty=0.0,
                              frequency_penalty=0.0, max_tokens=d.max_new_tokens, stop_token_ids=[self.eot_id],
                              ignore_eos=True, seed=seed % (1 << 63), logprobs=0, detokenize=False,
                              skip_special_tokens=False)  # fmt: skip

    def generate(self, prompt_ids: Sequence[int], seed: int) -> Generation:
        return self.generate_batch([prompt_ids], [seed])[0]

    def generate_batch(self, prompts: Sequence[Sequence[int]], seeds: Sequence[int]) -> list[Generation]:
        from vllm.inputs import TokensPrompt

        outs = self.llm.generate([TokensPrompt(prompt_token_ids=list(p)) for p in prompts],
                                 [self.params(s) for s in seeds], lora_request=self.lora, use_tqdm=False)  # fmt: skip
        gens = []
        for out in outs:
            c = out.outputs[0]
            ids = list(c.token_ids)
            ended = c.finish_reason == "stop"  # the only stop is the end-of-turn token
            if ended and (not ids or ids[-1] != self.eot_id):
                ids.append(self.eot_id)  # a backend that leaves the stop token out
            lp = c.cumulative_logprob
            gens.append(Generation(ids, ended, float(lp) if lp is not None and math.isfinite(lp) else -math.inf))
        return gens

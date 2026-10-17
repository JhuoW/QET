"""Tiny random-weight causal language models for CPU tests (float64 by default)."""

import torch
from transformers import LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM

MODEL_TYPES = ("qwen2", "llama")


def tiny_lm(model_type: str = "qwen2", *, vocab_size: int = 101, seed: int = 0, dtype=torch.float64, **overrides):
    """Two layers, hidden size 32, grouped-query attention, sdpa attention (eager computes its softmax in float32,
    which would cap the float64 tolerances); larger initial weights than the default so that the distributions are far
    from uniform. Returned in eval mode."""
    torch.manual_seed(seed)
    common = {
        "vocab_size": vocab_size,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 1024,
        "initializer_range": 0.3,
        "attn_implementation": "sdpa",
        **overrides,
    }
    if model_type == "qwen2":
        model = Qwen2ForCausalLM(Qwen2Config(**common, tie_word_embeddings=True))
    elif model_type == "llama":
        model = LlamaForCausalLM(LlamaConfig(**common, tie_word_embeddings=False))
    else:
        raise ValueError(model_type)
    return model.to(dtype).eval()

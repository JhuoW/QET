"""Memory sizing for LoRA training on a backbone (D-024e): the bytes autograd saves per token for backward, measured
on CPU with one decoder layer of the backbone's real sizes and LoRA exactly as `qet.train.trainer.add_lora` attaches
it, then scaled to the backbone's depth:

    python scripts/size_activations.py [--backbone Qwen/Qwen2.5-7B-Instruct] [--rank 64] [--max-tokens 8192 16384]

Model weights and [.., s, s] attention matrices are excluded (the GPU's SDPA kernels with a mask keep none). An
engineering estimate for choosing micro-batch sizes, not a result; the training log's peak memory checks it. Writes
nothing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def saved_bytes_per_token(backbone: str, rank: int, seq: int) -> tuple[float, object]:
    """Bytes saved for backward per token by one decoder layer, and the backbone's config."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    from qet.train.chat import backbone as pinned
    from qet.train.trainer import add_lora

    b = pinned(backbone)
    full = AutoConfig.from_pretrained(b.model_id, revision=b.revision)
    small = AutoConfig.from_pretrained(b.model_id, revision=b.revision, num_hidden_layers=1, vocab_size=1024)
    small._attn_implementation = "sdpa"
    model = add_lora(AutoModelForCausalLM.from_config(small, dtype=torch.bfloat16), rank=rank, alpha=2 * rank)
    params = {p.untyped_storage().data_ptr() for p in model.parameters()}
    seen: dict[int, int] = {}

    def pack(t: torch.Tensor) -> torch.Tensor:
        key = t.untyped_storage().data_ptr()
        square = t.dim() >= 2 and t.shape[-1] == seq and t.shape[-2] == seq
        if not square and key not in params and key not in seen:
            seen[key] = t.untyped_storage().nbytes()
        return t

    decoder = model.get_base_model().model
    embeds = decoder.embed_tokens(torch.randint(0, 1024, (1, seq))).detach()
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        decoder(inputs_embeds=embeds, use_cache=False)
    return sum(seen.values()) / seq, full


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backbone", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--seq", type=int, default=256, help="tokens in the measured forward pass")
    parser.add_argument("--max-tokens", type=int, nargs="+", default=[8192, 16384])
    args = parser.parse_args(argv)
    per_layer, cfg = saved_bytes_per_token(args.backbone, args.rank, args.seq)
    per_token = per_layer * cfg.num_hidden_layers
    gib = 2**30
    print(f"{args.backbone}: {per_layer / 1024:.1f} KiB per token per layer, {cfg.num_hidden_layers} layers, "
          f"{per_token / 2**20:.2f} MiB per token")  # fmt: skip
    for n in args.max_tokens:
        print(f"  {n} padded positions per micro-batch: {n * per_token / gib:.1f} GiB of saved activations")


if __name__ == "__main__":
    sys.path.insert(0, str(REPO / "src"))
    main()

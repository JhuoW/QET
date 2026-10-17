"""Download the pinned tokenizers of every backbone, and the weights of the backbones named with --weights, into
the Hugging Face cache (`make fetch`).

Run it once where there is network access (the login node); jobs then run with HF_HUB_OFFLINE=1. The Llama
repositories are gated: the account behind HF_TOKEN (or `hf auth login`) must already have been granted access.
"""

import argparse

from huggingface_hub import snapshot_download

from qet.train.chat import BACKBONES

TOKENIZER_FILES = [
    "config.json",
    "generation_config.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", nargs="*", default=[], metavar="MODEL_ID", help="also fetch these weights")
    args = parser.parse_args(argv)
    unknown = sorted(set(args.weights) - set(BACKBONES))
    if unknown:
        parser.error(f"unknown backbones {unknown}; supported: {sorted(BACKBONES)}")
    for model_id, b in sorted(BACKBONES.items()):
        patterns = TOKENIZER_FILES + (["*.safetensors", "*.safetensors.index.json"] if model_id in args.weights else [])
        path = snapshot_download(model_id, revision=b.revision, allow_patterns=patterns)
        print(f"{model_id}@{b.revision[:7]}{' (weights)' if model_id in args.weights else ''}: {path}")


if __name__ == "__main__":
    main()

"""The M1.5 warm-up configs and the M1.6 training configs (EVAL_PLAN §4, D-010, D-025d, D-030) from one template:

    python scripts/make_m1_configs.py warmups --steps L     # after configs/m1/warmup-measure.yaml has run
    python scripts/make_m1_configs.py train                 # after the warm-up runs: each run names its parent
    python scripts/make_m1_configs.py extras-warmups --steps L   # M1.7 extras (D-032c, D-033): the data curve and
    python scripts/make_m1_configs.py extras-train               # the injected false-independence ablation

Warm-up runs, one per renderer, backbone and seed: the quotient renderer (shared by configurations 2-6 and a2-a4) on
Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct, seeds 0-2; the chronological renderer (configuration 1) on Qwen, seeds
0-2; the native multi-turn renderer (a1) on Qwen, seed 0. Training runs, 33: configurations 1-6 x 3 seeds on Qwen;
2, 3, 6 x 3 seeds on Llama; a1, a2, a3 and a4 with K = 2, 4, 8 on Qwen, seed 0. Every training run gets the same budget
B and settings (matched compute, equal tuning: D-009's defaults, no grid) and evaluates nothing: evaluation runs
from the saved budget-fraction checkpoints later (M1.7). `train` takes each parent from the completed warm-up runs in
results/raw and refuses when one is missing or ambiguous.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "configs" / "m1"
COMPILE_RUN = "results/raw/M1-compile-20261018T023058Z"  # the frozen manifests (D-028, D-030)
BUDGET = 125_000_000  # D-030a
BACKBONES = {"qwen7b": "Qwen/Qwen2.5-7B-Instruct", "llama8b": "meta-llama/Llama-3.1-8B-Instruct"}
RENDERER_OF = {"1": "chrono", "a1": "native"}  # every other configuration renders with the quotient renderer
CONFIG_OF_RENDERER = {"quotient": "2", "chrono": "1", "native": "a1"}  # a configuration with that renderer
WARMUPS = [("quotient", b, s) for b in ("qwen7b", "llama8b") for s in (0, 1, 2)]
WARMUPS += [("chrono", "qwen7b", s) for s in (0, 1, 2)] + [("native", "qwen7b", 0)]
TRAIN = [(c, "qwen7b", s, None) for c in ("1", "2", "3", "4", "5", "6") for s in (0, 1, 2)]
TRAIN += [(c, "llama8b", s, None) for c in ("2", "3", "6") for s in (0, 1, 2)]
TRAIN += [("a1", "qwen7b", 0, None), ("a2", "qwen7b", 0, None), ("a3", "qwen7b", 0, None)]
TRAIN += [("a4", "qwen7b", 0, k) for k in (2, 4, 8)]
PASS1 = {"3": False, "1": False, "2": False, "a1": False, "a2": False}  # the rest score with pass 1
TRAIN_SETTINGS = {"lora_rank": 64, "lora_alpha": 128, "batch_graphs": 4, "lr": 1.0e-4, "warmup": 0.03,
                  "max_grad_norm": 1.0, "s_max": 256, "max_tokens": 8192, "max_target_tokens": 8192}  # fmt: skip
# M1.7 training extras (D-033), Qwen seed 0 in the quotient renderer: the data curve trains configurations 2 and 6 on
# the first 1,250 and 2,500 training graphs (round-robin over the cells, so every cell is covered); the ablation trains
# configuration 6 with injected false independences. Each variant has its own warm-up on the same data.
VARIANTS = {"n1250": {"train_count": 1250}, "n2500": {"train_count": 2500},
            "fi": {"inject": {"false_independence": 0.5, "seed": 0}}}  # fmt: skip
EXTRA_TRAIN = [("2", "n1250"), ("6", "n1250"), ("2", "n2500"), ("6", "n2500"), ("6", "fi")]
EVAL = {"decoding": [{"mode": "greedy"}, {"mode": "sample", "temperature": 1.0, "top_p": 1.0, "samples": 8}],
        "max_new_tokens": 512, "resume_prefix_tasks": 0}  # fmt: skip


def base(name: str, configuration: str, backbone: str, seed: int, variant: str = "") -> dict:
    v = VARIANTS.get(variant, {})
    data = {"milestone": "M1", "name": name, "configuration": configuration, "seed": seed,
            "backbone": BACKBONES[backbone], "dtype": "bfloat16",
            "data": {"env": "funcbenchgen", "kind": "grid",
                     "source": {"compile_run": COMPILE_RUN, "train": "train", "eval": "dev"},
                     "train": {"start": 0, "count": v.get("train_count", 5000)},
                     "test": {"start": 0, "count": 0}}}  # fmt: skip
    if "inject" in v:
        data["data"]["inject"] = v["inject"]
    return data


def write(path: Path, header: str, data: dict) -> None:
    text = "".join(f"# {line}\n" for line in header.splitlines()) + yaml.safe_dump(data, sort_keys=False)
    path.write_text(text, encoding="utf-8")
    print(path.relative_to(REPO))


def warmups(steps: int) -> None:
    for renderer, b, s in WARMUPS:
        data = base(f"warmup-{renderer}-{b}", CONFIG_OF_RENDERER[renderer], b, s)
        data["train"] = {"token_budget": BUDGET, "warmup_steps": steps, "warmup_only": True, "k": 4,
                         **TRAIN_SETTINGS, "checkpoint_fractions": [1.0]}  # fmt: skip
        data["eval"] = EVAL
        data["estimate"] = {"train_tokens_per_second": 7300, "eval_tokens_per_second": 1000}
        write(OUT / f"warmup-{renderer}-{b}-s{s}.yaml",
              f"M1.5 warm-up (D-010, D-025d): {steps} demo-order SFT steps in the {renderer} renderer, then the\n"
              "checkpoint that configuration runs start from. Generated by scripts/make_m1_configs.py.",
              data)  # fmt: skip


def parents(wanted: list[tuple]) -> dict[tuple, str]:
    """The completed warm-up run of each (renderer, backbone, seed) or (renderer, backbone, seed, variant) wanted."""
    found: dict[tuple, list[str]] = {}
    for m in sorted((REPO / "results" / "raw").glob("M1-warmup-*/manifest.json")):
        man = json.loads(m.read_text())
        name = man["config"]["path"].rsplit("/", 1)[-1].removesuffix(".yaml")  # warmup-<renderer>-<backbone>[-<v>]-s<N>
        if man.get("status") != "completed" or name.startswith("warmup-measure"):
            continue
        parts = name.split("-")
        key = (parts[1], parts[2], int(parts[-1][1:])) + (("-".join(parts[3:-1]),) if len(parts) > 4 else ())
        found.setdefault(key, []).append(f"results/raw/{man['run_id']}")
    missing = [w for w in wanted if len(found.get(w, [])) != 1]
    if missing:
        raise SystemExit(f"need exactly one completed warm-up run for each of {missing}: found {found}")
    return {w: found[w][0] for w in wanted}


def train() -> None:
    parent = parents(WARMUPS)
    for c, b, s, k in TRAIN:
        tag = f"a4k{k}" if k else (c if c.startswith("a") else f"c{c}")
        data = base(f"{tag}-{b}", c, b, s)
        data["train"] = {"token_budget": BUDGET, "parent": parent[(RENDERER_OF.get(c, "quotient"), b, s)],
                         "k": k or 4, **TRAIN_SETTINGS, "checkpoint_fractions": [0.25, 0.5, 0.75, 1.0]}  # fmt: skip
        data["eval"] = EVAL
        data["estimate"] = {"train_tokens_per_second": 7700 if not PASS1.get(c, True) else 10900,
                            "eval_tokens_per_second": 1000}  # fmt: skip
        write(OUT / f"train-{tag}-{b}-s{s}.yaml",
              f"M1.6 training run (EVAL_PLAN §4, D-030): configuration {c} on {BACKBONES[b]}, seed {s}, B = 125M\n"
              "tokens from its warm-up checkpoint; checkpoints at 25/50/75/100% of B; no evaluation here (M1.7).\n"
              "Generated by scripts/make_m1_configs.py.", data)  # fmt: skip


def extras_warmups(steps: int) -> None:
    for variant in VARIANTS:
        data = base(f"warmup-quotient-qwen7b-{variant}", "2", "qwen7b", 0, variant)
        data["train"] = {"token_budget": BUDGET, "warmup_steps": steps, "warmup_only": True, "k": 4,
                         **TRAIN_SETTINGS, "checkpoint_fractions": [1.0]}  # fmt: skip
        data["eval"] = EVAL
        data["estimate"] = {"train_tokens_per_second": 7300, "eval_tokens_per_second": 1000}
        write(OUT / f"warmup-quotient-qwen7b-{variant}-s0.yaml",
              f"M1.7 extra (D-033), {variant}: the {steps}-step warm-up (D-010) on the variant's training data.\n"
              "Generated by scripts/make_m1_configs.py.", data)  # fmt: skip


def extras_train() -> None:
    parent = parents([("quotient", "qwen7b", 0, v) for v in VARIANTS])
    for c, variant in EXTRA_TRAIN:
        data = base(f"c{c}-qwen7b-{variant}", c, "qwen7b", 0, variant)
        data["train"] = {"token_budget": BUDGET, "parent": parent[("quotient", "qwen7b", 0, variant)], "k": 4,
                         **TRAIN_SETTINGS, "checkpoint_fractions": [0.25, 0.5, 0.75, 1.0]}  # fmt: skip
        data["eval"] = EVAL
        data["estimate"] = {"train_tokens_per_second": 7400 if c == "2" else 11100, "eval_tokens_per_second": 1000}
        write(OUT / f"train-c{c}-qwen7b-{variant}-s0.yaml",
              f"M1.7 extra (D-033): configuration {c} on Qwen2.5-7B-Instruct, seed 0, variant {variant}, B = 125M\n"
              "tokens from the variant's warm-up; checkpoints at 25/50/75/100% of B. Generated by\n"
              "scripts/make_m1_configs.py.", data)  # fmt: skip


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="what", required=True)
    sub.add_parser("warmups").add_argument("--steps", type=int, required=True)
    sub.add_parser("train")
    sub.add_parser("extras-warmups").add_argument("--steps", type=int, required=True)
    sub.add_parser("extras-train")
    args = parser.parse_args(argv)
    {"warmups": lambda: warmups(args.steps), "train": train, "extras-warmups": lambda: extras_warmups(args.steps),
     "extras-train": extras_train}[args.what]()  # fmt: skip


if __name__ == "__main__":
    sys.path.insert(0, str(REPO / "src"))
    main()

"""Run configs (`configs/<milestone>/<name>.yaml`): one YAML file per run, validated before anything runs.

A config names the configuration (SPEC §8), the seed, the backbone, the data, the format warm-up (D-010), the training
settings (D-009, D-023), the evaluation protocol (SPEC §7, EVAL_PLAN §6-§7) and the throughput assumptions the dry run
uses for its GPU-hour estimate. Every field is explicit; unknown fields are errors.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from qet.train.chat import BACKBONES
from qet.train.methods import METHODS

DTYPES = ("bfloat16", "float32")
ENVS = ("toy",)


def _check_keys(section: str, data: dict[str, Any], required: set[str], optional: set[str]) -> None:
    if not isinstance(data, dict):
        raise ValueError(f"{section} must be a mapping")
    missing, unknown = required - set(data), set(data) - required - optional
    if missing:
        raise ValueError(f"{section}: missing {sorted(missing)}")
    if unknown:
        raise ValueError(f"{section}: unknown fields {sorted(unknown)}")


@dataclass(frozen=True)
class Split:
    start: int  # first task seed
    count: int


@dataclass(frozen=True)
class DataConfig:
    env: str
    kind: str  # toy: "functions" or "stateful"
    params: dict[str, Any]
    train: Split
    test: Split
    demo_seed: int = 0
    truncation: dict[str, Any] | None = None


@dataclass(frozen=True)
class TrainSettings:
    token_budget: int
    lora_rank: int = 64
    lora_alpha: int = 128
    batch_graphs: int = 8
    lr: float = 1e-4
    warmup: float = 0.03
    max_grad_norm: float | None = 1.0
    k: int | None = None
    s_max: int = 256
    max_tokens: int = 16384
    max_target_tokens: int | None = 8192
    checkpoint_fractions: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
    plus: bool = False
    warmup_steps: int = 0  # D-010 format warm-up: demo-order SFT steps in this configuration's renderer


@dataclass(frozen=True)
class EvalSettings:
    decoding: tuple[dict[str, Any], ...]  # e.g. {"mode": "greedy"}, {"mode": "sample", "samples": 8}
    max_new_tokens: int = 128
    resume_prefix_tasks: int = 0  # test tasks evaluated from legal prefixes (EVAL_PLAN §6); 0 = none
    eval_seed: int = 0  # seeds the Resume@prefix starts: the same for every training seed of a comparison


@dataclass(frozen=True)
class RunConfig:
    path: str
    sha256: str
    milestone: str
    name: str
    configuration: str
    seed: int
    backbone: str
    dtype: str
    data: DataConfig
    train: TrainSettings | None  # None: evaluation only (the base model)
    eval: EvalSettings
    estimate: dict[str, float] = field(default_factory=dict)  # assumed tokens per second until a pilot measures them

    @property
    def method(self):
        return METHODS[self.configuration]


def load_config(path: str | Path) -> RunConfig:
    raw = Path(path).read_bytes()
    data = yaml.safe_load(raw)
    _check_keys(
        "config",
        data,
        {"milestone", "name", "configuration", "seed", "backbone", "data", "eval"},
        {"dtype", "train", "estimate"},
    )
    if str(data["configuration"]) not in METHODS:
        raise ValueError(f"unknown configuration {data['configuration']!r}; expected one of {sorted(METHODS)}")
    if data["backbone"] not in BACKBONES:
        raise ValueError(f"unsupported backbone {data['backbone']!r}; supported: {sorted(BACKBONES)}")
    dtype = data.get("dtype", "bfloat16")
    if dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {DTYPES}")
    if not isinstance(data["seed"], int) or isinstance(data["seed"], bool):
        raise ValueError("seed must be an integer")
    for key in ("milestone", "name"):  # they name the run's folders
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", str(data[key])):
            raise ValueError(f"{key} may contain only letters, digits, '.', '_' and '-'")

    d = data["data"]
    _check_keys("data", d, {"env", "kind", "train", "test"}, {"params", "demo_seed", "truncation"})
    if d["env"] not in ENVS:
        raise ValueError(f"unknown env {d['env']!r}; expected one of {ENVS}")
    splits = {}
    for name in ("train", "test"):
        _check_keys(f"data.{name}", d[name], {"start", "count"}, set())
        splits[name] = Split(int(d[name]["start"]), int(d[name]["count"]))
    a, b = splits["train"], splits["test"]
    if a.start < b.start + b.count and b.start < a.start + a.count:
        raise ValueError("training and test seeds overlap")  # EVAL_PLAN §3.1: training seeds disjoint from test
    data_cfg = DataConfig(d["env"], d["kind"], dict(d.get("params") or {}), a, b, int(d.get("demo_seed", 0)),
                          d.get("truncation"))  # fmt: skip

    train = None
    if data.get("train") is not None:
        t = dict(data["train"])
        fields = set(TrainSettings.__dataclass_fields__)
        _check_keys("train", t, {"token_budget"}, fields - {"token_budget"})
        if "checkpoint_fractions" in t:
            t["checkpoint_fractions"] = tuple(float(x) for x in t["checkpoint_fractions"])
        train = TrainSettings(**t)
        if train.token_budget <= 0 or not all(0 < f <= 1 for f in train.checkpoint_fractions):
            raise ValueError("train: token_budget must be positive and checkpoint fractions in (0, 1]")

    e = data["eval"]
    _check_keys("eval", e, {"decoding"}, {"max_new_tokens", "resume_prefix_tasks", "eval_seed"})
    decoding = tuple(dict(x) for x in e["decoding"])
    if not decoding:
        raise ValueError("eval.decoding needs at least one setting")
    for x in decoding:
        _check_keys("eval.decoding", x, {"mode"}, {"temperature", "top_p", "samples"})
        if x["mode"] not in ("greedy", "sample"):
            raise ValueError(f"unknown decoding mode {x['mode']!r}")
        samples = x.get("samples", 1 if x["mode"] == "greedy" else 8)
        if not isinstance(samples, int) or samples < 1 or (x["mode"] == "greedy" and samples != 1):
            raise ValueError("greedy decoding has one rollout; sampling a positive number")
        if not float(x.get("temperature", 1.0)) > 0 or not 0 < float(x.get("top_p", 1.0)) <= 1:
            raise ValueError("decoding needs temperature > 0 and 0 < top_p <= 1")
    eval_cfg = EvalSettings(decoding, int(e.get("max_new_tokens", 128)), int(e.get("resume_prefix_tasks", 0)),
                            int(e.get("eval_seed", 0)))  # fmt: skip
    return RunConfig(
        path=str(path),
        sha256=hashlib.sha256(raw).hexdigest(),
        milestone=str(data["milestone"]),
        name=str(data["name"]),
        configuration=str(data["configuration"]),
        seed=data["seed"],
        backbone=data["backbone"],
        dtype=dtype,
        data=data_cfg,
        train=train,
        eval=eval_cfg,
        estimate={k: float(v) for k, v in (data.get("estimate") or {}).items()},
    )

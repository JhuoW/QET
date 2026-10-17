"""Run manifests (EVAL_PLAN §10): `results/raw/<run_id>/manifest.json`, written by code when a run starts and completed
when it ends. Reported runs must come from a clean working tree; the manifest records the commit and any dirty files.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import platform
import socket
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
RUNLOG = "docs/RUNLOG.md"  # rewritten by every run, so not counted as dirt

FIELDS = (  # EVAL_PLAN §10, in that order
    "run_id",
    "created_at",
    "git",
    "config",
    "seed",
    "configuration",
    "lambda",
    "s_max",
    "k",
    "backbone",
    "tokenizer",
    "data_manifest",
    "benchmark",
    "judge",
    "decoding",
    "budgets",
    "hardware",
    "libraries",
    "tokens",
    "gpu_hours",
    "parent_checkpoint",
    "status",
)


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.UTC).replace(microsecond=0)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def git_state(repo: Path = REPO) -> dict[str, Any]:
    """The commit and the tracked files that differ from it (the run log excepted)."""

    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True).stdout

    dirty = sorted(line[3:] for line in git("status", "--porcelain", "--untracked-files=no").splitlines())
    dirty = [f for f in dirty if f != RUNLOG]
    return {"commit": git("rev-parse", "HEAD").strip(), "dirty": bool(dirty), "dirty_files": dirty}


def is_committed(path: str | Path, repo: Path = REPO) -> bool:
    """Whether the file is tracked and unchanged since the last commit."""
    rel = os.path.relpath(Path(path).resolve(), repo)
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=repo, capture_output=True).returncode == 0
    changed = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=repo).returncode != 0
    return tracked and not changed


def hardware() -> dict[str, Any]:
    import torch

    gpus = (
        [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
    )
    return {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "cpus": os.cpu_count(),
        "gpus": gpus,
        "cuda": torch.version.cuda,
        "scheduler_job": os.environ.get("PBS_JOBID"),
    }


def libraries() -> dict[str, str | None]:
    out: dict[str, str | None] = {"python": platform.python_version()}
    for lib in ("qet", "torch", "transformers", "tokenizers", "peft", "accelerate", "numpy", "pyyaml", "jinja2"):
        try:
            out[lib] = version(lib)
        except PackageNotFoundError:
            out[lib] = None
    return out


class Manifest:
    """The manifest file of one run; every update rewrites it atomically."""

    def __init__(self, path: str | Path, data: dict[str, Any]):
        missing = [f for f in FIELDS if f not in data]
        if missing:
            raise ValueError(f"manifest is missing EVAL_PLAN §10 fields {missing}")
        self.path = Path(path)
        self.data = data
        self._write()

    def update(self, **fields: Any) -> None:
        self.data.update(fields)
        self._write()

    def _write(self) -> None:
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.data, indent=2, ensure_ascii=False, sort_keys=False) + "\n", encoding="utf-8")
        tmp.replace(self.path)


def read_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))

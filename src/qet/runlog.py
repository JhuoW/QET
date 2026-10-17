"""The run log `docs/RUNLOG.md`: one row per run, appended by code when a run starts and updated when it ends. Results
never go here; they live in `results/raw/<run_id>/` and tables come from `scripts/`."""

from __future__ import annotations

import fcntl
import os
import stat
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

COLUMNS = ("run_id", "submitted", "milestone", "config", "method", "backbone", "seed", "commit", "status", "GPU-h",
           "path", "note")  # fmt: skip
HEADER = "| " + " | ".join(COLUMNS) + " |"


def _cell(value: Any) -> str:
    text = "" if value is None else str(value)
    return " ".join(text.splitlines()).replace("|", "\\|")  # every line separator str.splitlines() knows


def _line(row: dict[str, Any]) -> str:
    unknown = set(row) - set(COLUMNS)
    if unknown:
        raise ValueError(f"unknown run log columns {sorted(unknown)}")
    return "| " + " | ".join(_cell(row.get(c)) for c in COLUMNS) + " |"


def _parse(line: str) -> list[str]:
    """The cells of a table row, unescaped (the inverse of `_line`)."""
    return [c.strip().replace("\\|", "|") for c in line.strip()[1:-1].split(" | ")]


def _lines(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if HEADER not in lines:
        raise ValueError(f"{path} has no run log table")
    return lines


def _write(path: Path, lines: list[str]) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")  # unique per writer
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    os.chmod(tmp, stat.S_IMODE(path.stat().st_mode))  # mkstemp makes 0600; keep the log's own permissions
    os.replace(tmp, path)


@contextmanager
def _locked(path: Path):
    """One writer at a time: runs that end together must not lose each other's rows."""
    with open(path.with_name(f".{path.name}.lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def append_row(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    with _locked(path):
        _append(path, row)


def _append(path: Path, row: dict[str, Any]) -> None:
    lines = _lines(path)
    if any(line.startswith(f"| {_cell(row['run_id'])} |") for line in lines):
        raise ValueError(f"run {row['run_id']} is already in the run log")
    lines.append(_line(row))
    _write(path, lines)


def update_row(path: str | Path, run_id: str, **fields: Any) -> None:
    """Change some cells of the run's row (status, GPU-h, note)."""
    path = Path(path)
    with _locked(path):
        _update(path, run_id, fields)


def _update(path: Path, run_id: str, fields: dict[str, Any]) -> None:
    lines = _lines(path)
    prefix = f"| {_cell(run_id)} |"
    hits = [i for i, line in enumerate(lines) if line.startswith(prefix)]
    if len(hits) != 1:
        raise ValueError(f"run {run_id} appears {len(hits)} times in the run log")
    cells = _parse(lines[hits[0]])
    row = dict(zip(COLUMNS, cells, strict=True))
    row.update(fields)
    lines[hits[0]] = _line(row)
    _write(path, lines)


def read_rows(path: str | Path) -> list[dict[str, str]]:
    lines = _lines(Path(path))
    start = lines.index(HEADER) + 2  # skip the header and the separator
    rows = []
    for line in lines[start:]:
        if line.startswith("| "):
            cells = _parse(line)
            rows.append(dict(zip(COLUMNS, cells, strict=True)))
    return rows

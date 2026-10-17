"""Tables from results/raw/ and manifests: python scripts/make_tables.py --runs <run_id ...> [--out PATH] [--kind K]

Every number is read from `results/raw/<run_id>/manifest.json` and `metrics.json` and printed as stored (rounded only
for display); nothing is typed by hand. The table names its comparison type (EVAL_PLAN §10: historical, fresh,
causal; smoke tables support no claims) and ends with the sha256 of every file it read, for the results-auditor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "results" / "raw"
KINDS = {
    "smoke": "Smoke run (E1.7): checks the pipeline end to end; no claims are drawn from these numbers.",
    "causal": "Causal comparison: our matched configurations (EVAL_PLAN §10).",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rel(path: Path) -> str:
    return str(path.relative_to(REPO)) if path.resolve().is_relative_to(REPO) else str(path)


def _fmt(x: object, digits: int = 3) -> str:
    if x is None:
        return "–"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def run_row(run_id: str, raw: Path = RAW) -> tuple[list[str], list[tuple[str, str]]]:
    folder = raw / run_id
    mpath, xpath = folder / "manifest.json", folder / "metrics.json"
    manifest, metrics = json.loads(mpath.read_text()), json.loads(xpath.read_text())
    std = metrics["standard"]
    greedy = std.get("greedy", {})
    sampled = next((v for k, v in sorted(std.items()) if k.startswith("sample")), {})
    episodes = sum(sum(v["ends"].values()) for v in std.values())
    ends: dict[str, int] = {}
    for v in std.values():
        for end, count in v["ends"].items():
            ends[end] = ends.get(end, 0) + count
    resume = metrics.get("resume_prefix") or {}
    by_cond = resume.get("by_condition", {})
    tokens = manifest.get("tokens") or {}
    cfg = manifest["configuration"]
    row = [
        run_id,
        f"{cfg['id']} {cfg['name']}",
        manifest["backbone"]["id"].split("/")[-1],
        str(manifest["seed"]),
        manifest["status"],
        _fmt(greedy.get("tasks")),
        _fmt(greedy.get("success")),
        _fmt(sampled.get("success")),
        _fmt(sampled.get("pass^4")),
        _fmt(ends.get("invalid_action", 0) / episodes if episodes else None),
        _fmt(ends.get("budget_exhausted", 0) / episodes if episodes else None),
        _fmt(by_cond.get("on_path", {}).get("success")),
        _fmt(by_cond.get("off_path", {}).get("success")),
        _fmt(resume.get("off_path_gap")),
        _fmt(tokens.get("scored")),
        _fmt(tokens.get("gradient")),
        _fmt(manifest.get("gpu_hours")),
        manifest["git"]["commit"][:7] + (" dirty" if manifest["git"]["dirty"] else ""),
    ]
    return row, [(_rel(mpath), _sha(mpath)), (_rel(xpath), _sha(xpath))]


HEADER = ["run", "configuration", "backbone", "seed", "status", "tasks", "greedy success", "sampled success",
          "pass^4", "invalid share", "budget-exhausted share", "resume on-path", "resume off-path", "off-path gap",
          "scored tokens", "gradient tokens", "GPU-h", "commit"]  # fmt: skip


def make_table(run_ids: list[str], kind: str, raw: Path = RAW) -> str:
    rows, sources = [], []
    for run_id in run_ids:
        row, src = run_row(run_id, raw)
        rows.append(row)
        sources += src
    lines = [KINDS[kind], "", "| " + " | ".join(HEADER) + " |", "| " + " | ".join("---" for _ in HEADER) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    lines += ["", "Sources (sha256):"] + [f"- `{path}` {digest}" for path, digest in sources]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--kind", choices=sorted(KINDS), default="causal")
    parser.add_argument("--out", type=Path, help="also write the table here (results/tables/...)")
    args = parser.parse_args(argv)
    table = make_table(args.runs, args.kind)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(table, encoding="utf-8")
    print(table, end="")


if __name__ == "__main__":
    main()

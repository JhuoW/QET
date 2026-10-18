"""Submit a list of jobs as PBS dependency chains, so that at most N GPUs are in use at once (D-034, N = 2):

    python scripts/pbs/chains.py PLAN [--chains 2] [--dry]

PLAN is a committed text file, one job per line (blank lines and # comments skipped):
    config <configs/...yaml> <HH:MM:SS>         python -m qet.run on one config
    batch  <configs/.../batch.txt> <HH:MM:SS>   several configs in one job (an evaluation batch, vLLM environment)
Line i goes to chain i mod N; each job waits for the previous job of its chain (depend=afterany: a chain goes on past
a failed job). Every submission goes through scripts/pbs/submit.sh (pinned checkout, the GPU guard). Writes
logs/pbs/chains-<stamp>.json with the plan and the job ids. --dry prints the submissions without making them.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
VLLM_PYTHON = str(Path.home() / "venvs" / "qet-vllm" / "bin" / "python")


def parse(plan: Path) -> list[tuple[str, str, str]]:
    jobs = []
    for line in plan.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        kind, path, walltime = line.split()
        if kind not in ("config", "batch") or not re.fullmatch(r"\d+:\d\d:\d\d", walltime):
            raise SystemExit(f"bad plan line: {line!r}")
        if not (REPO / path).exists():
            raise SystemExit(f"{path} does not exist")
        jobs.append((kind, path, walltime))
    return jobs


def command(kind: str, path: str, walltime: str, after: str | None) -> list[str]:
    cmd = [str(REPO / "scripts" / "pbs" / "submit.sh")]
    if kind == "config":
        cmd += ["--config", path, "--walltime", walltime]
    else:  # evaluation batches run vLLM
        cmd += ["--batch", path, "--python", VLLM_PYTHON, "--walltime", walltime]
    return cmd + (["--after", after] if after else [])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args(argv)
    jobs = parse(args.plan if args.plan.is_absolute() else REPO / args.plan)
    last: list[str | None] = [None] * args.chains
    submitted = []
    for i, (kind, path, walltime) in enumerate(jobs):
        chain = i % args.chains
        after = last[chain]
        cmd = command(kind, path, walltime, after)
        if args.dry:
            print(f"chain {chain}: {' '.join(cmd)}")
            last[chain] = f"<job {i}>"
            continue
        out = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        ids = re.findall(r"^(\d+\.\S+)$", out.stdout, re.M)
        if out.returncode != 0 or not ids:
            raise SystemExit(f"submission {i} ({path}) failed:\n{out.stdout}\n{out.stderr}")
        last[chain] = ids[-1]
        submitted.append({"chain": chain, "kind": kind, "path": path, "walltime": walltime, "after": after,
                          "job": ids[-1]})  # fmt: skip
        print(f"chain {chain}: {path} -> {ids[-1]}" + (f" (after {after})" if after else ""), flush=True)
    if not args.dry:
        stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
        record = REPO / "logs" / "pbs" / f"chains-{stamp}.json"
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text(json.dumps({"plan": str(args.plan), "chains": args.chains, "jobs": submitted}, indent=2))
        print(record)


if __name__ == "__main__":
    sys.exit(main())

"""`make smoke` (MILESTONES E1.7): configurations 2 and 6 on the toy environment with Qwen2.5-0.5B-Instruct, each run
trained briefly and evaluated (greedy, 8 samples, Resume@prefix), then one table from scripts/make_tables.py. Needs one
GPU; on the login node submit `qsub -o logs/pbs/ scripts/pbs/smoke.pbs` instead. No claims are drawn from smoke numbers.
"""

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

CONFIGS = ["configs/smoke/sft.yaml", "configs/smoke/qet.yaml"]


def main() -> None:
    import torch
    from make_tables import make_table

    if not torch.cuda.is_available():
        raise SystemExit("make smoke needs a GPU: qsub -o logs/pbs/ scripts/pbs/smoke.pbs")
    run_ids = []
    for c in CONFIGS:  # one process per run: each imports the code of the commit it records (D-025b)
        done = subprocess.run([sys.executable, "-m", "qet.run", "--config", str(REPO / c)], cwd=REPO, check=True,
                              stdout=subprocess.PIPE, text=True)  # fmt: skip
        run_ids.append(Path(done.stdout.strip().splitlines()[-1]).name)
    out = REPO / "results" / "tables" / f"smoke-{run_ids[-1].rsplit('-', 1)[-1]}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    table = make_table(run_ids, "smoke")
    out.write_text(table, encoding="utf-8")
    print(table, end="")
    print(f"table: {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()

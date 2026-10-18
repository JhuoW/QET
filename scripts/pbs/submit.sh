#!/bin/bash
# Submit one run (python -m qet.run) from a pinned checkout of HEAD (D-025b), so that commits made while the job waits
# or runs cannot change its code. From the repository root, after the dry run:
#   scripts/pbs/submit.sh configs/<milestone>/<name>.yaml [HH:MM:SS walltime, default 01:00:00]
# The checkout is ~/qet-worktrees/<commit> with FuncBenchGen at its pinned commit (cloned from this repository's
# submodule, no network); results/raw, checkpoints and data are symlinks to this repository's folders, and the run's
# RUNLOG row goes to this repository's docs/RUNLOG.md. Walltimes above 1 h route to gpu_ppu_normal.
set -euo pipefail
main=$(git rev-parse --show-toplevel)
cd "$main"
config=${1:?usage: scripts/pbs/submit.sh CONFIG [WALLTIME]}
walltime=${2:-01:00:00}
if git status --porcelain --untracked-files=no | grep -qv ' docs/RUNLOG.md$'; then
  echo "uncommitted changes: commit them first (the job runs the committed tree)"; exit 2
fi
git ls-files --error-unmatch "$config" >/dev/null
commit=$(git rev-parse HEAD)
wt="$HOME/qet-worktrees/$commit"
if [ ! -d "$wt" ]; then
  git worktree add --detach "$wt" "$commit"
  git -C "$wt" -c protocol.file.allow=always -c "submodule.third_party/FuncBenchGen.url=$main/third_party/FuncBenchGen" \
    submodule update --init third_party/FuncBenchGen
fi
for d in results/raw checkpoints data logs; do
  mkdir -p "$main/$d" "$(dirname "$wt/$d")"
  ln -sfn "$main/$d" "$wt/$d"
done
if [ -n "$(git -C "$wt" status --porcelain)" ]; then echo "the pinned checkout $wt is not clean"; exit 2; fi
qsub -v CONFIG="$config",WORKTREE="$wt",MAIN="$main" -l walltime="$walltime" -o "$main/logs/pbs/" "$main/scripts/pbs/run.pbs"

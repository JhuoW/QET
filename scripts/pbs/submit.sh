#!/bin/bash
# Submit one run (python -m qet.run) from a pinned checkout of HEAD (D-025b), so that commits made while the job waits
# or runs cannot change its code. From the repository root, after the dry run:
#   scripts/pbs/submit.sh configs/<milestone>/<name>.yaml [HH:MM:SS walltime, default 01:00:00]
# or a script that writes its own run folder and takes --runlog (e.g. scripts/vllm_check.py):
#   scripts/pbs/submit.sh --script SCRIPT [--args "ARGS"] [--python PYTHON] [--walltime HH:MM:SS] [--after JOB]
# or the option form of one config (needed with --after):
#   scripts/pbs/submit.sh --config CONFIG [--walltime HH:MM:SS] [--after JOB]
# or several configs in one job, one after another (an evaluation batch sharing one vLLM engine):
#   scripts/pbs/submit.sh --batch FILE [--python PYTHON] [--walltime HH:MM:SS] [--after JOB]
# GPU limit (the human, 2026-10-18, D-034): at most 2 jobs running or queued at once. --after JOB queues the job
# behind JOB (PBS depend=afterany), so two chains keep longer lists within the limit.
# The checkout is ~/qet-worktrees/<commit> with FuncBenchGen at its pinned commit (cloned from this repository's
# submodule, no network); results/raw, checkpoints and data are symlinks to this repository's folders, and the run's
# RUNLOG row goes to this repository's docs/RUNLOG.md. Walltimes above 1 h route to gpu_ppu_normal.
set -euo pipefail
main=$(git rev-parse --show-toplevel)
cd "$main"
config="" script="" args="" batch="" after="" python="python" walltime=01:00:00
if [ "${1:-}" = "--script" ] || [ "${1:-}" = "--batch" ] || [ "${1:-}" = "--config" ]; then
  while [ $# -gt 0 ]; do
    case "$1" in
      --config) config=$2; shift 2 ;;
      --script) script=$2; shift 2 ;;
      --batch) batch=$2; shift 2 ;;
      --args) args=$2; shift 2 ;;
      --python) python=$2; shift 2 ;;
      --walltime) walltime=$2; shift 2 ;;
      --after) after=$2; shift 2 ;;
      *) echo "unknown option $1"; exit 2 ;;
    esac
  done
  if [ -n "$script" ]; then git ls-files --error-unmatch "$script" >/dev/null; fi
  if [ -n "$batch" ]; then
    git ls-files --error-unmatch "$batch" >/dev/null
    while read -r c; do [ -n "$c" ] && git ls-files --error-unmatch "$c" >/dev/null; done < "$batch"
  fi
else
  config=${1:?usage: scripts/pbs/submit.sh CONFIG [WALLTIME]}
  walltime=${2:-01:00:00}
fi
if git status --porcelain --untracked-files=no | grep -qv ' docs/RUNLOG.md$'; then
  echo "uncommitted changes: commit them first (the job runs the committed tree)"; exit 2
fi
if [ -n "$config" ]; then git ls-files --error-unmatch "$config" >/dev/null; fi
max=${QET_MAX_GPUS:-2}  # the human's limit (D-034); raise it only when the human does
active=$(qstat -u "$USER" 2>/dev/null | awk 'NR > 5 && ($10 == "R" || $10 == "Q") { n++ } END { print n + 0 }')
if { [ -z "$after" ] && [ $((active + 1)) -gt "$max" ]; } || { [ -n "$after" ] && [ "$active" -gt "$max" ]; }; then
  echo "refused: $active jobs running or queued; the limit is $max GPUs at once (D-034)"; exit 3
fi
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
depend=()
if [ -n "$after" ]; then depend=(-W "depend=afterany:$after"); fi
qsub "${depend[@]}" -v "CONFIG=$config,SCRIPT=$script,ARGS=$args,BATCH=$batch,QET_PYTHON=$python,WORKTREE=$wt,MAIN=$main" \
  -l walltime="$walltime" -o "$main/logs/pbs/" "$main/scripts/pbs/run.pbs"

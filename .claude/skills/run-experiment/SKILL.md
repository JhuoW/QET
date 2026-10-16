---
name: run-experiment
description: Launch one training or evaluation run from a committed config with a dry run, cost estimate, human approval when required, manifest and run log entry.
disable-model-invocation: true
argument-hint: "<configs/.../name.yaml>"
---
Config: $ARGUMENTS

1. Preconditions: the config is committed and the working tree is clean; the run is in the EVAL_PLAN §4 run
   matrix or is a smoke / pilot run named in MILESTONES.md; seed, configuration id, backbone, token budget and
   data manifest are set; frozen manifests match the hashes in docs/DECISIONS.md.
2. `python -m qet.run --config $ARGUMENTS --dry-run` and report the plan, scored and gradient tokens, estimated
   GPU-hours and API cost.
3. Stop and ask the human, showing the estimate, if the run exceeds the GPU threshold in AGENTS.md
   "Environment", goes through the scheduler, uses a paid API, or needs a gated download.
4. Launch as the Environment section says. Record the job id; the runner writes the manifest and the RUNLOG row.
5. Do not wait in a loop. When asked later: check the job, confirm the manifest is complete with status
   `completed`, run the evaluation named in the config, and regenerate tables with `scripts/make_tables.py`.
6. Before reporting any number from this run, ask the results-auditor subagent to trace it.

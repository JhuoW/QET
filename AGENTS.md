# AGENTS.md — QET: implementation and experiments

This repository implements QET (Quotient Execution Training) for tool-using LLM agents and runs its
evaluation plan to produce auditable, paper-ready results. These instructions apply to every coding agent.
Claude Code reads them through CLAUDE.md.

## Sources of truth

| File | Content | Who changes it |
| --- | --- | --- |
| `docs/SPEC.md` | The method exactly as it must be implemented (normative) | Human only |
| `docs/EVAL_PLAN.md` | Benchmarks, configurations, protocols, statistics, pre-registered criteria | Human only |
| `docs/MILESTONES.md` | Ordered tasks, acceptance checks, human gates | Agent ticks boxes with evidence |
| `docs/DECISIONS.md` | Decision log and open questions | Agent proposes, human accepts |
| `docs/RUNLOG.md` | One row per run, appended by `qet.runlog` | Code; agent adds notes |
| `reference/` | Oracle, golden fixtures, acceptance tests | Nobody (human regenerates after a decision) |

Before coding, read the SPEC / EVAL_PLAN sections your task cites. If SPEC.md seems wrong or silent, write a
Proposed entry or open question in DECISIONS.md; do not improvise the method.

## Integrity rules (never break these)
1. Never invent, estimate, interpolate or "fill in" a number. Every number in a table, figure, report or message
   comes from a script that reads `results/raw/` and its manifest. If a run has not happened, say so.
2. Report negative and null results like positive ones. Never drop seeds, runs, tasks or checkpoints after
   seeing results.
3. Never tune on test data: select checkpoints and hyperparameters on dev only. AppWorld test: aggregate scores
   only, and never write per-task test traces anywhere that leaves this machine.
4. The falsification criteria (EVAL_PLAN §9) and frozen manifests are pre-registered: never change a metric,
   split, comparison or criterion after results exist.
5. Benchmark code and official evaluators live in `third_party/`, pinned by commit, and are never modified.
6. Never weaken a test to make it pass (skip, xfail, looser tolerance, deleted case). Never edit `reference/`.
7. Never claim something works without having run the check that shows it.

## Method invariants (SPEC.md; tests enforce them)
- One Reduce + Render code path serves training and inference. The prompt depends only on (task, H): byte-
  identical for every legal order of the same completed set; no step numbers, timestamps, dates or turn
  counters; full re-render each step; content-based truncation; no reasoning text carried across steps.
- log π(action) = sum of token log-probs of the action string plus the end-of-turn token. No length
  normalization. No renormalization over legal actions (configuration 4's renormalized π̄ lives only inside its
  weights).
- Lattice recursion in log space and float64; exact up to S_max states, otherwise the sub-lattice (SPEC §6.6).
- A training step is: pass 1 without gradients in eval mode → recursion → weights; pass 2 weighted SFT with the
  weights detached; no parameter update in between. Log the true objective, never the pass-2 loss value.
- Configurations 2–6 use one training loop and differ only in edge weights and start states (SPEC §8);
  configuration 1 differs from configuration 2 only in its renderer.
- Compile is conservative and never reads a benchmark's ground-truth DAG (audits only).
- Matched compute: count scored and gradient tokens for every configuration, QET's scoring pass included.

## Repository layout (built in E0; extend, don't rename)
```
src/qet/core/     canonical.py contracts.py compile.py reduce.py render.py lattice.py weights.py api.py
src/qet/train/    chat.py scoring.py methods.py trainer.py
src/qet/infer/    agent.py decoding.py
src/qet/envs/     toy.py funcbenchgen.py toolhop.py appworld.py stabletoolbench.py bfcl.py tinyagent.py nestful.py
src/qet/eval/     runner.py resume_prefix.py metrics.py stats.py
src/qet/          run.py manifest.py runlog.py
configs/          one YAML per run (configs/<milestone>/<name>.yaml)
scripts/          make_tables.py make_figures.py gate_report.py
tests/            unit, property and integration tests
third_party/      pinned benchmark repos (read-only)
reference/        oracle and fixtures (read-only)
results/raw/      written only by code, one folder per run_id (never edited by hand; not committed)
results/tables/  results/figures/  docs/reports/
```

## Commands (E0 creates the Makefile with exactly these targets)
```
make setup        # install the package and dev tools
make lint         # ruff check + ruff format --check
make test-fast    # CPU only, < 2 min: pytest tests reference -m "not gpu and not slow"
make test         # everything that runs on this machine
make smoke        # E1: tiny end-to-end run on 1 GPU, < 30 min
python -m qet.run --config configs/<m>/<name>.yaml [--dry-run] [--limit N]
python scripts/make_tables.py --runs <run_id ...>
python scripts/gate_report.py --gate A|B|appworld
```

## How to work
1. Take the first unchecked task in `docs/MILESTONES.md` whose prerequisites are done. One task at a time; never
   start a milestone whose gate is not passed.
2. Plan before editing: files to touch, tests to write, the exact acceptance command.
3. For core math, write the test first; fixtures and `reference/qet_ref.py` are the ground truth (tests only —
   production code never imports the oracle).
4. Small steps; `make lint test-fast` before every commit; commit the work as `<task-id>: <summary>`.
5. Then tick the box with `evidence: <command> → <result>; commit <hash of that commit>` under it and commit the
   tick as `<task-id>: evidence`. No evidence, no tick.
6. Record any choice the spec leaves open as a Proposed decision. Proceed on its default only if cheap to reverse.
7. If the same failure survives two honest attempts, stop and write down what you tried and what you suspect.
8. Keep state in files (MILESTONES, DECISIONS, RUNLOG, manifests), not in the conversation.

## Running experiments
- Every run starts from a committed config and a clean working tree; `--dry-run` first (plan, token and
  GPU-hour estimate, nothing written).
- A run writes `results/raw/<run_id>/manifest.json` (fields: EVAL_PLAN §10), logs and predictions, and appends
  to RUNLOG. Code writes these files itself (no shell redirection into `results/raw/`).
- Long jobs go through the scheduler; never keep a session busy with sleep loops. Check status when asked.
- A result is reportable only after a script regenerates it from `results/raw/` and the numbers have been
  traced by an independent check (the results-auditor in Claude Code, or a fresh-context review elsewhere).

## Stop and ask the human before
- any job above the GPU threshold below, any multi-node job, any scheduler submission not covered by a batch
  approval recorded in DECISIONS.md (for example "M1 run matrix, ≤ N GPU-hours");
- any paid API call beyond the approved budget (judges, LLM-drafted contracts, strong-model demonstrations);
- downloading gated models or data, or accepting a license (Llama, AppWorld data, ToolBench / DTA-Tool);
- changing SPEC.md, EVAL_PLAN.md, `reference/`, a frozen manifest, or a pre-registered criterion;
- any gate (E0 review, Gate A, AppWorld feasibility, Gate B): produce the report, then stop;
- deleting or overwriting results or checkpoints, rewriting git history, pushing, or publishing anything.

## Environment (the human fills this in before E1)
Filled in by the agent on 2026-10-17 from the machine itself (`qstat -Qf`, `pbsnodes`, `~/GPU_Usage/`) after the
human said to proceed without their checks. The budget lines are the agent's interim policy (D-021), not a budget
set by the human.
- Compute: NTU GaaS, PBS (`qsub`). The login node `hpc-gaas-hn2` has no GPU; GPU work runs only in PBS jobs. GPU
  nodes: 8 × NVIDIA H200 (141 GB), 100 CPUs, 2 TB RAM. Queue `gpu_ppu` (pay-per-use; routes to `gpu_ppu_high`,
  walltime ≤ 1 h, and `gpu_ppu_normal`, ≤ 336 h), project `ccds_siqiang.luo`; the free queues are not open to this
  account. Max concurrent jobs: not set (the agent runs one at a time).
- Ask before any job above: 1 GPU-hour, or any job that takes E1 past 10 GPU-hours in total (interim, D-021);
  budget per milestone: not set by the human (M1 needs one before the M1.4 pilot).
- API budget: none, so no paid API calls; keys only from environment variables (`OPENAI_API_KEY`, `HF_TOKEN`); never
  write keys to files, configs, logs or manifests.
- Paths: data `data/`, Hugging Face cache `~/.cache/huggingface` (models are downloaded on the login node; jobs run
  with `HF_HUB_OFFLINE=1`), checkpoints `checkpoints/`, results `results/raw/` (all git-ignored).
- Toolchain: Python 3.12.15 (`~/anaconda3` base environment, validated on an H200 by the human's job 87357; the code
  targets 3.11+), CUDA 12.8 from the PyTorch 2.8.0 wheels (driver 580), package manager pip with
  `-c ~/GPU_Usage/llm-base/constraints.txt`; PBS template `~/GPU_Usage/llm-base/validate.pbs`.

## Conventions
Python 3.11, type hints, ruff, pytest (markers `gpu`, `slow`); dataclasses for records; JSON / JSONL for data
and logs; YAML configs; explicit seeds everywhere; deterministic ordering (sort before iterating any set or
mapping that feeds a prompt, a manifest or a table); no notebooks as a source of results.

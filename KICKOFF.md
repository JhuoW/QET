# Kickoff — how to drive a coding agent through QET

This kit turns the QET method and its evaluation plan into a repository a coding agent can work in safely.
It is written for Claude Code; any agent that reads `AGENTS.md` (Codex, Cursor, Gemini CLI, …) gets the same
rules, and `.claude/` only adds Claude Code automation.

## How the kit steers the agent

| Layer | Files | Loaded |
| --- | --- | --- |
| Standing rules: integrity, method invariants, workflow, gates | `AGENTS.md` (imported by `CLAUDE.md`) | every session |
| Specification: what to build and how to evaluate it | `docs/SPEC.md`, `docs/EVAL_PLAN.md` | when a task cites them |
| Work queue and memory | `docs/MILESTONES.md`, `docs/DECISIONS.md`, `docs/RUNLOG.md` | read and updated per task |
| Ground truth the agent cannot argue with | `reference/` oracle, golden fixtures, acceptance tests | run by tests |
| Enforcement | `.claude/settings.json` (permissions, hooks), `.claude/hooks/` | always on |
| Fresh-context checks | `.claude/agents/spec-reviewer.md`, `results-auditor.md` | on request |
| Repeatable procedures | `.claude/skills/next-task`, `run-experiment`, `gate-report` | `/name` |
| Path-scoped reminders | `.claude/rules/core.md`, `envs.md` | when matching files are opened |

The design choice that matters most: E0 is not "done" when the agent says so, but when
`reference/acceptance/` passes — 121 tests covering the method page's worked examples (B(∅) = 0.4332, L = 0.6553, the ρ
table, the seven-step Reduce over all 24 legal orders, the noisy history, every configuration's edge weights)
plus random DAGs and sub-lattices. The suite was checked against a correct implementation (all pass) and against
seeded bugs: renormalizing over legal actions fails 53 tests; a Reduce that ignores conflicts fails 4.

## One-time setup (about 30 minutes)

1. Create the repository and add the kit at its root:
   `git init qet && cd qet && cp -r <kit>/. . && git add -A && git commit -m "Agent kit"`.
2. Fill in the **Environment** section of `AGENTS.md` (GPUs, scheduler, the GPU-hour threshold for asking,
   budgets, paths) and answer the open questions at the end of `docs/DECISIONS.md` — at least Q1 (action string
   order) before E0 starts. Accept decisions yourself by changing Proposed to Accepted.
3. Check the oracle: `python3 reference/make_fixtures.py --check && python3 -m pytest reference/test_oracle.py -q`.
4. Start Claude Code in the repository root. Run `/memory` (CLAUDE.md and AGENTS.md listed) and `/hooks`
   (three hooks listed). The hooks need `python3` on PATH; on Windows change the hook commands to `python`.

## Session plan

Keep one task per session and run `/clear` between tasks; the repository files carry the state. A `/goal`
evaluator only reads the conversation, so every condition below names a command whose output proves it.

**Session 0 — orientation, no code** (`claude --permission-mode plan`):

```text
Read AGENTS.md, docs/SPEC.md, docs/EVAL_PLAN.md, docs/MILESTONES.md and docs/DECISIONS.md. Do not write code.
Interview me with AskUserQuestion about every open question in DECISIONS.md and anything in SPEC.md you find
ambiguous or inconsistent. Then list the DECISIONS.md updates my answers imply, and propose a plan for E0.1–E0.3
with the acceptance command of each task.
```

**E0 — core library (CPU only).** Scaffold first with `/next-task E0.1`. Then each of E0.2–E0.6 takes two
prompts: a plan in plan mode, which you approve, then a goal that carries it out.

```text
Plan task E0.2 from docs/MILESTONES.md. Read SPEC §2–§3, the canonical-form and contract parts of
reference/qet_ref.py, and reference/fixtures/canonical.json. List the files and the tests you will write first,
the acceptance command, and any choice the spec leaves open. Do not write code yet.
```

```text
/goal E0.2 is done: `pytest reference/acceptance -q -k "canonical_json or action_string"` reports only passed tests,
the new tests in tests/core pass, `make lint` exits 0,
`git diff --stat $(git rev-list --max-parents=0 HEAD) -- reference/` prints nothing, the work is committed and
the E0.2 box in docs/MILESTONES.md has an evidence line; or stop after 25 turns and report what is left.
```

| Task | The plan reads | Acceptance filter |
| --- | --- | --- |
| E0.2 | SPEC §2–§3 | `-k "canonical_json or action_string"` |
| E0.3 | SPEC §4, §8 (ablation a2) | `-k "compile or canonical_order"` |
| E0.4 | SPEC §5 | `-k "reduce or render or same_completed"` |
| E0.5 | SPEC §6, §8, §9 | `-k lattice` |
| E0.6 | SPEC §10 | the whole suite, `pytest reference -q` |

Once a few tasks have gone cleanly, the rest can run as one goal: `/goal Tasks E0.4–E0.6 … are ticked with
evidence lines and the last run of pytest reference -q reports only passed tests …, or stop after 60 turns`.

Then `Ask the spec-reviewer subagent to review src/qet/core, save its report to docs/reports/E0-review.md, fix
every blocking finding, and tick E0.7.` Read the core diff yourself (`git log -p -- src/qet/core`): it is the
best hour you will spend on this project.

**E1 — training and inference stack (1 GPU).** One `/next-task` per session for E1.1–E1.6. The three tests to
watch: the two-pass gradient equals the autograd gradient (E1.3a), the inference prompt hash equals the
training prompt hash at every step (E1.4), and `make smoke` end to end (E1.7).

**M1 — FuncBenchGen and ToolHop.** `/next-task` through M1.4, then `/gate-report A`. You approve the GPU
budget, the frozen test manifest and the final wording of EVAL_PLAN §9 (edit it yourself), then tick Gate A.
Runs go one by one through `/run-experiment configs/m1/<name>.yaml`; the skill dry-runs, estimates cost and asks
before scheduler submissions. To approve a batch instead (say, the whole M1 run matrix), record it as an
accepted decision with its GPU-hour cap and allow `Bash(sbatch *)` for that session through `/permissions`.
Training jobs run on the cluster, not in the chat. In a later session:

```text
Check the jobs submitted since <date> in docs/RUNLOG.md. For completed runs, verify the manifests, run the
evaluations their configs name, regenerate the tables with scripts/make_tables.py, and have the results-auditor
trace every number before you show me anything.
```

Then `/gate-report B`. You decide whether M2 runs.

**M2, M3, R** follow the same pattern: tasks via `/next-task`, runs via `/run-experiment`, gates via
`/gate-report`, the auditor before any number reaches you.

## Running with less supervision

- Inside a session, `/goal` keeps Claude working across turns; it does not change the permission mode. Auto
  mode removes per-tool prompts; `acceptEdits` is the conservative middle ground. The `ask` rules (scheduler,
  push, rm) still stop for you.
- Headless: `claude -p "/goal <condition>" --output-format stream-json --verbose`. Hooks and permission rules
  still apply; in `-p` mode nothing is there to approve an `ask`, so those calls do not run.
- Parallel work: one git worktree and session per independent adapter (for example ToolHop and the toy
  environment), merged by you.
- Use `bypassPermissions` only inside a disposable container without your credentials or data.

## What stays with you

- Decisions, licenses (Llama, AppWorld, ToolBench / DTA-Tool), compute and API keys, gate approvals, spending.
- Reviews: the core diff after E0, every gate report, and every table that goes into the paper. The auditor
  recomputes numbers; it does not replace your judgment about whether the comparison is the right one.
- Changes to protected files. Start a maintenance session with `QET_ALLOW_PROTECTED_EDITS=1 claude` to let
  Claude edit SPEC.md, EVAL_PLAN.md or `reference/` on your instruction (the permission rules still block
  `third_party/` and `results/raw/`). After changing the oracle, run `python3 reference/make_fixtures.py` and
  commit oracle, fixtures and decision together.

## Warning signs

| Sign | Likely cause | Response |
| --- | --- | --- |
| A number in chat with no script behind it | Estimation or memory | Ask for the generating command and the auditor's table |
| Existing tests edited in the same commit as the code they test | Weakened test | `git diff` the tests; reject loosened tolerances, new skips |
| "Improvements" to the method | Scope creep | It must appear as a decision; reject or accept explicitly |
| Dev and test numbers both quoted during tuning | Leakage | Check selection logs; the auditor checks splits |
| A session running for hours on one task | Drift | `/goal clear`, `/clear`, restart from MILESTONES.md |

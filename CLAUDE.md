@AGENTS.md

# Claude Code notes

- Start every task that touches `src/qet/core` or `src/qet/train` in plan mode: explore, write the plan with the
  acceptance command, wait for approval. A plan the human approved earlier in the session (for example right
  before a `/goal`) counts as that approval: carry it out without asking again, unless it must change
  materially — then stop and show the change.
- Use `/next-task` to run one MILESTONES task end to end, `/run-experiment <config>` for any training or
  evaluation run, and `/gate-report <A|B|appworld>` to assemble a gate report and stop.
- Before a task in `src/qet/core` or `src/qet/train` is ticked, ask the **spec-reviewer** subagent to review the
  diff. Before any experimental number goes into a report, a table or a message, ask the **results-auditor**
  subagent to trace it. Their blocking findings must be fixed, or answered in DECISIONS.md, before you go on.
- Hooks (`.claude/settings.json`) block edits to `reference/`, `third_party/`, `results/raw/`, `docs/SPEC.md`,
  `docs/EVAL_PLAN.md` and the Claude config itself, and ask before scheduler jobs, pushes and deletions. When a
  hook blocks you, never route around it (Python, sed, cp, git checkout into those paths); say what you need.
- A Stop hook runs `make test-fast` when tracked Python files changed. If it reports failures, fix them before
  ending the turn, or state plainly why they cannot be fixed now.
- Run `/clear` between unrelated tasks; progress lives in `docs/`, not in this conversation. When compacting,
  keep: the current task id, files changed, failing tests with their exact command, open questions.
- Treat claims about results (yours and subagents') as unverified until a script reproduces them from
  `results/raw/`.

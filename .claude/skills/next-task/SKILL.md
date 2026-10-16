---
name: next-task
description: Run one task from docs/MILESTONES.md end to end — plan, test-first implementation, acceptance check, review, evidence, commit.
disable-model-invocation: true
argument-hint: "[task-id]"
---
Task: $ARGUMENTS — if empty, the first unchecked task in docs/MILESTONES.md whose prerequisites and gates are done.

1. Read the task, the SPEC / EVAL_PLAN sections it cites, and docs/DECISIONS.md. If the task is a GATE, run
   /gate-report for it instead and stop.
2. Plan: files to touch, tests to write first (always for core math), the exact acceptance command, risks.
   If the task touches src/qet/core or src/qet/train, show the plan and wait for approval.
3. Implement in small steps, running `make lint test-fast` as you go.
4. Run the acceptance command from MILESTONES.md and keep its output.
5. If core or training code changed, ask the spec-reviewer subagent to review the diff; fix its blocking
   findings or answer them in docs/DECISIONS.md.
6. Commit as `<task-id>: <summary>`. Then tick the box with
   `evidence: <command> → <result>; commit <hash>`, add Proposed decisions for any choice the spec left open,
   and commit that as `<task-id>: evidence`.
7. Report in three lines: what was done, the evidence, and the next task or the gate that blocks it.

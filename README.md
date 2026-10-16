# QET — implementation and experiments

Quotient Execution Training (QET) trains tool-using LLM agents on the equivalence class of a demonstration
instead of its one recorded order: Compile turns a successful trace into a DAG, a quotient interface renders
every legal history of the same progress as the same prompt, and a forward–backward recursion over the lattice
of progress states turns "finish from any legal state, in any legal order" into weighted SFT.

- Method, as implemented: `docs/SPEC.md`
- Experiments: `docs/EVAL_PLAN.md`; progress: `docs/MILESTONES.md`; runs: `docs/RUNLOG.md`
- Coding agents: `AGENTS.md` (Claude Code: `CLAUDE.md`, `.claude/`)
- Humans driving the agents: `KICKOFF.md`
- Ground truth for the core math: `reference/` (`pytest reference -q`)

---
name: gate-report
description: Assemble the report for a human gate (E0, A, appworld, B) from scripts and raw results, have it audited, then stop for the human's decision.
disable-model-invocation: true
argument-hint: "<E0|A|appworld|B>"
---
Gate: $ARGUMENTS

1. Generate `docs/reports/gate-$ARGUMENTS.md` with `python scripts/gate_report.py --gate $ARGUMENTS` (create or
   extend the script as needed; numbers come only from code). The required content is listed under the GATE
   item in docs/MILESTONES.md.
2. Gate B: give each falsification criterion of EVAL_PLAN §9, exactly as pre-registered, with its estimate,
   95% interval and verdict (holds / fails / inconclusive). Do not rephrase the criteria.
3. Ask the results-auditor subagent to audit the report; append its table and verdict.
4. End with the decision the human must make, the options and their costs. Then stop: do not start the next
   milestone and do not tick the gate yourself.

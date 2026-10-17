## Audit (results-auditor)

Transcribed from the results-auditor's reports; the auditor recomputed every number from `results/raw/` and the
committed manifests with its own code (in the session scratchpad) and edited nothing in the repository.

- **First audit, report at 3d18db1: FAIL.** No numeric mismatch, but four blocking items: an impossible McNemar cell
  (Δ > ψ), a precision footnote the data contradict, a cost list without dev tuning, and a §9 proposal that changed
  "at matched compute" without saying so; plus 15 notes, among them the 64-token decoding limit that would leave
  11.9% of test tasks unsolvable.
- **Re-audit, report at 031fd79: PASS WITH NOTES.** All four blocking items fixed and checked against the data;
  every new number recomputed and matching; nothing regressed; D-030 agrees with the report. Its eight non-blocking
  notes (wording in D-030 and the report, the ablation runs' grid cost, the warm-up as an upper bound, the joint-power
  assumption, the time outside the phases, the last step's overshoot, a RUNLOG row for the superseded statistics run)
  are applied in the commit after 031fd79.

Re-audit table (031fd79):

| Report item | Source | Verdict |
| --- | --- | --- |
| 22 source sha256 values; working copy equals 031fd79 | files on disk | OK |
| Statistics re-run M1-pilot-stats-20261017T190141Z | manifest, RUNLOG, `cmp` with the first run | OK: 195883c, clean, completed, byte-identical outputs |
| §1-3 (21 rows, every cell) | report.json, per-task recomputation | OK, 0 mismatches |
| §2 footnote | raw_records.jsonl, audits.jsonl | OK: every raw-precision < 1 graph has a value with several producers (train 402/402, dev 89/89, test 382/382, heldout 63/63, renamed 421/421) |
| §3 bullet, §4, §5 | stats.json, graphs.jsonl | OK, 0 mismatches |
| §6 steps, distinct graphs, pass rates, memory, time outside steps | train_log.jsonl, manifest | OK: 5 + 15 steps, 60 graphs, 18,509 and 7,745 tokens/s, 90.8 GiB, 79.8 s |
| §6 tokens per epoch, projection table (24 cells), warm-up sensitivity | graphs.jsonl, train.json, train_log.jsonl | OK: 7,286 tokens/s, +3.8% |
| §6 evaluation paragraph | greedy.jsonl, dev.json, test.json | OK: 36 vs 41 steps, 104 and 3,440 GPU-hours |
| §7 decoding-limit table (40 cells) and proposal | own action strings and the pinned tokenizers | OK: longest 240, max_new_tokens 512 |
| §8 McNemar table, counted pools, smallest Δ (21 cells) | Connor (1987), own widths | OK |
| §9 wording vs EVAL_PLAN §6, §8, §9 | EVAL_PLAN.md | OK: labelled as a proposal, departure stated |
| §10 numbers, grid cost | the projection | OK: 254, 508, 141 GPU-hours |
| D-030 vs the report | DECISIONS.md | OK, after the "first 10 graphs" wording fix |

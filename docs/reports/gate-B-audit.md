# Gate B audit (results-auditor, 2026-10-20)

Verdict: PASS, no blocking findings. Every number in sections 1-8 of docs/reports/gate-B.md was recomputed independently
from results/raw/*/predictions/*.jsonl and the manifests (own width computation, scipy binomtest for McNemar, a numpy
bootstrap and a replay of the project's RNG scheme); every number matches at the stated rounding, and the 177 sha256
values in the report's Sources match the files on disk. Protocol checks passed: clean manifests, greedy with
max_new_tokens 512 and eval_seed 0 everywhere, checkpoint frac-1 of a completed training run with matching seed, the
frozen test set in every run (D-028 hash), crit and full configs differing only in the 8 samples, identical
Resume@prefix starts across the 9 runs of configurations 3, 5 and 6 (37,218 episodes each), exactly one evaluation per
(configuration, seed), no selection on test. The verdicts follow from the EVAL_PLAN §9 wording committed (95cb04e)
before the first test evaluation.

Non-blocking notes and what was done: (1) commit the report and script: done with this file. (2) ToolHop was planned only
for configurations 2 and 6: the "Not run" paragraph now says so. (3) results of 3 and 6 existed when D-032e was decided:
now stated. (4) the comparisons are labelled fresh: added. (5) why criterion 3 has no McNemar: added. (6) ToolHop greedy
varies by seed (0.232 / 0.168 / 0.151) and sections 5-7 give point means only: descriptive, no comparison claimed.
(7) different vLLM batch composition of the crit batches: disclosed (D-032d/e). (8) section 8 discloses that
configurations 4-6 trained on 440-472 distinct graphs against about 4,800 for configuration 2. After the edits for
(2)-(5) the report was regenerated; only the prose paragraphs changed, every number is identical.

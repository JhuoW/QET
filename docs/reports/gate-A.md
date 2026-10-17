# Gate A report (M1: FuncBenchGen)

Sources: compile run `M1-compile-20261017T161839Z` (commit 2e428e3), pilot statistics `M1-pilot-stats-20261017T172519Z` (first 500 training graphs, tokenizer Qwen/Qwen2.5-7B-Instruct), pilot run `M1-pilot-qet-s0-20261017T175029Z`.

## 1. Compile coverage and fallback

| split | tasks | demos verified | chain fallbacks | Compile missed a true constraint | caught by the replay audit | unsafe orders left |
| --- | --- | --- | --- | --- | --- | --- |
| train | 5,000 | 5,000 | 542 (10.8%) | 542 | 542 | 0 |
| dev | 896 | 896 | 101 (11.3%) | 101 | 101 | 0 |
| test | 4,480 | 4,480 | 521 (11.6%) | 521 | 521 | 0 |
| heldout_larger | 240 | 240 | 91 (37.9%) | 91 | 91 | 0 |
| renamed | 4,480 | 4,480 | 522 (11.7%) | 522 | 522 | 0 |
| stateful_train | 1,200 | 1,200 | 0 (0.0%) | 0 | 0 | 0 |
| stateful_test | 120 | 120 | 0 (0.0%) | 0 | 0 | 0 |

Every demonstration passed the environment's checker before Compile (otherwise the compile run stops). Compile received no task constraints (D-020a). The replay audit ran 8 random orders plus one hoisted order per node (D-029).

## 2. Agreement with the generator's true DAGs

| split | precision (raw → final) | recall (raw → final) | recall by reachability (raw → final) | unsafe-order rate (raw → final) |
| --- | --- | --- | --- | --- |
| train | 0.995 → 0.928 | 0.993 → 0.915 | 0.993 → 1.000 | 0.043 → 0.000 |
| dev | 0.994 → 0.927 | 0.992 → 0.912 | 0.992 → 1.000 | 0.044 → 0.000 |
| test | 0.994 → 0.923 | 0.993 → 0.908 | 0.993 → 1.000 | 0.046 → 0.000 |
| heldout_larger | 0.987 → 0.684 | 0.981 → 0.661 | 0.981 → 1.000 | 0.117 → 0.000 |
| renamed | 0.994 → 0.923 | 0.992 → 0.908 | 0.992 → 1.000 | 0.048 → 0.000 |
| stateful_train | 0.272 → 0.272 | 1.000 → 1.000 | 1.000 → 1.000 | 0.000 → 0.000 |
| stateful_test | 0.272 → 0.272 | 1.000 → 1.000 | 1.000 → 1.000 | 0.000 → 0.000 |

"Final" is after the chain fallback; FuncBenchGen precision below 1 comes from the fallbacks, the stateful extension's from decision edges around payments that truly commute.

## 3. Structure: width, incomparable pairs, states

Graphs as trained, after the chain fallback (a fallback graph has width 1 and n + 1 states).

| split | width 1 | 2 | 3-4 | 5+ | incomparable pairs | states ≤ 16 | 17-64 | 65-256 | > S_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 20.1% | 15.5% | 29.2% | 35.2% | 25.4% | 19.2% | 42.4% | 16.8% | 21.5% |
| dev | 20.5% | 16.6% | 28.7% | 34.2% | 25.4% | 19.2% | 43.5% | 16.1% | 21.2% |
| test | 20.9% | 16.4% | 28.5% | 34.1% | 24.8% | 18.8% | 43.5% | 16.4% | 21.3% |
| heldout_larger | 54.6% | 0.0% | 0.4% | 45.0% | 27.2% | 0.0% | 54.6% | 0.0% | 45.4% |
| renamed | 20.9% | 16.4% | 28.5% | 34.3% | 25.2% | 18.9% | 43.0% | 16.6% | 21.5% |
| stateful_train | 0.0% | 0.0% | 33.3% | 66.7% | 29.6% | 30.8% | 35.8% | 0.2% | 33.3% |
| stateful_test | 0.0% | 0.0% | 33.3% | 66.7% | 29.9% | 32.5% | 34.2% | 0.0% | 33.3% |

- pilot sample (500 training graphs): width bins {'1': 91, '2': 75, '3-4': 146, '5+': 188}; 116 graphs above S_max = 256 (trained on a sub-lattice of K = 4 orders), mean exact lattice size 43.6

## 4. Invariance

- On the pilot sample: 38,700 prefixes of 5 random legal orders per graph rendered against the demo-order history of the same set: 0 differences (SPEC §5.6 a).
- Unit tests (reported as tests, not results): tests/core/test_reduce_render.py (300 random graphs × 20 orders × every prefix), tests/infer/test_agent.py (the agent following legal orders sees the training prompt of every state), tests/core/test_fuzz.py.

## 5. Tokens per graph (one epoch, scored + gradient)

| configuration | scored, mean | gradient, mean | largest graph (scored + gradient) |
| --- | --- | --- | --- |
| 2 | 0 | 24,251 | 45,304 |
| 3 | 0 | 126,110 | 1,172,041 |
| 4 | 126,110 | 126,110 | 2,344,082 |
| 5 | 126,110 | 126,110 | 2,344,082 |
| 6 | 126,110 | 126,110 | 2,344,082 |
| a2 | 0 | 24,270 | 45,304 |
| a3 | 50,768 | 50,768 | 311,422 |
| a4 | 57,024 | 57,024 | 366,056 |

By width bin (configurations 2 and 6): 1: 24,661 / 49,323; 2: 14,874 / 43,160; 3-4: 20,981 / 214,645; 5+: 30,333 / 463,012

## 6. Throughput, memory and the GPU-hour projection

Pilot `M1-pilot-qet-s0-20261017T175029Z` (configuration 6, Qwen/Qwen2.5-7B-Instruct, 500 training graphs, a 20-minute training stop; throughput only, no claims):

| measurement | value |
| --- | --- |
| training steps (after 5 warm-up steps) | 15 |
| pass 1 (scoring, no gradients) | 6,693,636 tokens in 362 s: 18,509 tokens/s |
| pass 2 (weighted SFT, forward and backward) | 6,693,636 tokens in 864 s: 7,745 tokens/s |
| weights, optimizer update, diagnostic | 0.37 s (0.03% of step time) |
| peak GPU memory (max over steps) | 90.8 GiB of 140.4 GiB (H200, 143,771 MiB) |
| GPU-hours by phase | load 0.001, train 0.384, eval 0.006; total 0.394 |

Training projection (a proposal: the token budget B per run is the human's decision). Matched compute gives every run the same B of scored + gradient tokens (EVAL_PLAN §8); a run's time follows its share of scoring tokens, measured above per pass, plus the pilot's other step time per token. Epochs are over the 5,000 training graphs, from the pilot sample's mean tokens per graph in each of the 224 grid cells, weighted by the full split's graphs per cell (configuration 2: 129,526,192 tokens per epoch; configuration 6: 1,292,010,968).

| B (tokens per run) | epochs, configuration 2 | epochs, configuration 6 | GPU-hours per run (configuration 2 / 6) | M1 matrix, 33 runs |
| --- | --- | --- | --- | --- |
| 125,000,000 | 0.97 | 0.10 | 4.5 / 3.2 | 127 |
| 250,000,000 | 1.93 | 0.19 | 9.0 / 6.4 | 254 |
| 500,000,000 | 3.86 | 0.39 | 17.9 / 12.7 | 508 |
| 1,000,000,000 | 7.72 | 0.77 | 35.9 / 25.4 | 1,017 |

The matrix is EVAL_PLAN §4's: configurations 1-6 × 3 seeds on Qwen2.5-7B-Instruct, 2, 3 and 6 × 3 seeds on Llama-3.1-8B-Instruct (assumed to train at the measured Qwen throughput; not measured), and a1, a2, a3 and a4 with K = 2, 4, 8 on one seed. Training only: the format warm-up runs (D-010), data preparation on the CPU and evaluation come on top.

Evaluation with the current decoding path (Hugging Face, one sequence at a time, D-026b): the pilot ran 6 greedy episodes, 41 generation steps and 818 generated tokens in 23 s (0.57 s per step). One evaluation of one checkpoint on the provisional test set (4,480 tasks, greedy plus 8 samples, about 16.3 steps per episode along the demonstrations) would take about 104 GPU-hours, so 33 runs need at least 3,440 GPU-hours before Resume@prefix, the checkpoint curves, ToolHop and the held-out sets. This path does not scale: batched decoding (vLLM or a batched loop with the same stopping and sampling rules, D-026b) has to land and be measured before M1.7.

## 7. Power analysis and proposed test size per cell

Paired tasks needed to detect a success-rate difference Δ between two configurations with McNemar's test (two-sided α = 0.05, power 0.8), for a share ψ of discordant tasks:

| Δ \ ψ | 0.10 | 0.20 | 0.30 |
| --- | --- | --- | --- |
| 0.05 | 312 | 626 | 940 |
| 0.10 | 77 | 155 | 234 |
| 0.15 | 33 | 68 | 103 |

Criterion 1 pools test graphs of width ≥ 3: 62.6% of the provisional test graphs; the smallest width bin (EVAL_PLAN §3.1) is width 2 with 16.4%. Smaller test sets would take the first k graphs of each cell of the provisional manifest; their pools are those shares of their size (expected, not counted). "Smallest Δ" is the least difference a pool detects with the power above (greedy, one seed per comparison; three seeds per configuration add precision).

| graphs per cell | test tasks | width ≥ 3 pool | smallest Δ (ψ = 0.1 / 0.2 / 0.3) | smallest bin: tasks, Δ at ψ = 0.2 | evaluation GPU-hours per checkpoint, current decoding |
| --- | --- | --- | --- | --- | --- |
| 5 | 1,120 | 702 | 0.034 / 0.048 / 0.058 | 184, 0.092 | 26 |
| 10 | 2,240 | 1,403 | 0.024 / 0.034 / 0.041 | 368, 0.066 | 52 |
| 20 | 4,480 | 2,806 | 0.017 / 0.024 / 0.029 | 736, 0.047 | 104 |

Proposal: 10 graphs per cell (2,240 test tasks): at ψ = 0.2 its width ≥ 3 pool detects Δ = 0.034 and its smallest bin 0.066, against 0.024 and 0.047 for 20 per cell at twice the evaluation cost. The renamed set follows the test graphs it renames; held-out larger stays as built (D-028b). The human chooses; then the test manifests are frozen.

## 8. Proposed wording for EVAL_PLAN §9 (for the human to accept, change or reject)

Common terms (proposal): FuncBenchGen test graphs; each run's checkpoint chosen by dev greedy success among its 25/50/75/100% budget checkpoints (ties to the later one); success averaged over the three seeds per task; differences with paired-bootstrap 95% intervals (10,000 resamples, clustered by task); "significant" means the interval excludes 0. McNemar per seed is reported alongside.

1. On graphs of measured width ≥ 3, if the interval of greedy success (6 − 3) or (6 − 4) contains 0 or lies below it, the objective claim fails.
2. If (2 − 1) is significant and positive but the interval of (6 − 2) contains 0 or lies below it, the gain belongs to the interface, not the objective.
3. With the off-path gap = Resume@prefix on-path minus off-path success at the frozen starts: if the interval of (gap 3 − gap 6) or of (gap 5 − gap 6) contains 0 or lies below it, claim C4 fails.
4-5. Unchanged (M2: StableToolBench and AppWorld).

## 9. Decisions for the human

1. **Training budget.** Proposal: B = 250,000,000 tokens per run, about 254 GPU-hours of training for the 33-run matrix at the measured throughput (500,000,000: about 508). At 250M, configuration 2 gets 1.93 epochs of the 5,000 graphs and configuration 6 0.19: QET's lattice costs about 10.0 times the tokens per graph, so at matched compute it sees a fraction of the graphs. That is the comparison EVAL_PLAN §8 defines; the 25-100% curves show whether more compute would change the ranking. Each run is above the 1 GPU-hour per-job cap (D-021), so the budget is recorded as a batch approval in DECISIONS.md.
2. **Evaluation.** The current decoder would need about 104 GPU-hours per checkpoint evaluation at 20 graphs per cell. Proposal: the agent builds batched decoding (no GPU spending beyond short validation jobs within the per-job cap), measures it, and the evaluation budget is set from that measurement.
3. **Test size per cell** (section 7) and the held-out sets (D-028); then the test manifests are frozen.
4. **EVAL_PLAN §9 wording** (section 8): EVAL_PLAN.md is human-only.
5. **Open questions that touch M1:** Q17 (a4 on every graph), D-028c (the stateful extension), Q16, Q18; and whether to measure Llama-3.1-8B throughput with its own short pilot (assumed equal to Qwen's above).

**STOP.** The agent does not start M1.5 or any further GPU run until the human decides.

## Sources (sha256)

- `results/raw/M1-compile-20261017T161839Z/manifest.json` 934da3a58fc1e0a4f25dc49c463ffe42d2a8cc5428de3cde570d96a60d096d08
- `results/raw/M1-compile-20261017T161839Z/train/report.json` 979502051c9721b5f569dfdf148081bf7cd0fabae72f6b50ff7ff50a5ed850dc
- `results/raw/M1-compile-20261017T161839Z/dev/report.json` 85530f90769287a35fc71661def84b49a5718609c347a8ef538db3b90f2fc053
- `results/raw/M1-compile-20261017T161839Z/test/report.json` a2fa97bbb42b3526ef7c7bad46fd0483a9932e21b474a4ef408e9bf6fa03e9a2
- `results/raw/M1-compile-20261017T161839Z/heldout_larger/report.json` 97b95d4a47399709b2de0efbcf80ad1f76ab6fb33856161830f9c290ce438325
- `results/raw/M1-compile-20261017T161839Z/renamed/report.json` cc55acf8cd0d392b76ac1fbe9c4d2c3c0ada0ebbc90e1efddf22154a46ae4325
- `results/raw/M1-compile-20261017T161839Z/stateful_train/report.json` 7e1652e0b67655ca38066f23ffb691ad7b89e470f4d73f09b75a48333041c62a
- `results/raw/M1-compile-20261017T161839Z/stateful_test/report.json` 136a5c8dfd394249d7135196ecf062034116ef31985405280947afef555a3ecf
- `results/raw/M1-pilot-stats-20261017T172519Z/manifest.json` 07f1c2ec9e93b4772f1ac8ede920616cec8f09abd8b40e7ab1ce5db51569e467
- `results/raw/M1-pilot-stats-20261017T172519Z/stats.json` e28664e61ddb21f84ebed7c0de16c21a28d258040220474279128cf7f8c40e93
- `results/raw/M1-pilot-qet-s0-20261017T175029Z/manifest.json` 2dee8b0362628c8ef74e05ae6e3f86478f3b1544e9605bb786a760de2735ddf8
- `results/raw/M1-pilot-qet-s0-20261017T175029Z/train_log.jsonl` e6b0bd2e61d487adef5f0dbb7d44713d82d930aad99c290357670d9aceb152ae
- `configs/m1/manifests/train.json` 35919d61dce815d1ec85f81d015d05053a6c5ff4e9a0378556cb2ead12386427
- `results/raw/M1-pilot-stats-20261017T172519Z/graphs.jsonl` 7dd134eece5ac8e093fda02920f53749aa3dbe5168c8a82aed23bdc37635d28d
- `results/raw/M1-pilot-qet-s0-20261017T175029Z/predictions/greedy.jsonl` 6ca8374be961b175f43cc96f5dbea567008d1419d1ca312635bbfff114e91b22
- `configs/m1/manifests/test.json` 36bec2d1d8ae4eb4e703c88fe5bf3fb3969aaea1600ba2c3ada18af6f7ff1835

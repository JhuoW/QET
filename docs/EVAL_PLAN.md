# QET — evaluation plan (implementation version)

Status: **normative**, human-owned. Source: the benchmarks and experimental-plan doc in Claude Docs
(rev. 33, 2026-10-16). The falsification criteria (§9) and test manifests are frozen at Gate A and never change
after results are seen. Published numbers below are reference points unless §2's checklist holds.

## 1. Claims and the comparison that tests each

| Claim | Controlled comparison (configurations from SPEC §8) | What the benchmark must offer |
| --- | --- | --- |
| C1 The quotient interface helps by itself | 1 vs 2 | Both prompt formats run and are scored the same way |
| C2 Not fixing the order inside the class helps | 3 vs 6, same states and starts | Many legal orders, so the KL term of eq. (4) is non-zero |
| C3 Weighting by later completion helps | 4 vs 6 | Ready actions whose follow-ups differ in difficulty |
| C4 Prefix training gives continuation | 6 vs 5, and 6 vs 3, on Resume@prefix | Environment state restorable to an unseen legal prefix |
| C5 The sub-lattice is affordable | exact vs a4 with K ∈ {2, 4, 8} | Wide graphs |
| H5 Gains track structural freedom | all, across width and commutation ratio | Controllable width, depth, distractors |

## 2. Benchmark requirements and the comparability checklist

A benchmark is usable when it has multi-call tasks with independent calls (chains are a null control),
deterministic replayable tools with snapshots, an order-agnostic programmatic checker, training demonstrations
with observations (or executable gold programs) split by task, derivable read/write contracts, a one-call-per-
step loop with re-rendered prompts, and published baselines on a named base model.

A published number is a **comparison** only if all hold: same split, metric and scoring script; same base model
and size; same training data (or a documented superset with the extra ablated); same inference budget in one
unit (decoding, samples per task, maximum steps); same judge model and version; same tools and documentation.
Otherwise it is a **reference point**. A different prompt template is acceptable for state- or answer-based
scores, disclosed, with C1 separating its effect.

## 3. Benchmarks and adapters

Pin every benchmark under `third_party/` at a commit; adapters live in `src/qet/envs/`; never modify official
code. Each adapter provides: task → x (request, tools, initial observation), `step(call) → observation`,
snapshot/restore, the native success check, contracts with their source, the step budget, and a test that
replaying the gold solution scores 100%.

### 3.1 FuncBenchGen — core, M1
- Code: https://github.com/megagonlabs/FuncBenchGen (BSD-3); paper https://arxiv.org/html/2509.26553v2.
- Hidden function DAG; settings: core functions, depth, irrelevant functions (connected / disconnected).
  Values are random three-digit integers; intermediate values are hidden; correct inputs give fixed outputs,
  wrong inputs give random incorrect values; calls capped at twice the minimum.
- Adapter work: an open-model runner on our inference loop (the repo uses LiteLLM for hosted APIs); training
  graphs from seeds disjoint from test; one successful executed trace per graph, compiled by our rules (the
  generator's DAG only audits); the method page's extension with stateful tools and known-truth commutable pairs, reported
  separately. The released sweep script covers only part of the grid and the paper lists distractor settings
  inconsistently, so freeze our own configuration and test manifest (hash in DECISIONS.md).
- Grid: core functions 5, 10, 20; depth 1 … n−1; irrelevant 0, 10, 20 in connected, disconnected and mixed form.
  Start with 5,000 training graphs; fix test size per cell at Gate A by power analysis. Bin results by measured
  width 1, 2, 3–4, 5+ (width 1 = chain, where configuration 5 equals SFT). Held-out sets: more calls than in
  training; renamed tools and values.
- Failure taxonomy to reproduce: value not yet known (66.8–81.3% of failures), incorrect value (17.9–33.2%).
- Reference (zero-shot, published; 5 / 10 / 20 core functions): GPT-5 72.5 / 38.2 / 15.0; Gemini-2.5-Pro
  46.5 / 14.4 / 6.0; Qwen3-235B-A22B-Instruct-2507 11.0 / 8.2 / 3.8; GPT-4.1 12.0 / 2.2 / 0.2. Restating known
  values raised GPT-5 from 62.5% to 81.3% (5 core) — the confound configuration 1 controls.
- Like-for-like comparison: our base model before and after training, same settings and call cap.

### 3.2 ToolHop — external check, M1 (evaluation only)
- https://arxiv.org/html/2501.02506v4: 995 queries, chains of 3–7 locally executable tools (3,912 tools).
  Mandatory-tool-use setting; queries untouched. Chains only: tests transfer and no harm, not order freedom.
- Reference: Qwen2.5-7B-Instruct 9.85%, Qwen2.5-72B-Instruct 45.43%, GPT-4o 49.04% (answer accuracy).

### 3.3 AppWorld — core, M2
- https://github.com/StonyBrookNLP/appworld (Apache-2.0; protected data redistributed encrypted);
  https://arxiv.org/pdf/2407.18901. 750 tasks: Train 105, Dev 60, Test-N 168, Test-C 417; 9 apps, 457 APIs.
  Database-diff unit tests (TGC per task, SGC per scenario); ground truth released for train and dev only;
  `save_state` / `load_state`; MCP server and function-calling API docs.
- Feasibility gate (before any training): audit 20–30 train tasks — can successful runs be written as atomic
  calls without hidden Python state; do contracts follow from REST methods; how many graphs stay wider than a
  chain; how large are their lattices. Train on the certifiable subset, disclose coverage, evaluate on the full
  official test sets. If most graphs collapse to chains, report QET as inapplicable there.
- Adapter work: atomic-call agent over MCP or function calling, one call per step, call budget as the unit;
  demonstrations from executing ground-truth programs on train tasks with call logging (plus rejection-sampled
  successes from a strong model, if approved); sub-lattice with S_max = 256.
- Dev split by scenario: one half for model selection, the other for Resume@prefix. Test sets: aggregate scores
  only; never publish derived test traces.
- Reference (different interfaces or budgets): GPT-4o IPFunCall (15 turns of batched calls) Test-N 32.1 / 16.1,
  Test-C 18.0 / 10.1 TGC / SGC; GPT-4o ReAct (code) 48.8 / 32.1, 30.2 / 13.0; Qwen2.5-32B code agent: no
  fine-tuning 39.2 / 18.6, RFT 47.9 / 26.4, LOOP 71.3 / 53.6 (Test-N). Like-for-like comparisons are internal.

### 3.4 StableToolBench + DTA-Tool — published head-to-head, M2
- StableToolBench https://arxiv.org/html/2403.07714v4: 765 solvable tasks in six subsets (I1-Inst 163, I1-Cat
  153, I1-Tool 158, I2-Inst 106, I2-Cat 124, I3-Inst 61); cached API responses, GPT-4-turbo simulates misses;
  SoPR / SoWR judged by gpt-4-turbo-2024-04-09.
- DTA-Tool https://huggingface.co/datasets/dongsheng/DTA-Tool: 21,342 successful ToolBench paths as DAGs
  (99.1% contain parallel calls); DTA paper https://arxiv.org/html/2501.12432v2; checkpoints released.
- Plan: train QET (configuration 6) and demo-order SFT (configuration 2) on the same successful paths with
  Llama-2-7B, 3 seeds each. Compile from argument provenance; GET endpoints read-only after a replay audit
  through the cache; DTA's GPT-built DAGs only as an ablation. Converter from our trajectories to ToolBench's
  answer format.
- Re-score released ToolLLaMA and DTA-Llama checkpoints and our runs in one job, one cache snapshot, one judge
  (if the judge is retired, one fixed replacement for all, approved by the human). Report the share of test
  calls served by the simulator. Drift is real: ToolLLaMA (DFSDT) averages 46.6 SoPR in one paper, 54.2 in the
  other. Published (DTA paper, six-subset average SoPR / SoWR): DTA-Llama 66.1 / 59.1; ToolLLaMA DFSDT 54.2 /
  47.1; ToolLLaMA ReAct 37.9 / 39.3; GPT-4 DFSDT 70.3 / 64.2.
- Not evidence for the certification claims (simulated misses). QET emits one call per step and DTA parallel
  rounds: compare SoPR / SoWR; report steps only as cost.

### 3.5 Supplementary, M3 (only after Gate B passes)
- **BFCL v3 multi-turn** (no-regression): 800 scored cases in Base, Missing Parameters, Missing Functions, Long
  Context; the response checker uses unordered containment, so legal reorderings pass. Keep user-turn
  boundaries in the rendered state; report the official aggregate. Reference: xLAM-2-8b-fc-r 69.25 multi-turn,
  72.83 overall (other training data).
- **TinyAgent** (plan mode): 16 functions, LLMCompiler plans with `$k` references, success = right function set
  and a DAG isomorphic to gold; TinyLlama-1.1B and Wizard-2-7B, LoRA, 3 epochs, lr 7e-5. Reference: TinyAgent-7B
  84.95%, TinyAgent-1.1B 80.06%, GPT-4-Turbo 79.08%. Write the isomorphism scorer and calibrate it by re-scoring
  the released models first; resolve the 40K vs 80K dataset-size question from the dataset card.
- **NESTFUL** (plan mode, test only): Win Rate by execution is primary (GPT-4o 0.60, three-shot);
  full-sequence accuracy is order-sensitive and only a diagnostic.
- Deferred: ComplexFuncBench (only under Cheng et al.'s cached protocol), GTool / TaskBench (planning
  metrics), τ-bench (user simulator).

## 4. Configurations, backbones, seeds, run matrix

Six core configurations and four ablations as defined in SPEC §8. Main backbone Qwen2.5-7B-Instruct; second
backbone Llama-3.1-8B-Instruct repeats configurations 2, 3 and 6; StableToolBench uses Llama-2-7B; TinyAgent
uses TinyLlama-1.1B and Wizard-2-7B. Three training seeds for core runs.

| Milestone | Runs | Content |
| --- | --- | --- |
| M1 | ≈ 33 | 18 core (6 configurations × 3 seeds, Qwen2.5-7B) + 9 (configurations 2, 3, 6 × 3 seeds, Llama-3.1-8B) + 6 single-seed ablations (a1, a2, a3, a4 with K = 2, 4, 8); ToolHop is evaluation only |
| M2 | ≈ 18 | AppWorld 12 (configurations 2, 3, 5, 6 × 3 seeds) + StableToolBench 6 (configurations 6 and 2 × 3 seeds) + inference-only re-scoring |
| M3 | ≈ 6 | best SFT baseline and QET on BFCL / TinyAgent / NESTFUL |

GPU budget: set at Gate A from the pilot's measured tokens per graph and throughput.

## 5. Experiment blocks

| Block | Question | Benchmarks | Configurations | Primary metrics |
| --- | --- | --- | --- | --- |
| A Execution success | Does QET improve completion? | FuncBenchGen, ToolHop (M1); AppWorld, StableToolBench (M2) | six core; released ToolLLaMA, DTA-Llama | official success, greedy and sampled; calls per task |
| B Continuation | Does QET finish from legal states the demo never visited? | FuncBenchGen, AppWorld held-out dev, StableToolBench | 1, 2, 3, 5, 6 | Resume@prefix success by condition and depth; off-path gap |
| C Structure and compiler | Do gains track width; does certification hold? | FuncBenchGen grid, stateful extension, held-out larger and renamed sets | core + QET trained with injected false-independence edges | success by width / depth / distractors; compile audits; unsafe-order rate |
| D Compute and data | Is the extra computation worth it? | FuncBenchGen, AppWorld | core; exact vs a4 | success vs GPU-hours, tokens, number of demos |

## 6. Resume@prefix (continuation protocol)
1. Split by task or scenario before any expansion: no task's orders or prefixes in both training and evaluation.
2. For each evaluation task with a verified trace, compile its graph and list legal prefixes at depths 1 … n−1.
3. Three start conditions per task, balanced across depths: empty; on-path (a prefix of the demo order);
   off-path (a legal set the demo never visited).
4. Restore the real environment by executing a legal order of the prefix from the snapshot (`load_state` on
   AppWorld, replay on FuncBenchGen, cache hits on StableToolBench). Never paste fabricated results into a prompt.
5. Give each agent only what it would see: QET gets the rendered state; chronological baselines get the executed
   calls as history and, separately, no history.
6. The agent finishes with no gold graph, ready set or future outputs; score with the native checker.
7. Report success by condition and depth, and the off-path gap (on-path minus off-path). On AppWorld use the
   held-out dev half only. The decisive comparison: 6 vs 3 and 6 vs 5; beating configuration 1 alone proves little.

## 7. Metrics
- Success: official metric; greedy, plus sampled single-rollout success = mean over 8 samples at T = 1.0;
  pass^4 per task estimated as C(c, 4) / C(8, 4) from c successes in 8 samples. Never best-of-k.
- Errors: value not yet known vs incorrect value (FuncBenchGen taxonomy), invalid action, budget exhausted;
  per call and per task.
- Compile: coverage, chain-fallback share, width, share of incomparable pairs, states per graph, replay-audit
  failures, agreement with FuncBenchGen's true DAGs (edge precision / recall, unsafe-order rate), and the
  byte-identical prompt test reported as a unit test, not a result.
- Cost: scored and gradient tokens, GPU-hours, demonstrations used, inference calls and LLM rounds per task.

## 8. Controls and statistics
- Matched compute: every configuration gets the same budget of scored + gradient tokens (QET's pass 1 counts);
  configurations without pass 1 therefore get more updates. Evaluate checkpoints at fixed budget fractions
  (25 / 50 / 75 / 100%) and report curves, not only end points; also report GPU-hours.
- Equal tuning: same optimizer, LoRA rank, warm-up and number of dev trials per configuration; model selection on
  dev success only.
- Decoding: the official greedy settings for published comparisons, plus sampled single-rollout success; one
  step budget per benchmark in one unit for all systems.
- Seeds: 3 for core configurations, the second backbone's three configurations, and StableToolBench.
- Tests: paired bootstrap (10,000 resamples) with 95% intervals, clustering prefixes and rollouts by task and
  AppWorld variants by scenario; McNemar on paired task outcomes (greedy). Report effect sizes with intervals.
- No hidden reasoning across steps for any system.

## 9. Falsification criteria (pre-registered at Gate A, 2026-10-18; frozen)
Common terms: FuncBenchGen test graphs; every run evaluated at its checkpoint at 100% of B, so the compared runs have
spent the same compute (up to the last step's overshoot of B, D-023f) and no checkpoint is selected; success averaged
over the three seeds of each task; differences with paired-bootstrap 95% intervals (10,000 resamples, clustered by
task); "significant" means the interval excludes 0. McNemar per seed is reported alongside.
1. Greedy success on graphs of measured width ≥ 3: if the interval of (6 − 3) or of (6 − 4) contains 0 or lies below
   it, the objective claim fails.
2. Greedy success on all test graphs: if (2 − 1) is significant and positive but the interval of (6 − 2) contains 0
   or lies below it, the gain belongs to the interface, not the objective.
3. Resume@prefix greedy success on the test tasks that have off-path starts, with the off-path gap = on-path minus
   off-path success at the frozen starts: if the interval of (gap 3 − gap 6) or of (gap 5 − gap 6) contains 0 or lies
   below it, claim C4 fails.
4. If QET does not beat re-scored ToolLLaMA on StableToolBench, the published comparison fails; parity with
   DTA-Llama is acceptable (QET does not target parallel latency).
5. If the AppWorld audit finds most graphs collapse to chains, QET is reported as inapplicable there; compile
   rules are never tuned on test data.

## 10. Reporting and provenance
- Three comparison types in separate tables: historical (published numbers), fresh (re-scored released
  checkpoints under our pinned setup), causal (our matched configurations).
- Planned tables: T1 FuncBenchGen main by width bin; T2 Resume@prefix; T3 ablations; T4 ToolHop; T5 AppWorld;
  T6 StableToolBench re-scored; T7 compile and structure audits; T8 supplementary. Figures: F1 success vs width;
  F2 success vs compute; F3 success vs number of demos; F4 error taxonomy; F5 exact vs sub-lattice.
- Every table and figure is produced by a script in `scripts/` from `results/raw/`; no hand-typed numbers.
- Run manifest (`results/raw/<run_id>/manifest.json`): run_id, created_at, git commit and dirty flag (reported
  runs must be clean), config path and sha256, seed, configuration id, λ, S_max, K, backbone id and revision,
  tokenizer and chat-template sha256, data manifest path and sha256, benchmark / evaluator commits, judge model
  and version, decoding parameters, budgets, hardware, library versions, scored / gradient tokens, GPU-hours,
  parent (warm-up) checkpoint, status.
- Benchmark manifests (frozen): data and evaluator commits, test ids or seeds, renderer version, decoding, limits,
  judge.

## 11. Risks, data rules, licenses
- Judge and cache drift (StableToolBench): one run, pinned judge and cache; show published and re-scored side by side.
- Budget units differ (AppWorld IPFunCall turns, ReAct LLM calls, our calls): compare in one unit; label the rest.
- Non-determinism (FuncBenchGen wrong-input values, simulated cache misses): fix seeds, log every observation,
  report simulated share, keep live-service tasks out of training.
- Contamination: DTA-Tool derives from ToolBench training data — check query overlap with test; prefer fresh
  FuncBenchGen seeds.
- Licenses: FuncBenchGen BSD-3; TinyAgent MIT; NESTFUL and Cheng et al. code Apache-2.0; AppWorld Apache-2.0
  with encrypted protected data (test ground truth withheld); DTA project page states none; ToolBench-X
  academic-only (not used). Check terms before releasing derived data; the human accepts licenses.

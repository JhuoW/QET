# Milestones

Work top to bottom. A box is ticked only with an evidence line underneath:
`evidence: <command> → <result>; commit <hash>`. **GATE** items are decided by the human: prepare the report,
then stop. Scope changes are the human's; status updates are the agent's.

## E0 — Core library, CPU only (SPEC §2–§6, §8–§10)
Goal: `src/qet/core` reproduces every fixture; property tests hold on random instances.

- [x] **E0.1 Scaffold.** `pyproject.toml` (src layout, package `qet`), Makefile targets of AGENTS.md, ruff and
  pytest config (`testpaths = ["tests", "reference"]`), `.gitignore` (results/raw, checkpoints, data, .env), empty
  modules of the layout. Accept: `make setup && make lint` pass; `make test-fast` runs (fixture tests may fail).
  evidence: `make setup && make lint` → exit 0 (qet-0.1.0 installed editable; ruff: all checks passed);
  `make test-fast` → runs: 44 passed, 120 failed (acceptance tests, facade not written yet); commit 7db3deb
- [x] **E0.2 Canonical form and contracts** (`canonical.py`, `contracts.py`; start `api.py` with the facade
  functions this task covers — each later task adds its own, E0.6 completes it). Accept:
  `pytest reference/acceptance -k "canonical_json or action_string"` passes; unit tests for glob resources, missing
  template arguments, volatile paths.
  evidence: `pytest reference/acceptance -q -k "canonical_json or action_string"` → 17 passed; `pytest
  tests/core/test_canonical.py tests/core/test_contracts.py -q` → 36 passed; `make lint` → exit 0; spec-reviewer:
  no blocking findings, its 3 should-fix items fixed in 983ab5e (D-015, Q7-Q10); commit 983ab5e (work: a74f5c4)
- [x] **E0.3 Compile** (`compile.py`, graph record I/O, compile report). Accept: `-k "compile or canonical_order"`
  passes; property and differential tests on ≥ 500 random demos (tests may import `reference/qet_ref.py`, for
  example its `random_demo` and `compile_edges`): identical edges to the oracle, edges point forward, labels
  valid, every conflicting pair and every (earlier, write) pair has an edge, demo order is topological.
  evidence: `pytest reference/acceptance -q -k "compile or canonical_order"` → 32 passed; `pytest
  tests/core/test_compile.py tests/core/test_fuzz.py -q` → 26 passed (600 oracle demos + 800 richer demos,
  identical edges and canonical orders); spec-reviewer: no blocking findings, review items fixed or recorded
  (D-016, D-019, Q11-Q12); commit 903897a (work: 8d3edd9)
- [x] **E0.4 Reduce and Render** (`reduce.py`, `render.py`, all three renderers of SPEC §5.7). Accept:
  `-k "reduce or render or same_completed"` passes; property test: ≥ 300 random graphs × 20 random orders × every
  prefix render byte-identically to the demo-order history of the same set, and match the oracle's render.
  evidence: `pytest reference/acceptance -q -k "reduce or render or same_completed"` → 19 passed; `pytest
  tests/core/test_reduce_render.py -q` → 20 passed (300 graphs × 20 orders × every prefix, equal to the oracle);
  spec-reviewer: no blocking findings (D-017, D-019); commit 903897a (work: ddec7c8)
- [x] **E0.5 Lattice and weights** (`lattice.py`, `weights.py`: enumeration with cap, μ, log B, flow, all
  methods of SPEC §8, sub-lattice sampling, B⁺). Accept: `-k lattice` passes; property tests for SPEC §9 items
  1–9 on ≥ 300 random DAGs (n ≤ 8), float64, plus differential tests against the oracle on random DAGs.
  evidence: `pytest reference/acceptance -q -k lattice` → 52 passed; `pytest tests/core/test_lattice.py
  tests/core/test_identities.py -q` → 27 passed (SPEC §9 items 1–9 on 300 DAGs, slow finite-difference test
  included; 300-DAG differential per method); spec-reviewer: no blocking findings (D-018); commit 903897a (work:
  3c7e901)
- [x] **E0.6 Facade** (`api.py`, SPEC §10). Accept: `pytest reference -q` reports only passed tests (0 failed,
  0 errors, 0 skipped).
  evidence: `pytest reference -q` → 134 passed (0 failed, 0 errors, 0 skipped); `make test` → 279 passed;
  `make lint` → exit 0; spec-reviewer: no blocking findings; commit 903897a (work: be7917b)
- [x] **E0.7 Review.** Run the spec-reviewer (fresh context) on `src/qet/core`; fix or answer every blocking
  finding (answers go to DECISIONS.md). Accept: review report saved to `docs/reports/E0-review.md`.
  evidence: `docs/reports/E0-review.md` saved: end-of-E0 review at ce7ef3e and follow-ups at 73e7413 and f11fbce,
  each "no blocking findings"; should-fix items fixed (73e7413, f11fbce), the rest recorded in D-015-D-020 and
  Q7-Q15; `pytest reference -q` → 134 passed; `make test` → 292 passed; `make lint` → exit 0; commit b604c87
- [x] **GATE E0** (optional but recommended): the human reads the core diff and the review.
  waived by the human on 2026-10-17: "You do not need to let me check, you goal is to implement the QET and its
  evaluations." (not a review; the core diff and `docs/reports/E0-review.md` remain unread by the human)

## E1 — Training and inference stack on a tiny model (SPEC §6.5–§7; 1 GPU)
Goal: the end-to-end pipeline works and is provably consistent before any expensive run.

- [x] **E1.1 Chat mapping and tokenization** (`train/chat.py`, D-002). Accept: target ids = action tokens + end-
  of-turn id; decoding the ids returns the action string; prompts are identical across two processes with
  different system dates (Llama 3.1 `date_string` pinned); a test per supported backbone (Qwen2.5, Llama 3.1)
  using their tokenizers.
  evidence: `pytest tests/train/test_chat.py -q -m ""` → 39 passed (Qwen2.5-0.5B/7B and Llama-3.1-8B tokenizers at
  pinned revisions; target = action tokens + end of turn, exact decode; golden ids; two processes with faked years
  2001/2031 and different hash seeds give identical prompt ids, and a clock-reading Llama template differs unpinned);
  `make lint` → exit 0; spec-reviewer: no blocking findings (review of d7f0f7f, follow-up of 321a214: S1-S6 fixed),
  last should-fix fixed in 8253e11 (D-022); commit 8253e11 (work: d7f0f7f, 321a214)
- [x] **E1.2 Scorer** (`train/scoring.py`). Accept: log π equals a manual per-token computation; batch and
  padding invariance within 1e-4 (bf16) / 1e-9 (float64 on CPU); eval mode asserted; token counts reported.
  evidence: `pytest tests/train/test_scoring.py -q -m "not gpu"` → 17 passed (manual per-token log pi within 1e-12;
  float64 batch/padding invariance within 1e-9; gradients; eval mode; no_grad in pass 1; float32 log-softmax and
  float64 sums on bf16/fp16 models; token counts); `pytest -m gpu` in PBS job 87789.gaas on an H200 at c21333c →
  2 passed, bf16 batch invariance worst |delta log pi| = 0 over 24 pairs (Qwen2.5-0.5B-Instruct); `make lint` →
  exit 0; spec-reviewer: no blocking findings, S1-S3 fixed in 6e854c8, the all-ones mask of note N1 reverted in
  c21333c after job 87769.gaas measured 0.50 nats of bf16 batch dependence (D-024b); commit c21333c (work: cf5641e)
- [x] **E1.3 Two-pass trainer** (`train/trainer.py`, `train/methods.py`; LoRA via PEFT with dropout 0). Accept:
  (a) on a tiny random-weight model in float64 on CPU, the two-pass gradient equals the autograd gradient of the
  objective through the lattice for configurations 5 and 6, and of the logged objective with state weights held
  constant for configurations 3 and 4 (max relative difference < 1e-8); (b) a torch table policy reproduces
  `reference/fixtures/toy_training.json` within 1e-6 using the production weight code; (c) the logged objective is
  Σ μ(−log B), not the pass-2 loss; (d) no optimizer step between passes (assert parameter hash).
  evidence: `pytest tests/train/test_trainer.py -q -m ""` → 38 passed (a: two-pass gradient = autograd for 3, 4, 5, 6
  and a4 with the sub-lattice and n = 0, 1, relative difference < 1e-8; b: toy_training.json within 1e-6 through the
  production weights and train_step; c: logged objective = Σ μ(−log B) < pass-2 loss; d: parameter hashes equal
  within each of 3 steps for foreach and fused AdamW, and in-place, .data and optimizer-step updates between the
  passes refused; all ten configurations' logged objectives equal the oracle's; one earlier run of this file had
  one unidentified failure, not reproduced in 4 reruns); `make lint` → exit 0; spec-reviewer: no blocking findings
  (review of e6cff1b, follow-up of b0078b8: S1-S7 fixed; F1, F2 fixed in 352d36d; Q17 opened); commit 352d36d
  (work: e6cff1b, b0078b8)
- [x] **E1.4 Inference agent** (`infer/agent.py`; HF generate for tests, vLLM for scale). Accept: a scripted
  policy that follows a random legal order on the toy environment succeeds 100% and, at every step, sees a prompt
  whose sha256 equals the training prompt of that state; invalid actions end the episode (D-006); greedy and
  8-sample decoding both run.
  evidence: `pytest tests/infer/test_agent.py -q -m "not gpu"` → 22 passed (24 toy tasks × 5 random legal orders
  succeed 100% and every step's prompt sha256 equals the training prompt of its state, also for the chronological
  and native renderers along the demo order; 13 kinds of invalid output end the episode after one step without an
  environment call; the budget; a resumed prefix; greedy and 8-sample decoding run and are reproducible; decoding
  log-probability = scorer's within 1e-9); `pytest -m gpu` in PBS job 87789.gaas at c21333c → the Qwen2.5-0.5B
  decoding test passed (5 toy tasks × greedy + 8 samples); `make lint` → exit 0; commit b9e8810
- [x] **E1.5 Toy environment** (`envs/toy.py`): seeded random function DAG tasks (integer values, hidden
  intermediates, distractors, wrong inputs → seeded random outputs) plus a stateful variant (account read / pay,
  orders) that exercises effect and decision edges. Accept: gold replay scores 100%; Compile on toy demos has
  edge recall 1.0 against the generator's true DAG.
  evidence: `pytest tests/envs/test_toy.py -q` → 26 passed (7 settings × 60 seeds: gold replay 100% in demo and
  generation order; Compile edge recall 1.0, direct and by reachability, unsafe-order rate 0, no underived values;
  precision 1.0 on function tasks and < 1 on stateful ones, where payments truly commute; replay audit 0
  mismatches); `make lint` → exit 0; commit 130699a
- [x] **E1.6 Runner, manifests, run log** (`run.py`, `manifest.py`, `runlog.py`, `configs/` schema). Accept:
  `python -m qet.run --config configs/smoke/qet.yaml --dry-run` prints the plan, token and GPU-hour estimates
  and writes nothing; a real run writes `results/raw/<run_id>/` with a manifest containing every field of
  EVAL_PLAN §10 and appends a RUNLOG row.
  evidence: `python -m qet.run --config configs/smoke/qet.yaml --dry-run` → plan, tokens per epoch, steps and an
  estimated 0.120 GPU-hours (assumed throughput), results/raw/ and docs/RUNLOG.md unchanged; `pytest tests/eval -q`
  → 12 passed (dry run writes nothing; a tiny-model run writes every §10 field, data manifest, predictions, metrics,
  checkpoints and a RUNLOG row; failed runs recorded; dirty trees refused); real runs: PBS job 87796.gaas wrote
  results/raw/E1-smoke-sft-s0-20261017T074443Z/ and E1-smoke-qet-s0-20261017T075405Z/, whose manifests have every
  EVAL_PLAN §10 field and match their RUNLOG rows (results-auditor); commit 3241e93
- [x] **E1.7 Smoke** (`make smoke`, < 30 min on 1 GPU): Qwen2.5-0.5B-Instruct, configurations 2 and 6, toy env,
  short training, evaluation with Resume@prefix, a table from `scripts/make_tables.py`. Accept: runs end to end;
  results-auditor confirms every number in the smoke table. No claims are drawn from smoke numbers.
  evidence: `make smoke` in PBS job 87796.gaas (H200, commit c6f32b6, clean tree) → exit 0, 17 min 54 s walltime;
  configurations 2 and 6 each trained (format warm-up, then a 4M-token budget) and evaluated (greedy, 8 samples,
  Resume@prefix); table results/tables/smoke-20261017T075405Z.md from scripts/make_tables.py; results-auditor:
  every number traced to results/raw/, no mismatches, no blocking issues (notes in D-025); no claims drawn

## M1 — FuncBenchGen + ToolHop (EVAL_PLAN §3.1–3.2)
- [x] **M1.1 Adapter**: FuncBenchGen pinned in `third_party/`; task → x; step; success check; call cap; error
  taxonomy. Accept: gold replay 100%; wrong-input behaviour matches the repo; tests.
  evidence: `pytest tests/envs/test_funcbenchgen.py -q` → 8 passed (gold replay 100% in 4 grid cells × 15 graphs ×
  3 random topological orders of the true DAG; observations equal the repository evaluator's results call by call
  over 40 mixed calls, "value not yet known" and "incorrect value" included; success needs the minimum number of
  calls; tasks deterministic, the global RNG untouched; pure contracts); third_party/FuncBenchGen pinned at
  0729e25 (submodule, unmodified); `make lint` → exit 0; commit 7f3ba9d
- [x] **M1.2 Manifests**: frozen grid configuration, training seeds disjoint from test seeds, held-out larger and
  renamed sets, stateful extension; hashes recorded in DECISIONS.md.
  evidence: `PYTHONHASHSEED=7 python scripts/make_m1_manifests.py` → 7 manifests (224-cell grid; 5,000 train, 896
  dev, 4,480 provisional test, 240 held-out larger, 4,480 renamed, 1,200 + 120 stateful); `PYTHONHASHSEED=99 python
  scripts/make_m1_manifests.py --check` → all 7 identical; `pytest tests/envs/test_m1_manifests.py
  tests/envs/test_funcbenchgen.py -q` → 19 passed (disjoint splits, the full grid, held-out larger needs more calls,
  renamed graphs isomorphic with new names, tasks rebuild to their hashes under any hash seed); hashes in D-028
  (drafts until Gate A); commit 31c2ec0
- [x] **M1.3 Demonstrations and compile** (D-007): one executed successful trace per training graph; compile;
  audit against the true DAGs (edge precision / recall, unsafe-order rate); compile report.
  evidence: `python scripts/compile_m1.py` → results/raw/M1-compile-20261017T161839Z (commit 2e428e3, clean;
  RUNLOG row): all 16,416 tasks of the 7 M1 manifests have a checker-verified demonstration (D-007), Compile on
  the trace alone (no task constraints, D-020a), the replay audit with random and hoisted orders (D-029) and the
  audit against the true DAG; per split: records, raw records, per-task audits and the compile report
  (report.json); `pytest tests/envs/test_m1_compile.py -q -m ""` → 5 passed; the statistics go to Gate A through
  the results-auditor; commit 76c0230
- [x] **M1.4 Pilot**: 500 training graphs; lattice sizes, width bins, tokens per graph; a short configuration-6
  run on Qwen2.5-7B-Instruct to measure throughput; GPU-hour projection for the run matrix.
  evidence: `python scripts/m1_pilot_stats.py --compile-run results/raw/M1-compile-20261017T161839Z` →
  results/raw/M1-pilot-stats-20261017T190141Z (commit 195883c, clean; RUNLOG row; byte-identical to the earlier
  run M1-pilot-stats-20261017T172519Z): width bins, lattice sizes and tokens per graph for configurations 2-6 and
  a2-a4 on the first 500 training graphs, invariance on every prefix of 5 random orders per graph; `qsub -v
  CONFIG=configs/m1/pilot-qet.yaml -o logs/pbs/ scripts/pbs/run.pbs` (job 88015.gaas, 1 H200, after the dry run) →
  results/raw/M1-pilot-qet-s0-20261017T175029Z (commit 3c8b23a, clean, completed, no code change during the run;
  RUNLOG row): pass-1 and pass-2 throughput, peak memory, decoding speed; `python scripts/gate_report.py --gate A`
  → docs/reports/gate-A.md (structure, tokens per graph, throughput, memory, the GPU-hour projection for the
  33-run matrix, the decoding limit, test-size options), results-auditor: FAIL at 3d18db1 (4 blocking items,
  fixed), PASS WITH NOTES at 031fd79 (notes applied); commit bbb8f99
- [x] **GATE A** (`scripts/gate_report.py --gate A` → `docs/reports/gate-A.md`): compile coverage and fallback,
  width distribution, incomparable-pair share, states per graph, true-DAG agreement, invariance test results,
  tokens per graph, GPU budget, proposed test size per cell (power analysis), final wording of EVAL_PLAN §9.
  STOP. The human approves budget, test manifest and pre-registration.
  decided by the human on 2026-10-18 (D-030): training budget "125M tokens" per run (the batch approval for M1.5
  and M1.6); test size "10 per cell"; tuning "No grid"; max_new_tokens "512"; EVAL_PLAN §9 "Accept as proposed"
  (report section 9; the human pastes it into EVAL_PLAN.md, and no test evaluation runs before it is there); Q17
  "Every graph"; D-028c "Accept it". Report audited by the results-auditor (PASS WITH NOTES at 031fd79; report
  commit bbb8f99).
- [x] **M1.5 Format warm-up checkpoints** (D-010) per backbone and seed.
  evidence: `scripts/pbs/submit.sh configs/m1/warmup-measure.yaml` (job 88092.gaas) →
  results/raw/M1-warmup-measure-qwen7b-s0-20261018T032752Z (commit f506d83, clean, pinned checkout): on Qwen2.5-7B,
  seed 0, quotient renderer, 0 of 200 sampled dev states (100 dev tasks spread over the grid x 2 states) were invalid
  actions after 5 warm-up steps (5 of 200 before any step; 2 of 200 at step 10, 0 at 15-40), so the rule fixed before
  the run gives 5 steps (D-010 update), applied to every renderer and backbone without its own check;
  `python scripts/make_m1_configs.py warmups --steps 5`, dry runs pass, then the 10 warm-up jobs (88101-88110; the
  three Llama jobs failed at model load because the weights were not cached, recorded as failed, and ran again as
  88116-88118 after the human approved the download) → 10 completed warm-up runs, one per renderer, backbone and seed
  (5 steps, 0.53-0.63M tokens, 0.08-0.09 GPU-hours each; checkpoint files match their manifests; RUNLOG rows); the 33
  training configs name them as parents; results-auditor: PASS WITH NOTES; commit 19eadf6
- [ ] **M1.6 Training runs** (EVAL_PLAN §4, ≈ 33), each through `/run-experiment` with matched compute.
- [ ] **M1.7 Evaluation**: blocks A–D on FuncBenchGen; ToolHop for the base model and configurations 2 and 6;
  Resume@prefix; injected false-independence ablation; compute and data curves.
- [ ] **GATE B** (`--gate B` → `docs/reports/gate-B.md`): each falsification criterion with its interval and
  verdict; results-auditor pass. STOP. The human decides whether M2 runs.

## M2 — AppWorld + StableToolBench (only after Gate B passes)
- [ ] **M2.1 AppWorld setup** (the human accepts the data terms): atomic function-calling agent, one call per
  step, save / load state, call budget.
- [ ] **GATE AppWorld feasibility**: audit 20–30 train tasks (atomic calls, REST contracts, width, lattice
  sizes) → `docs/reports/appworld-feasibility.md`. STOP.
- [ ] **M2.2 Demonstrations, compile, dev split by scenario** (selection half, Resume@prefix half; frozen).
- [ ] **M2.3 Runs**: configurations 2, 3, 5, 6 × 3 seeds with the sub-lattice.
- [ ] **M2.4 Evaluation**: Test-N and Test-C official TGC / SGC (aggregate only); Resume@prefix on held-out dev.
- [ ] **M2.5 StableToolBench**: pinned cache snapshot and judge (human approves judge and API budget); re-score
  released ToolLLaMA and DTA-Llama; train configurations 6 and 2 on DTA-Tool paths with Llama-2-7B × 3 seeds;
  simulated-call share; overlap check with test queries.
- [ ] **M2.6 Report** → `docs/reports/M2.md`.

## M3 — Supplementary
- [ ] **M3.1 BFCL v3 multi-turn** no-regression (user-turn boundaries kept).
- [ ] **M3.2 TinyAgent** plan mode: scorer calibrated on released models first.
- [ ] **M3.3 NESTFUL** plan mode, Win Rate.

## R — Paper artifacts
- [ ] **R.1** Tables T1–T8 and figures F1–F5 from `scripts/`, three comparison types separate.
- [ ] **R.2** Reproducibility bundle: configs, manifests, benchmark manifests, environment lock, commands.
- [ ] **R.3** results-auditor over every table, figure and number in `docs/reports/final.md`: 0 untraceable.

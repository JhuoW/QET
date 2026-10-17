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

- [ ] **E1.1 Chat mapping and tokenization** (`train/chat.py`, D-002). Accept: target ids = action tokens + end-
  of-turn id; decoding the ids returns the action string; prompts are identical across two processes with
  different system dates (Llama 3.1 `date_string` pinned); a test per supported backbone (Qwen2.5, Llama 3.1)
  using their tokenizers.
- [ ] **E1.2 Scorer** (`train/scoring.py`). Accept: log π equals a manual per-token computation; batch and
  padding invariance within 1e-4 (bf16) / 1e-9 (float64 on CPU); eval mode asserted; token counts reported.
- [ ] **E1.3 Two-pass trainer** (`train/trainer.py`, `train/methods.py`; LoRA via PEFT with dropout 0). Accept:
  (a) on a tiny random-weight model in float64 on CPU, the two-pass gradient equals the autograd gradient of the
  objective through the lattice for configurations 5 and 6, and of the logged objective with state weights held
  constant for configurations 3 and 4 (max relative difference < 1e-8); (b) a torch table policy reproduces
  `reference/fixtures/toy_training.json` within 1e-6 using the production weight code; (c) the logged objective is
  Σ μ(−log B), not the pass-2 loss; (d) no optimizer step between passes (assert parameter hash).
- [ ] **E1.4 Inference agent** (`infer/agent.py`; HF generate for tests, vLLM for scale). Accept: a scripted
  policy that follows a random legal order on the toy environment succeeds 100% and, at every step, sees a prompt
  whose sha256 equals the training prompt of that state; invalid actions end the episode (D-006); greedy and
  8-sample decoding both run.
- [ ] **E1.5 Toy environment** (`envs/toy.py`): seeded random function DAG tasks (integer values, hidden
  intermediates, distractors, wrong inputs → seeded random outputs) plus a stateful variant (account read / pay,
  orders) that exercises effect and decision edges. Accept: gold replay scores 100%; Compile on toy demos has
  edge recall 1.0 against the generator's true DAG.
- [ ] **E1.6 Runner, manifests, run log** (`run.py`, `manifest.py`, `runlog.py`, `configs/` schema). Accept:
  `python -m qet.run --config configs/smoke/qet.yaml --dry-run` prints the plan, token and GPU-hour estimates
  and writes nothing; a real run writes `results/raw/<run_id>/` with a manifest containing every field of
  EVAL_PLAN §10 and appends a RUNLOG row.
- [ ] **E1.7 Smoke** (`make smoke`, < 30 min on 1 GPU): Qwen2.5-0.5B-Instruct, configurations 2 and 6, toy env,
  short training, evaluation with Resume@prefix, a table from `scripts/make_tables.py`. Accept: runs end to end;
  results-auditor confirms every number in the smoke table. No claims are drawn from smoke numbers.

## M1 — FuncBenchGen + ToolHop (EVAL_PLAN §3.1–3.2)
- [ ] **M1.1 Adapter**: FuncBenchGen pinned in `third_party/`; task → x; step; success check; call cap; error
  taxonomy. Accept: gold replay 100%; wrong-input behaviour matches the repo; tests.
- [ ] **M1.2 Manifests**: frozen grid configuration, training seeds disjoint from test seeds, held-out larger and
  renamed sets, stateful extension; hashes recorded in DECISIONS.md.
- [ ] **M1.3 Demonstrations and compile** (D-007): one executed successful trace per training graph; compile;
  audit against the true DAGs (edge precision / recall, unsafe-order rate); compile report.
- [ ] **M1.4 Pilot**: 500 training graphs; lattice sizes, width bins, tokens per graph; a short configuration-6
  run on Qwen2.5-7B-Instruct to measure throughput; GPU-hour projection for the run matrix.
- [ ] **GATE A** (`scripts/gate_report.py --gate A` → `docs/reports/gate-A.md`): compile coverage and fallback,
  width distribution, incomparable-pair share, states per graph, true-DAG agreement, invariance test results,
  tokens per graph, GPU budget, proposed test size per cell (power analysis), final wording of EVAL_PLAN §9.
  STOP. The human approves budget, test manifest and pre-registration.
- [ ] **M1.5 Format warm-up checkpoints** (D-010) per backbone and seed.
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

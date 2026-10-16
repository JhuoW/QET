# E0 review (task E0.7): `src/qet/core` checked against SPEC and the oracle

- **Commit reviewed:** `ce7ef3e` (`E0.3-E0.6: evidence`). Working tree clean.
- **Date:** 2026-10-17
- **Reviewer:** spec-reviewer subagent, fresh context.
- **Scope:**
  - All of `src/qet/core`: `canonical.py`, `contracts.py`, `compile.py`, `reduce.py`, `render.py`, `lattice.py`, `weights.py`, `api.py`.
  - Checked against `docs/SPEC.md` §2–§6 and §8–§10, and against the oracle `reference/qet_ref.py`.
  - `tests/core/` was read as evidence, not trusted.
  - Deviations D-015 to D-019 and open questions Q7 to Q12 in `docs/DECISIONS.md` were judged for justification and conservativeness.
  - `src/qet/train` and `src/qet/infer` are E1 stubs. They are out of scope except where a core API makes later work easier or harder.

**Verdict: no blocking findings.** There are 2 should-fix items (both for `docs/DECISIONS.md`, plus small guards or diagnostics) and 10 notes. I found no deviation from SPEC in the code that was not already recorded and on the conservative side.

## Commands and results

| Command | Result |
| --- | --- |
| `make lint` | `All checks passed!` and `57 files already formatted`; exit 0 |
| `make test-fast` | `278 passed, 1 deselected in 16.57s`; exit 0 |
| `python3 -m pytest reference -q` | `134 passed in 3.21s` |
| `python3 -m pytest tests -q -m slow` | `1 passed, 144 deselected in 21.61s` |
| `make test` | `279 passed in 38.81s` |
| `python3 reference/make_fixtures.py --check` | `fixtures up to date -> .../reference/fixtures`; exit 0 |
| `git diff --stat kit -- reference/` | empty (`kit` = `1557732 Agent kit`) |
| `git diff --stat kit HEAD -- docs/SPEC.md docs/EVAL_PLAN.md third_party .claude` | empty |
| `git log -p -- tests/`, removed lines mentioning `range(`, tolerances, `skip`, `xfail` | none |

One test was removed in the test history: `test_omitted_fields_are_empty`. D-015(b) made an omitted `reads` or `writes` an error, and the test was replaced by the stricter `test_reads_and_writes_must_be_declared_and_volatile_may_be_omitted`.

### Ad-hoc checks

The scripts were throwaway files in the session scratchpad (`/tmp/claude-1304605107/-home-wei-zhuo-My-Proj-QET/7691d9a6-1b16-4c03-bbdf-175d7ef6ae6d/scratchpad/`) and were not committed. Each one put the worktree's `src` first on `sys.path` and asserted `qet.__file__` pointed there.

| Check | Result |
| --- | --- |
| `identities.json` (eq. 4 for both models, eq. 5 at the root) through `api.lattice_summary`. No test does this. | All equal within rel 1e-12. Eq. 5 bound: 0.8770156536188256 vs 0.8770156536188258. |
| `render.json` `running_targets_by_state` through `canonical.action_string`. No test does this. | Identical. |
| `compile.json` `expected_n_states` / `expected_n_orders` through `full_lattice` and `completion_counts`. No test does this. | All 16 cases match. |
| `toy_training.json` (4 runs × 200 steps) re-run with the production `weights.edge_weights` and a pure-Python table policy | Maximum absolute deviation 0.0. π(A given {B}) goes 0.10 → 0.956 under QET and stays at 0.100 under demo SFT. |
| Finite differences on sub-lattices: 150 random DAGs, S = prefix sets of 2–4 orders. `tests/core` only does this on full lattices. | Maximum abs(−∂objective/∂log π − w): qet (λ = 0 and 0.5) 1.2e-9; sampled_paths 1.0e-9; frontier with occupancy held fixed 8.5e-10. |
| Differential fuzz with a new generator, 4000 demos: glob placeholders whose values contain `*`, null and absent contracts, an unknown tool, repeated calls, nested empty / error / null observations, truncation | 0 differences from the oracle in Compile, Reduce or Render. 0 invariance failures over 6 random legal orders × every prefix. |
| Hand-made adversarial inputs through the facade and the oracle | Identical for: duplicates (all 6 permutations of `f(x=1)`, `f(x=1.0)`, `f(x=1)`); unknown and null contracts; 13 empty or error observation shapes; glob pairs (`order:*`, `*:7`); non-identifier keys; volatile paths through lists and dotted keys. The only differences are resources containing a newline (D-015a), where ours adds the `effect` edge. |
| n = 0 and n = 1, every method | Identical to the oracle. For n = 0, μ(V = ∅) = 1 and the loss is −log π(end). |
| Zero-probability edges on the running example | Ours returns 0 where the oracle returns NaN (an unreachable state with B = 0). Ours raises `ValueError` where a state that receives occupancy has B = 0; the oracle gives NaN/inf or `ZeroDivisionError`. This is D-018(d). |
| 10 independent nodes (1024 states, above S_MAX) | `build_lattice` gives a 35-state sub-lattice. Every method with `plus=True` matches the oracle on the same S. B_S ≤ B ≤ B⁺; μ sums to 1; finish weight is 1 within 1e-15; invalid explicit state sets are rejected. |
| `PYTHONHASHSEED` in {0, 1, 2, 12345, random} | Identical sha256 over compile edges, canonical orders, Reduce, all three renderers, graph records, the compile report, sub-lattice states and lattice summaries. |
| `glob_match` against a `re.DOTALL` regex, 100 000 random pairs | All equal. |

## Findings

| # | Severity | Location | What is wrong | Concrete failing input | Suggested fix |
| --- | --- | --- | --- | --- | --- |
| 1 | should-fix (DECISIONS.md + diagnostic) | `docs/DECISIONS.md:55-56` (D-005); `src/qet/core/compile.py:102`; `src/qet/core/compile.py:355-392` | D-005 says a boolean argument "equal to an earlier output gets a data edge". That understates the behaviour. Under SPEC §4 rule 1 (implemented faithfully at `compile.py:102`, same as the oracle), any argument value found neither in the request nor in an earlier observation is "underived": a boolean flag, `null`, a default such as `limit=10`, a code such as `lang="en"`. It then depends on **every** earlier call, whether or not any output equals it. Calls carrying such constants compile to a chain. The compile report counts these edges as plain `data`, so the Gate A width and incomparable-pair numbers hide the cause. | Request `Get the weather in NYC, SG and LON.`, contract `get_weather: reads weather:{city}`, three calls `get_weather(city=X, verbose=true)`. Edges: `[[0,1,data],[0,2,data],[1,2,data]]`, a chain with 4 states instead of 8. The same happens with `units=null`, `limit=10` or `lang="en"`. With no flag: `[]`. With `start_values=[true]`: `[]`. | Correct D-005's wording. Add an open question: should adapters declare `true`, `false`, `null` and schema defaults and enums as start values by default? Add an underived-value count (and the values) to the compile record and report as a diagnostic, leaving the edges unchanged. |
| 2 | should-fix (guard until Q11 is answered) | `src/qet/core/compile.py:315-327`; `src/qet/core/render.py:117-118` | `compile_demo` accepts `task["initial_observation"]`, counts its leaves as available from the start (which drops their data edges) and stores it in the record. No renderer shows it, so a training target at ∅ can contain values that are absent from its prompt. Q11 records the gap ("decide before an adapter (E1.5, M1) uses it"), but nothing enforces it. | The `compile.json` "initial_observation" case: request `Compute h.`, `initial_observation {"inbox":[{"id":42}]}`, steps f(), g(), `h(a=42)`. There are no edges, so `{"tool":"h","args":{"a":42}}` is a target at ∅, and the ∅ prompt contains no `42`. | Until Q11 is decided, make `compile_demo` (the graph-record path, not the fixture facade `compile_edges`) raise when `initial_observation` is not null. Alternatively, render it according to the chosen answer. |
| 3 | note | `src/qet/core/contracts.py:143-161`; `docs/DECISIONS.md:104-105` (D-015a); commit `983ab5e` message | D-015(a) and the commit message call the matcher "linear-time". It is the greedy backtracking matcher: O(len(pattern)·len(s)) in the worst case, as its own docstring says. Results are correct and there is no exponential blowup. | `glob_match("*"+"a"*4000+"b", "a"*8000)` takes 1.79 s. One `conflict` between `note:{title}` calls, with `title="*"+"a"*3000+"b"` against `"a"*6000`, takes 2.08 s. | Correct the wording. Or match segments with `str.find`: anchor the prefix and suffix, then match middle segments greedily from the left (close to linear). |
| 4 | note | `src/qet/core/compile.py:39-51`; `src/qet/core/render.py:118` | The request is used raw (not NFC). A request in NFD misses whole-word start-value matches, which adds conservative but spurious data edges. It is also rendered in NFD while targets are NFC. The oracle behaves the same. | NFD request `Get the weather in Zürich and Genève.` with calls `get_weather(city="Zürich")`, `get_weather(city="Genève")` gives `[[0,1,data]]`. The NFC request gives `[]`. The rendered `[Task]` line is not NFC. | NFC the request (and tool descriptions) in `StartValues` and in `render`, or require adapters to do it. Record as a decision. |
| 5 | note | `src/qet/core/render.py:20-24`; `src/qet/core/compile.py:309` | `compile_demo` supports a finish tool with another name (`task["finish"]`, D-016d), but the pinned system text always says "Call finish". | Task with `"finish": "submit"`: the record's final tool is `submit`, but the system prompt still says "Call finish when the task is complete." | Decide that adapters always name the finish tool `finish` and drop the override, or record the mismatch as accepted. |
| 6 | note | `src/qet/core/lattice.py:252-260` | `edge_logp` rejects NaN but accepts positive values and +inf. A sign slip in the E1 scorer (passing −log π) silently gives plausible but wrong weights. A +inf input makes every weight and the objective NaN, with no exception. | Running example with every log π negated: the weights come out as the "mirror" (0.2368/0.2632 instead of 0.2632/0.2368), with no error. Setting log π(0 given ∅) = +inf makes all weights `nan`. | Raise when the value is > 0 or infinite (+inf). A correct float32 log-softmax sum is never positive. |
| 7 | note | `src/qet/core/weights.py:175-183`; `src/qet/core/weights.py:224` | `logged_objective` requires the keywords `log_b` and `mu` even for `path_sft`, `sampled_paths` and `frontier`, which do not use them. `summarize` needs log π for every lattice edge. Configurations 1–2 have no pass 1. | `logged_objective(ew, lat, logp)` without `log_b`/`mu` raises `TypeError`. It works only with `log_b=None, mu=None`. | Make `log_b` and `mu` optional and compute them when needed, so E1 can log the demo-path NLL from pass-2 log-probs alone. |
| 8 | note (E1 hazard) | `src/qet/core/canonical.py:31-32, 46-47`; `src/qet/core/reduce.py:57` | `canon` raises on NaN, ±inf and NFC key collisions (this is SPEC §2). At inference, a parsed model action or an environment observation with such values would raise out of Reduce/Render instead of ending the episode as `invalid_action` (D-006). | `api.reduce([{"tool":"get_weather","args":{"city":"NYC"},"obs":float("nan")}], C)` raises `ValueError`. The same happens for a NaN argument and for obs `{"é":1,"é":2}` (one NFC, one NFD). | E1.4: the action parser catches these and returns `invalid_action`; adapters sanitize observations; test both. |
| 9 | note | `src/qet/core/lattice.py:303-310`; `src/qet/core/api.py:100` | `parse_state_key` accepts non-canonical keys. Two spellings of the same edge silently overwrite each other in `lattice_summary` (facade only). | Keys `"0,1|2"` and `"1,0|2"` (also `" 1"`, `"1_0"`) parse to the same edge; the last one wins. | Require `state_key(parse_state_key(k)) == k` and reject duplicates. |
| 10 | note | `src/qet/core/compile.py:158-167` | `width()` finds augmenting paths recursively. Graphs with more than about 1000 nodes hit Python's recursion limit in `compile_report`. | A chain of 1500 nodes passed to `width`. | Use an iterative augmenting path, or document the limit. |
| 11 | note | `src/qet/core/compile.py:112-113` | Rule 5 (`task_order`) is the one channel through which an adapter could feed a benchmark's ground-truth DAG into Compile, which SPEC §4 forbids. Core cannot detect this. | A FuncBenchGen adapter that passes the generator's DAG edges as `task_order`. | Add an M1.3 audit item: adapters never pass ground-truth edges as task constraints. |
| 12 | note (coverage) | `tests/core/`; `reference/acceptance/test_core_fixtures.py` | `identities.json`, `render.json["running_targets_by_state"]` and `compile.json`'s `expected_n_states` / `expected_n_orders` are not compared against `src/qet` by any test. I reproduced all of them here and they all match. | None (gap in tests, not a failure). | Add `tests/core` cases for them. `toy_training.json` belongs to E1.3(b). |

### Assessment of D-015 to D-019 and Q7 to Q12

- **D-015(a)** (`*` matches newlines): justified and conservative. Ours reports a conflict whenever the oracle does. On 3000 random cases it differs only when a resource contains a newline, and then ours adds the edge. Compile and Reduce use the same function, so invariance is kept. Only the "linear-time" wording is wrong (note 3).
- **D-015(b)** (contract validation): conservative. It fails loudly on inputs the oracle accepts silently, such as `"volatile": "ts"`, which the oracle would iterate as the keys `t` and `s`.
- **D-016:** report-level choices only. (b) The chain fallback implies every order, so it is conservative.
- **D-017:** (a) matches SPEC §5.2 literally. (b) is consistent with §1 (canonical observations) and §5.4. (c) and (d) are validation only.
- **D-018:** (d) gives 0 instead of NaN for unreachable states and raises at reached states with B = 0, which is conservative. (g) matches the oracle and the sub-lattice fixtures. (f) is facade-only and consistent with SPEC §10.
- **D-019:** (a) is justified. I re-derived the oracle's Proposition 1 counterexample (`f(a=1, b=2)` for `{"a":1,"b":2}` and `{"a=1, b":2}`). Ours stays invariant; the cost is two entries sharing a display name (Q10). (b) An error instead of reading a missing `obs` as null.
- **Q7–Q12** remain genuine SPEC questions for the human. Q11 should be enforced in code until it is answered (finding 2).

## Checks that passed, with evidence

**§2 Canonical form**
- `canon` (`canonical.py:20-52`): integral floats below 2^53 become ints, -0.0 becomes 0, NaN and ±inf raise (31-32), NFC strings and keys, keys sorted by code point, NFC collisions (46-47) and non-string keys raise.
- `canon_json` (`canonical.py:55-57`) is exactly the SPEC call.
- The tool-first action string (D-001) is at `canonical.py:60-70`; `show_call` is at `canonical.py:73-81`.
- Evidence: a 3000-value oracle differential (`tests/core/test_canonical.py:126-131`) and the acceptance canonical fixtures.

**§3 Contracts**
- A tool without a contract gets `EVERYTHING` (`contracts.py:33, 114-116`).
- A missing template argument becomes `*` (`contracts.py:81-87`). String values are substituted verbatim and other values as canon_json (`contracts.py:90-91`).
- Only `*` is a glob (`contracts.py:143-161`; 100 000-pair fuzz above). Two patterns meet when either glob-matches the other (`contracts.py:164-166`).
- The Bernstein condition is symmetric (`contracts.py:173-175`).
- Volatile fields are deleted from a fresh canonical copy, and the input is not mutated (`contracts.py:118-125, 133-140`).

**§4 Compile**
- Compile and Reduce use the same conflict function: `compile.py:20` and `reduce.py:16` both import `contracts.conflict`.
- Compile never reads a ground-truth DAG. Its only inputs are request, steps, contracts, initial observation, task order and start values (`compile.py:77-85`); there are no model or environment calls.
- The five rules match SPEC §4 exactly:
  - data, from every producer or else from every earlier call: `compile.py:98-103`;
  - effect: 106-107;
  - decision: 104 and 108-109;
  - dup: 110-111;
  - task, with forward validation: 112-113 and 68-74.
- Labels are sorted (114), and edges point forward, so the demo order is topological.
- Uncertain cases add edges: no contract, an underived value, a missing template argument.
- D-005 start values (`compile.py:32-55`) match the oracle: 500-request differential at `tests/core/test_compile.py:62-73`.
- Compile edges and canonical orders are identical to the oracle on 600 + 800 + 4000 random demos. The compile-fixture state and order counts match in all 16 cases.

**§5 Reduce and Render**
- Reduce takes the smallest-key topological order under conflict constraints only (`reduce.py:91-101, 65-88`), with key `(tool, canon_json(args), k)` and names `show_call` + `#k` (`reduce.py:43-47`).
- The prompt depends only on (task, H). `render.py:134-160` builds lines only from task fields, entry names and canonical observations. There are no step numbers, dates, timestamps or run ids; the only numbering is the chronological `1.`, `2.` that SPEC §5.7 requires (`render.py:155`).
- Every step is re-rendered in full: `render` takes the whole history and keeps no state.
- Truncation is content-based and applied before both sections (`render.py:68-81, 130`).
- One module builds every prompt (`render`, `render_messages`); the three variants share `_entries` and `_task_lines`.
- Iteration is sorted everywhere (`render.py:92, 109`, and canon's sorted keys). Hash-seed invariance was confirmed above.
- Invariance (§5.6 a, §9.10) holds on:
  - 300 graphs × 20 orders × every prefix (`tests/core/test_reduce_render.py:266-289`);
  - 300 × 10 rich cases (`tests/core/test_fuzz.py:117-137`);
  - 4000 × 6 cases here;
  - all 24 legal orders of the seven-step fixture.

**§6 Lattice**
- States are enumerated breadth-first with a cap: 256 states stay exact and 257 switch to the sub-lattice (`lattice.py:64-79`; `tests/core/test_lattice.py:88-99`).
- μ (`lattice.py:223-240`): n ≤ 1 gives μ(∅) = 1 (231), and layer counts are taken over the states in use (233). λ in [0, 1) is validated.
- The recursion is in log space and float64, from the largest states down, with `logsumexp` using `fsum` (`lattice.py:244-249, 263-280`).
- Successors outside S count 0 for B_S and 1 for B⁺ (`lattice.py:275-278`).
- Explicit S is validated (`lattice.py:101-125`).
- Sub-lattice: the demo order plus K−1 Kahn orders, chosen uniformly from the ascending ready list, seeded by sha256(graph id, epoch, run seed). The seed does not depend on the configuration (`lattice.py:164-192, 174-178`).

**§6.4 and §8 Configurations**
- One weight function serves every configuration (`weights.py:97-161`).
- The flow runs from small states up, and the finish weight is m(V) (`weights.py:40-55`, line 54).
- Weights by configuration:
  - qet: ρ = m·q (`weights.py:144-150`);
  - uniform_aug: m·c(I+v)/c(I) with exact integer counts (`weights.py:58-63, 136-140`);
  - frontier: m·π̄, the only renormalization, kept inside its weights (`weights.py:152-158`);
  - path_sft: 1 on the edges of the given order (115-118);
  - sampled_paths: deduplicated path posterior (85-95, 119-132).
- Core never length-normalizes log π.
- Logged objectives are the true ones, never the pass-2 loss for qet (`weights.py:175-197`): Σμ(−log B) for qet; Σ m·ℓ + end term for frontier; −log Σ P over the sampled paths; weighted NLL for path_sft and uniform_aug.
- Configurations 3–6 share one `Lattice` (same states and edges). The finish weight is 1 for every method.

**§9 Identities**
- Items 1–9 hold on 300 DAGs with n ≤ 8, at SPEC tolerances (`tests/core/test_identities.py`: rel 1e-12; finite differences abs 1e-7). The slow finite-difference test passes. Sub-lattice finite differences pass (above).

**§10 Facade**
- Signatures and defaults match (`tests/core/test_api.py:22-39`). Outputs are plain JSON. All 134 reference tests pass.

**Hygiene**
- Production code never imports the oracle (acceptance test, and a grep of `src/qet/core` imports).
- `reference/`, `docs/SPEC.md`, `docs/EVAL_PLAN.md`, `third_party/` and `.claude/` are unchanged since `kit`.
- No test gained a skip or xfail, a looser tolerance or fewer samples.

**Not reviewed (E1):** pass 1 / pass 2, eval mode and `no_grad`, LoRA dropout, token counters, chat templates and `date_string`, and the scorer. Notes 6–8 cover the core APIs these will call.

## Response to the findings (implementing agent, not the reviewer; commit 73e7413)

| # | Response |
| --- | --- |
| 1 | Edges unchanged (SPEC §4 rule 1, as in the oracle). Diagnostic added: `compile.underived` in every graph record and `underived_values` in the compile report (D-016e). The start-value policy is a question for the human (Q13); D-005 got a pointer note in f11fbce. Test: `test_underived_values_are_reported_without_changing_the_edges`. |
| 2 | `compile_demo` raises `NotImplementedError` for a non-null `initial_observation` until Q11 is decided (D-016f); the fixture facade `compile_edges` still accepts it. Test: `test_graph_records_refuse_an_initial_observation_until_q11_is_decided`. |
| 3 | Segment matcher (anchored prefix and suffix, middle segments with `str.find`); D-015(a) wording corrected. Tests: `test_long_patterns_and_values_stay_fast`, the 20 000-pair DOTALL-regex differential. |
| 4 | Recorded as Q14; no code change, so the implementation keeps agreeing with the oracle and the fixtures. |
| 5 | Recorded as Q15. |
| 6 | `edge_logp` raises for NaN, positive values and +inf; -inf is allowed (D-018h). Test: `test_log_probabilities_must_not_be_positive`. |
| 7 | `log_b` and `mu` optional in `logged_objective` (completed in f11fbce, see below). Test: `test_logged_objective_needs_log_b_and_mu_only_for_qet`. |
| 8 | Recorded as D-020(b) for E1.4 and the adapters. |
| 9 | Only canonical state and edge keys are accepted (D-018h). Test: `test_facade_keys_must_be_canonical`. |
| 10 | `width` uses iterative augmenting paths. Test: `test_width_matches_a_recursive_matching_on_larger_dags_and_scales`. |
| 11 | Recorded as D-020(a), proposed for the M1.3 compile audit. |
| 12 | `tests/core/test_fixture_extras.py` compares `identities.json` (eq. 4, eq. 5, the chronological counterexample), `render.json` targets per state and `compile.json` state and order counts with `src/qet`. `toy_training.json` stays with E1.3(b). |

After the response: `make lint` exit 0; `make test` 291 passed; `python3 -m pytest reference -q` 134 passed.

## Follow-up review of the response (commit 73e7413)

- **Commit reviewed:** `73e7413` (`E0.7: address final review …`), compared against `ce7ef3e`. Working tree clean.
- **Reviewer:** spec-reviewer subagent, fresh context.
- **Scope:** `git diff ce7ef3e 73e7413 -- src tests docs/DECISIONS.md`. This covers:
  - `src/qet/core/{compile,contracts,lattice,weights}.py`;
  - `tests/core/{test_compile,test_contracts,test_lattice,test_fixture_extras}.py`;
  - D-015(a), D-016(e, f), D-018(h), D-020, Q11 and Q13–Q15 in `docs/DECISIONS.md`.

  It was read against `docs/SPEC.md` §3, §4, §6, §8 and §10, and against `reference/qet_ref.py`.

**Verdict: no blocking findings.** The response fully resolves 10 of the 12 findings. Findings 1 and 7 are only partly resolved. The change made for finding 7 introduces one new should-fix (N1), and there are three new notes.

### Commands and results

| Command | Result |
| --- | --- |
| `make lint` | `All checks passed!` and `59 files already formatted`; exit 0 |
| `make test` | `291 passed in 39.12s`; exit 0. This includes the slow finite-difference test. |
| `make test-fast` | `290 passed, 1 deselected in 15.83s`; exit 0 |
| `python3 -m pytest reference -q` | `134 passed in 2.97s` (only passed) |
| `python3 -m pytest tests -q -m slow` | `1 passed, 156 deselected in 21.42s` |
| `python3 reference/make_fixtures.py --check` | `fixtures up to date -> .../reference/fixtures`; exit 0 |
| `git diff --stat kit -- reference/` | empty (`kit` = `1557732 Agent kit`) |
| `git diff --stat kit HEAD -- docs/SPEC.md docs/EVAL_PLAN.md third_party .claude` | empty |
| Lines removed by `git diff ce7ef3e 73e7413 -- tests reference` | One: the `from qet.core.weights import …` line, re-added with `logged_objective`. |
| `git grep -c '^def test_'` at `ce7ef3e` and at `73e7413` | 136 → 148 test functions: +3 in `test_compile.py`, +1 in `test_contracts.py`, +3 in `test_lattice.py`, +5 in the new `test_fixture_extras.py`. Pytest items go from 279 to 291. |

No test gained a skip, an xfail or a looser tolerance. The only new `skip` is a `# fmt: skip` formatter comment.

Under pytest, every `qet.*` module loads from the worktree's `src`, not from the editable install of the main checkout. I checked this with a pytest plugin.

### Ad-hoc checks

The scripts are throwaway files in `/tmp/claude-1304605107/-home-wei-zhuo-My-Proj-QET/7691d9a6-1b16-4c03-bbdf-175d7ef6ae6d/scratchpad/fu/` and were not committed. Each one asserts that `qet.__file__` is under the worktree's `src`. "Old" means the `ce7ef3e` tree, imported as `qet_old`.

| Check | Result |
| --- | --- |
| `glob_match` against `re.fullmatch` with DOTALL, and against the old matcher. Every pattern over {`a`, `b`, `*`, `\n`} up to length 6, times every string over the same alphabet up to length 5. | 7,454,265 pairs; 0 mismatches for the new matcher, 0 for the old. |
| `glob_match`: 300,000 structured random pairs (repeated and overlapping segments, `**`, NFC and NFD `é`, newlines at segment joins). Hand cases: `a*a`/`a`, `ab*ba`/`aba`, `*a*a*`/`a`, `\n*\n`/`\n`, `x*aba*aba*y`/`xabababy`. | 0 mismatches. Every hand case agrees with the regex. |
| `patterns_meet`, new against old, on all pairs of patterns up to length 4 | 116,281 pairs, 0 differences |
| Speed of the new matcher on near-miss middle segments: `a^49 b a^49` in `a^10^6`; `a^999 b` in `a^10^6`; 5000 one-character segments | 3.4 ms, 3.0 ms and 0.5 ms. The old worst case (`*a^3000b` against `a^6000`) now takes under 0.01 ms. |
| `width` against a brute-force largest antichain: 3000 DAGs with n ≤ 12, half with shuffled labels, edge densities 0 to 1 | 0 mismatches |
| `width` against the old recursive version: 1500 DAGs with n from 13 to 150; crowns with k ≤ 400; staircases with k ≤ 700, which force long augmenting paths and so exercise the flips | 0 mismatches. Random graphs with n = 1500 and n = 3000 take ≤ 0.7 s. |
| Compile edges, new against old against oracle, on 11,000 demos: 3000 from `random_demo`; 4000 from `test_fuzz.random_case` (unknown tools, null contracts, duplicate calls, glob and missing-argument resources, non-identifier keys, empty and error observations); 4000 dominated by constant arguments | Edges identical in all three for every demo |
| `underived` against an independent recomputation from oracle helpers | Equal on all 11,000 demos. Every listed `(v, value)` has `data` edges from all earlier calls, and v = 0 is never listed. |
| `compile_demo` on the 8909 cases without an initial observation | Edges equal `compile_edges`; `underived` equals `_analyze`; the JSON round trip is exact; the `compile_report` totals, graph counts and top-20 list are correct. |
| Guard: `initial_observation` set to `None`, `{}`, `[]`, `0`, `False`, `""`, `{"inbox":[{"id":42}]}` or `"x"` | Only `None` (or an absent key) is accepted. All 2091 random cases with an initial observation are refused. The facade still compiles the `compile.json` case. |
| Fixtures: keys of every edge-keyed object (2490) and state-keyed object (1888), and every `logp` value (344) | All keys are canonical and every value is ≤ 0. All 52 `lattice.json` runs give byte-identical facade output before and after the change. |
| Key parsers on 150,000 exhaustive and random strings (digits, commas, `\|`, spaces, signs, `_`, non-ASCII digits, `end`/`End`) | Exactly the 185 canonical keys are accepted, and each round-trips. The rest raise `ValueError`; the old parser accepted 1746 of them. |
| The previous review's failing inputs, replayed | Finding 1: still a chain, and now reports `underived ((1,"true"),(2,"true"))`; with `start_values=[True]` there are no edges; `null`, `10` and `"en"` behave like `true`. Finding 6: negated log π → `ValueError`; +inf → `ValueError`; −inf is accepted and the tool weights sum to 2.25. Finding 9: `"1,0\|2"` → `ValueError`. Finding 10: a 1500-node chain has width 1. |
| n = 0, n = 1, and 10 independent nodes (above S_max) | Widths are 0, 1 and 10. With an undeclared `verbose=false` the 10 nodes compile to a chain with 9 underived entries; the report buckets the independent graph as `>256`. For n ≤ 1, `logged_objective` equals `summarize` for every λ. |
| `PYTHONHASHSEED` in {0, 1, 2, 12345, random} | Identical sha256 over edges, `underived`, widths, records and the compile report |

### Response to findings 1–12

| finding | resolution | adequate? |
| --- | --- | --- |
| 1 (underived values, D-005 wording) | Diagnostic added:<br>- `_analyze` records `(v, canon_json(value))` whenever rule 1 falls back to "every earlier call" (`src/qet/core/compile.py:120-130, 142`).<br>- It is stored in `CompileInfo.underived` (`compile.py:236, 244, 256`).<br>- The report gains `underived_values` (`compile.py:412-413, 427-428, 444-448`).<br>- Test: `tests/core/test_compile.py:329`.<br>- Recorded as D-016(e) (`docs/DECISIONS.md:130-134`) and Q13 (`docs/DECISIONS.md:216-221`). | **Partly.** The diagnostic and Q13 are correct, and I verified that edges are unchanged. But D-005 itself (`docs/DECISIONS.md:53-58`) still gives the understated wording and does not point to Q13 (N2). |
| 2 (`initial_observation` guard) | `compile_demo` raises `NotImplementedError` for any non-null value (`compile.py:359-364`); the facade `compile_edges` still accepts it. Test: `test_compile.py:348`. D-016(f) and Q11 updated (`docs/DECISIONS.md:134-136, 208-213`). | **Yes.** The guard also refuses `{}` and `[]`, which is stricter than needed but safe. Two small gaps remain (N3). |
| 3 ("linear-time" glob) | Replaced by a segment matcher: anchored prefix and suffix, then left-to-right `str.find` (`src/qet/core/contracts.py:143-160`). The D-015(a) wording is corrected (`docs/DECISIONS.md:105-108`). Speed test: `tests/core/test_contracts.py:148`. | **Yes.** Exhaustively equal to a DOTALL regex, including empty segments, overlapping prefix and suffix, and newlines. No slow inputs found. |
| 4 (NFD request) | Q14 (`docs/DECISIONS.md:222-224`) | **Yes**, as an open question for the human. |
| 5 (finish name vs system text) | Q15 (`docs/DECISIONS.md:225-227`); D-016(d) points to it. | **Yes**, as an open question. |
| 6 (sign of log π) | `edge_logp` raises on NaN and on any x > 0, +inf included; −inf is allowed (`src/qet/core/lattice.py:258-259`). Recorded as D-018(h). Test: `tests/core/test_lattice.py:223`. | **Yes.** No fixture value is rejected. |
| 7 (`logged_objective` required `log_b`/`mu`) | `log_b` and `mu` are now optional; for qet they are computed from `lam`, which defaults to 0.5 (`src/qet/core/weights.py:175-192`). Test: `test_lattice.py:244`. | **Partly.** The `TypeError` is gone, but the λ default can silently log the wrong objective for configuration 5 (N1). |
| 8 (NaN, inf and NFC collisions at inference) | D-020(b) (`docs/DECISIONS.md:181-184`), deferred to E1.4 and M1 | **Yes**, as a recorded deferral. |
| 9 (non-canonical facade keys) | `parse_state_key` and `parse_edge_key` accept only the canonical spelling (`lattice.py:303-319`). Recorded in D-018(h). Test: `test_lattice.py:231`. | **Yes.** It can no longer happen that two different keys map to the same edge. |
| 10 (recursive `width`) | Iterative Kuhn augmenting paths (`compile.py:180-208`). Tests: `test_compile.py:353-381`. | **Yes.** Checked against brute force and against the old recursive version, crowns and staircases included. |
| 11 (`task_order` as a channel for the true DAG) | D-020(a) (`docs/DECISIONS.md:178-181`), proposed as an M1.3 audit item | **Yes**, as a recorded deferral. |
| 12 (fixture coverage) | `tests/core/test_fixture_extras.py`: eq. 4 (both cases), eq. 5, the chronological counterexample, the targets of all 5 states, and the state and order counts of all 16 compile cases. Float comparisons use rel 1e-12. | **Yes.** `toy_training.json` is left to E1.3(b), as suggested. |

### New findings

| # | Severity | Location | What is wrong | Concrete failing input | Suggested fix |
| --- | --- | --- | --- | --- | --- |
| N1 | should-fix | `src/qet/core/weights.py:180, 190-192` | `logged_objective(..., lam=0.5)`: for qet with `mu` omitted, μ is rebuilt with λ = 0.5 whatever λ the weights used, because `EdgeWeights` does not record λ. SPEC §8 says configuration 5 (λ = 0) logs −log B(∅); it would silently log Σ μ₀.₅·(−log B) instead. SPEC §6.5 and AGENTS.md require logging the true objective. The new test only uses λ = 0.5. | On the running example, `ew = edge_weights("qet", lat, lp, lam=0.0)`, then `logged_objective(ew, lat, lp)` returns 0.655310, while −log B(∅) = `summarize("qet", lat, lp, lam=0.0).objective` = 0.836556. On 400 random DAGs, 313 differ for λ = 0 and λ = 0.9 (every case with n ≥ 2). On a 38-state sub-lattice: 10.951613 against 13.765926. | Record λ (or μ) in `EdgeWeights` for the flow-based methods and use it, raising if a passed `lam` disagrees. Or drop the default (`lam: float \| None = None`) and raise for qet when neither `mu` nor `lam` is given. Add a λ = 0 test. |
| N2 | note | `docs/DECISIONS.md:53-58` (D-005) | D-005 is a Proposed entry, so it may be edited, yet it still only says "a boolean argument equal to an earlier output gets a data edge". Q13 records the larger effect, but D-005 neither states it nor points to Q13. | Reading D-005 alone, nobody would expect three `get_weather(city=X, verbose=true)` calls to compile to a 4-state chain. | Add one sentence to D-005: any argument value that is neither available from the start nor in an earlier observation depends on every earlier call; see Q13 and D-016(e). |
| N3 | note | `src/qet/core/compile.py:346-350` (docstring); `compile.py:298-316` (`GraphRecord.from_json`) | The docstring still lists "optional initial_observation" as part of `task`. `from_json` and `read_records` load a record with a non-null `initial_observation` without the Q11 guard. No records exist on disk yet (`results/` is empty), so nothing is affected today. | `GraphRecord.from_json({..., "initial_observation": {"inbox": [{"id": 42}]}, ...})` loads without error. | Fix the docstring. Until Q11 is decided, refuse a non-null `initial_observation` where E1 builds lattice packs from records, or in `from_json`. |
| N4 | note (facade only) | `src/qet/core/lattice.py:306-310` | The canonical check builds the bitmask before checking the text, and `state_key` → `members` then shifts once per bit. Cost is quadratic in the largest node id. Fixture ids are below 16, so nothing is affected. | `parse_edge_key("1000000\|end")` takes 10.3 s; the old parser took under 1 ms. | Optional: check the text first (ascending ASCII integers, no leading zeros) before shifting, or bound ids by n in `lattice_summary`. |

Outside the diff: on this node `nvidia-smi -L` prints `No devices found.` but exits 0. The Makefile's `HAVE_GPU` therefore comes out set, and `make test` drops `-m "not gpu"`. This is harmless while no test has a `gpu` mark (there are none today), but it will matter once E1 adds GPU tests.

**Verdict: no blocking findings.** Fix N1 (the λ default in `logged_objective`) before E1 uses it to log configuration 5; N2–N4 are notes.

## Response to the follow-up findings (implementing agent, not the reviewer; commit f11fbce)

| # | Response |
| --- | --- |
| N1 | `EdgeWeights` records the λ of its start distribution (flow-based methods), and `logged_objective` builds μ from it unless `mu` is given; its `lam` parameter is gone, and qet weights that do not record λ raise. Configuration 5 now logs −log B(∅) (D-018i). Test: `test_logged_qet_objective_uses_the_lambda_of_its_weights` (running example at λ = 0, and 100 random DAGs at λ = 0, 0.5, 0.9). |
| N2 | D-005 got a note pointing to Q13 and D-016(e). |
| N3 | Docstring fixed; `GraphRecord.__post_init__` refuses a non-null `initial_observation`, so records read from disk are guarded too (D-016f). Test extended. |
| N4 | Keys are checked as text first (strictly ascending ASCII ids without leading zeros), then converted. Test: `parse_edge_key("1000000\|end")` in under 0.5 s. |
| GPU | The Makefile looks for a `GPU <i>:` line in `nvidia-smi -L`; on this node `make -n test` now passes `-m "not gpu"`. |

After the response: `make lint` exit 0; `make test` 292 passed; `python3 -m pytest reference -q` 134 passed.

## Second follow-up review (commit f11fbce)

- **Scope:** `git diff 73e7413 f11fbce`: 7 files, +66/−24.
  - Files: `Makefile`, `src/qet/core/{weights,compile,lattice}.py`, `tests/core/{test_lattice,test_compile}.py`, and the D-005 note, D-016(f) and D-018(i) in `docs/DECISIONS.md`.
  - Checked against `docs/SPEC.md` §6.3, §6.6, §8 and §10, and against `reference/qet_ref.py`.
- **Reviewer:** spec-reviewer subagent, fresh context.
- **Ad-hoc scripts:** throwaway files in the session scratchpad, not committed. Each one put the worktree's `src` first on `sys.path` and asserted `qet.__file__`.
- **Code under test:** the main checkout's `src` is byte-identical to the worktree's (`diff -rq`), so the pytest runs tested this commit.

### Commands and results

| Command | Result |
| --- | --- |
| `make lint` | `All checks passed!`, `59 files already formatted`; exit 0 |
| `make test` | `292 passed in 38.13s`; exit 0 |
| `make -n test` | `python3 -m pytest tests reference -q -m "not gpu"`. On this node `nvidia-smi -L` prints `No devices found.` and exits 0. |
| `make test-fast` | `291 passed, 1 deselected in 16.27s` |
| `python3 -m pytest reference -q` | `134 passed in 3.05s` (only passed) |
| `git diff --stat kit -- reference/` | empty. Also empty for `docs/SPEC.md`, `docs/EVAL_PLAN.md`, `third_party` and `.claude`. |
| `pytest --co` at 73e7413 vs f11fbce, parametrized ids included | 291 → 292. The only new id is `test_logged_qet_objective_uses_the_lambda_of_its_weights`, and none was removed. 73e7413 itself gives `291 passed`. |
| New `tests/core/test_lattice.py` run against 73e7413's `src` | `2 failed, 20 passed`. The N1 test fails (`EdgeWeights` has no `lam`), and so does the key test (timing bound). So the new tests catch the old bugs. |

**Ad-hoc checks**

- **N1 differential:** 18 000 checks, 0 mismatches, worst relative error 5.4e-16.
  - Setup: 600 random DAGs with n = 0..7, about half on sub-lattices of 1–4 orders. λ ∈ {0, 1e-12, 0.25, 0.5, 0.9, 0.999999}. All 5 methods.
  - Compared `logged_objective(edge_weights(...), lat, logp)`, with no `mu` and no `log_b`, against the oracle's `summary(...)["objective"]`.
  - For qet with λ = 0 the result equals −log B_S(∅) on every lattice.
  - All 52 `lattice.json` runs through the same path: 0 mismatches. The fixtures have qet λ = 0 only on the full lattice; the random check adds sub-lattices.
- **N3:** records without an initial observation still work:
  - `to_json`/`from_json` round trips, also with the key absent;
  - `fallback_to_chain`, applied twice;
  - `dataclasses.replace`;
  - JSONL write and read;
  - `compile_report`.

  A non-null value is refused by both `compile_demo` and `from_json`, including `{}`, `[]`, `0`, `False` and `""`. `replace(rec, initial_observation=…)` is refused too.
- **N4:** 300 000 random keys built from digits, `,`, `|`, `end`, spaces, tabs, newlines, signs, `_`, `.`, `0x`, `1e3`, U+0661, U+FF11, U+00B2 and U+200B.
  - A key is accepted exactly when it is canonical: 0 non-canonical keys accepted, 0 canonical keys rejected, 0 round-trip failures.
  - 20 000 canonical keys (n ≤ 70) round-trip.
  - All 4 354 keys in `lattice.json`, `identities.json` and `toy_training.json` parse and round-trip.
  - Parse time: a 30 000-id state key takes 0.023 s, and `5000000|end` takes under 1 ms.
- **Makefile:** a fake `nvidia-smi` first on PATH, then `make -n test`:
  - a `GPU 0: …` line, or a MIG listing: no marker filter;
  - `No devices found.`, a driver failure (exit 9), or no `nvidia-smi` at all: `-m "not gpu"`, with nothing printed on stderr.

### Items from the first follow-up review

| item | resolved? | evidence |
| --- | --- | --- |
| N1: logged qet objective used λ = 0.5 | yes | `src/qet/core/weights.py:38` adds the `lam` field. `:162` records it for uniform_aug, frontier and qet. `:191-195` uses it when `mu` is None and raises when neither is known. Tests: `tests/core/test_lattice.py:258-272`. The differential above covers configurations 3–6 for every λ, including λ = 0, n = 0 and 1, and sub-lattices. One silent path is left: an explicit `mu` that does not match (note 1). |
| N2: D-005 did not point to Q13 | yes | `docs/DECISIONS.md:59-61` |
| N3: docstring; `from_json` had no Q11 guard | yes | The guard is in `GraphRecord.__post_init__` (`src/qet/core/compile.py:274-278`), so `from_json` (`:305-322`), `replace` and `fallback_to_chain` (`:389-392`) all pass through it. Docstring: `:354-355`. D-016(f): `docs/DECISIONS.md:137-139`. Test: `tests/core/test_compile.py:351-353`. Nothing breaks for records without an initial observation (ad-hoc N3 check). |
| N4: key parsing quadratic in the largest id | yes | `src/qet/core/lattice.py:19, 305-322` checks the text before converting: ASCII `0\|[1-9][0-9]*` ids, strictly ascending. Only the canonical spelling is accepted, and no fixture key is rejected. Tests: `tests/core/test_lattice.py:236-245`. The cost still grows with the id's value (note 2). |
| Makefile: `HAVE_GPU` set on GPU-less nodes | yes | `Makefile:5` requires a `^GPU [0-9]` line. On this node `make -n test` passes `-m "not gpu"`. See also the fake-PATH cases above. |
| No test weakened | yes | The only removed lines are the import line (extended), the bad-key tuple (extended), and `lam=0.5` dropped from a call whose keyword no longer exists. No skip, xfail or tolerance change. 291 → 292 tests. |

### New findings

| # | severity | file:line | what is wrong | concrete failing input | suggested fix |
| --- | --- | --- | --- | --- | --- |
| 1 | note (E1 hazard) | `src/qet/core/weights.py:181-195` | `logged_objective` uses an explicit `mu` without checking it against the λ the weights record. The same holds for `log_b`, and `lat` is not checked against the weights' lattice. Example: if a trainer reuses its λ = 0.5 `qet_objective` μ for configuration 5, it logs L(λ = 0.5) instead of −log B(∅), and nothing reports it. | SPEC running example, `ew = edge_weights("qet", lat, lp, lam=0.0)`. `logged_objective(ew, lat, lp)` gives 0.836556 = −log 0.4332, which is correct. `logged_objective(ew, lat, lp, mu=start_weights(lat, 0.5))` gives 0.65531 with no error. | Raise when `mu` is given and differs from `start_weights(lat, ew.lam)`, or drop the `mu` parameter. Optionally require `set(ew.weights) == set(lat.edges)`. |
| 2 | note (facade only; pre-existing) | `src/qet/core/lattice.py:305-322`; `src/qet/core/api.py:100` | Facade key ids are not bounded by n. `1 << id` is built for any id, so a short key can cost a lot of memory. Keys that name no edge are silently ignored. | `parse_edge_key("1000000000\|end")` takes 0.15 s and +254 MB RSS; an 11-digit id would need a 12.5 GB int. `lattice_summary(1, [[]], {"\|0": -0.1, "0\|end": -0.2, "0\|0": -0.3, "5\|end": -0.4}, "qet", 0.5)` returns 0.3 and ignores the last two keys. | In `lattice_summary`, reject keys with an id ≥ n, or whose node is already in the state, before building bitmasks. |
| 3 | note | `Makefile:5` | `nvidia-smi` lists GPUs whatever `CUDA_VISIBLE_DEVICES` says, so a job that hides its GPUs this way still selects gpu tests. None exist yet, and a failure would be loud. | `CUDA_VISIBLE_DEVICES=""` on a GPU node | Also require `CUDA_VISIBLE_DEVICES` to be unset or to name a device, or detect with torch once torch is a dependency. |

**Verdict:** no blocking findings. N1–N4 and the Makefile remark are resolved. There are three new notes, and none of them affects a current result or test.

## Response to the second follow-up notes (implementing agent, not the reviewer)

All three notes concern code that E1 writes (the trainer that logs objectives, GPU tests) or the test facade, and none
affects a current result or test. They are recorded as guards for E1 in `docs/DECISIONS.md` D-020(c)-(e) rather than
changed here, so `src/qet/core` stays as reviewed at f11fbce.

## Status at the close of E0.7

Five spec reviews of `src/qet/core` (E0.2 at a74f5c4, E0.3-E0.6 at 19f5d54, the end-of-E0 review at ce7ef3e, and the
two follow-ups at 73e7413 and f11fbce) report **no blocking findings**; there is **no open blocking finding**. Open items
for the human are the Proposed decisions D-015 to D-020 and the questions Q1 and Q7 to Q15 in `docs/DECISIONS.md`.

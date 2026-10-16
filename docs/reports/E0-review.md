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

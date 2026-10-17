# Decisions and open questions

Agents add entries as **Proposed** (options, default, reason) and may proceed on the default only when it is
cheap to reverse. Only the human marks an entry **Accepted** or **Rejected**. Never rewrite an accepted entry;
supersede it with a new one. Changes to SPEC.md, EVAL_PLAN.md or reference/ need an accepted entry first.

| Id | Status | Topic |
| --- | --- | --- |
| D-001 | Proposed — needs confirmation | Action string: tool first |
| D-002 | Proposed | Chat mapping and end token |
| D-003 | Proposed | English template text |
| D-004 | Proposed | Baseline renderers |
| D-005 | Proposed | Values available from the start |
| D-006 | Proposed | Invalid actions end the episode |
| D-007 | Proposed | FuncBenchGen demonstration order |
| D-008 | Proposed | Frontier supervision state weights |
| D-009 | Proposed | Training stack and default hyperparameters |
| D-010 | Proposed | Format warm-up |
| D-011 | Proposed | Sampled-order marginalization ablation |
| D-012 | Proposed | Uniform augmentation by exact completion counts |
| D-013 | Proposed | Shared sub-lattice across configurations |
| D-014 | Proposed | Loss normalization per graph |
| D-015 | Proposed | Resource glob semantics and contract validation |
| D-016 | Proposed | Graph record and compile report details |
| D-017 | Proposed | Renderer details left open by SPEC §5 |
| D-018 | Proposed | Lattice and weight details left open by SPEC §6, §8 |
| D-019 | Proposed | Repeat numbering per identical call; steps carry an observation |
| D-020 | Proposed | Guards for later milestones (task constraints, inference canonicalization) |
| D-021 | Proposed | Compute and access policy until the human sets a budget |
| D-022 | Proposed | Chat details left open by SPEC §6.7 and D-002 |

### D-001 Action string: tool first
The method page defines the canonical call string by sorting all JSON keys, which puts `args` before `tool`
(`{"args":{…},"tool":"get_flight_price"}`), while its own system prompt describes `{"tool": …, "args": {…}}`.
Default: tool first, args keys sorted (`{"tool":"get_flight_price","args":{"from":"NYC","to":"SG"}}`), so the
first decision token is the tool name and the system prompt matches the target. No QET quantity depends on it;
all configurations use the same strings. Alternative: the method page's literal form (`ACTION_KEY_ORDER = "sorted"` in the
oracle, then regenerate fixtures). Decide before E0.2.

### D-002 Chat mapping and end token
system message = Render's system text; one user message = the rest of the prompt; assistant target = action
string followed by the tokenizer's end-of-turn token (Qwen2.5 `<|im_end|>`, Llama 3.1 `<|eot_id|>`), which plays
the role of the method page's `<END>` and is included in log π. Always pass an explicit system message (Qwen otherwise inserts
a default one). Llama 3.1: pass a fixed `date_string` so the template's date line never changes.

### D-003 English template text
The prompt text in SPEC §5.2 replaces the Chinese example of the method page; benchmarks are English. Pinned by
`render.json`.

### D-004 Baseline renderers
Configuration 1 uses the `chronological` renderer (same template, numbered history in execution order, same
restated values). Ablation a1 uses `native_multiturn` (standard chat history, no restated values). This gives
a ladder: native chat → chronological with restated values → quotient interface.

### D-005 Values available from the start
Standalone numeric tokens and whole-word strings of the request, leaves of the initial observation, and
adapter-declared start values (for example enum constants from tool schemas, with the list recorded in the
compile report). Booleans and null are not automatically available, so a boolean argument equal to an earlier
output gets a data edge (conservative). Adapters may raise the minimum string length for request matching if
audits show spurious matches.
Note added after the E0.7 review: by SPEC §4 rule 1, any argument value that is neither available from the start
nor found in an earlier observation (a flag, null, a default, a code) depends on every earlier call; see Q13 and
the compile report's underived values (D-016e).

### D-006 Invalid actions end the episode
An output that is not a JSON object with exactly `tool` and `args`, names an unknown tool, or has non-object
args ends the episode as a failure counted as `invalid_action`. Same for every system.

### D-007 FuncBenchGen demonstration order
The demonstration for a training graph is a random topological order of the generator's true DAG (seeded per
graph), executed in the environment to record observations. Compile then sees only the trace. This mimics a
demonstrator's accidental order without leaking the DAG into Compile.

### D-008 Frontier supervision state weights
Configuration 4 weights edge (I, v) by m(I) · π̄(v | I), where π̄ renormalizes the model over ready actions and
m is the occupancy of π̄ started from μ_G (λ = 0.5). Its objective is then the μ-weighted Jensen bound of the
method page's eq. (5), and its support and starts match configuration 6.

### D-009 Training stack and default hyperparameters
Hugging Face transformers + PEFT LoRA (rank 64, alpha 128, dropout 0, all linear layers), bf16, gradient
checkpointing, a custom two-pass loop (no TRL trainer), AdamW lr 1e-4 with 3% warm-up and cosine decay, vLLM
for inference with prefix caching. Tuning budget per configuration: the same small grid on dev (for example
lr ∈ {5e-5, 1e-4, 2e-4}). Revisit at Gate A with throughput numbers.

### D-010 Format warm-up
Before each configuration, a fixed number of demo-order SFT steps in that configuration's own renderer
(configurations 2–6 share one warm-up checkpoint per backbone and seed). Warm-up tokens count toward the budget.
Default length: until format errors on dev fall below 1%, measured once in the pilot, then fixed for all.

### D-011 Sampled-order marginalization ablation
a3: from the root only, the demo order plus K−1 random topological orders (default K = 4), deduplicated; the
objective is −log Σ_k P(σ_k) over exactly those paths (paths that cross at a shared state are not recombined,
unlike the sub-lattice, whose B_S counts every path inside S); weights are the path posterior. It isolates the
value of recombination in the lattice from marginalizing over a few sampled orders.

### D-012 Uniform augmentation by exact completion counts
Configuration 3 uses the exact expected weights of uniform sampling over complete topological orders from each
start state of μ_G (λ = 0.5), via completion counts, instead of sampling orders. Same states and starts as
configuration 6, no sampling noise.

### D-013 Shared sub-lattice
When a graph exceeds S_max, configurations 3–6 and a4 use the same sampled sub-lattice S for that graph and
epoch (seeded), so their states and edges stay identical.

### D-014 Loss normalization per graph
Pass-2 losses are summed over target tokens and weighted edges, then averaged over the graphs in the batch (the
expectation in eq. 2), for every configuration. Token-averaged losses would weight graphs by their size.

### D-015 Resource glob semantics and contract validation
`src/qet/core/contracts.py`. (a) `*` in a resource pattern matches any string, newlines included, and nothing
else is special. The matcher anchors the text before the first `*` and after the last one and finds the segments in
between left to right with `str.find`, so nothing backtracks and model-generated arguments full of `*` cannot blow
up (an earlier backtracking matcher was O(len(pattern)·len(s)) at worst, not linear as first written here). The oracle's regex `.*` stops at a newline, so in the oracle even the universal `*` of a tool
without a contract (or of a template with a missing argument) fails to meet a resource that contains a newline,
which contradicts SPEC §3. The two differ only for resources with a newline (from a template or an argument
value), and there ours always reports the conflict (tested: we conflict whenever the oracle does). See Q7.
(b) Contract declarations are validated when parsed: only the fields `reads`, `writes`, `volatile`, each a list of
strings; `reads` and `writes` are required (`[]` declares none; the oracle treats an omitted field as empty, so a
typo would silently declare a pure tool), `volatile` may be omitted; outside `{arg}` placeholders (whose names are
word characters, the oracle's `\{(\w+)\}`) a pattern may not contain `?`, `[`, `]` (SPEC: "resources must not
contain other glob characters"), `{` or `}` (the oracle keeps a placeholder such as `{order-id}` as literal text,
which silently loses the conflict); volatile paths are non-empty. Every deviation from the oracle fails loudly on
input the oracle accepts, and no fixture is affected. Argument values are substituted unchanged, as in the oracle
(a `*` inside a value acts as a glob; every other character is literal). Default: as described; cheap to reverse
(one module).

### D-016 Graph record and compile report details
SPEC §1 and §4 leave these open; `src/qet/core/compile.py` does the following. (a) The record's `compile` block
also carries `volatile_consumed`: `[u, v, path]` whenever call v takes as an argument a value that appears in
volatile field `path` of u's raw observation and is not available from the start (SPEC §3: "Compile reports such
cases"); the compile report lists them. (b) The chain fallback after a failed replay audit replaces all edges by
`i -> i+1` labelled `audit` only; the dropped analysis is what the audit refuted. (c) The compile report counts
states exactly up to S_max and buckets larger graphs as `>S_max`; the incomparable-pair share is pooled over all
node pairs of the dataset; width is the largest antichain (Dilworth). (d) A demonstration given to `compile_demo`
ends with exactly one call of the task's finish tool (default `finish`; see Q15). (e) The record's `compile` block
also carries `underived`: `[v, canon_json(value)]` for every argument value of a later call that is available
neither from the start nor from an earlier observation, i.e. that made rule 1 add an edge from every earlier call;
the compile report counts them and lists the most common values, so Gate A can see which constants turn graphs into
chains (Q13). Edges are unchanged. (f) Until Q11 is decided, `compile_demo` refuses a task whose
`initial_observation` is not null, and so does every `GraphRecord` (records read from disk included); the fixture
facade `compile_edges` still accepts it. So no graph record trains on values that its prompts never show. Default: as described; cheap to reverse.

### D-017 Renderer details left open by SPEC §5
(a) An empty `[Tools]` list renders as the single line `  (none)`, as SPEC §5.2 says of every empty section; the
oracle prints no line there. No fixture or benchmark task has an empty tool list, so the difference is moot in
practice, but it is recorded. (b) `native_multiturn` (D-004) uses the same canonical observations as the other
renderers: volatile fields deleted and truncation applied before `[Observation] <canon_json(obs)>`; its first user
message is exactly the `[Task]` and `[Tools]` lines. The facade returns `{"system", "messages"}` for it (the chat
turns after the system message); training and inference use `render_messages`. (c) Truncation settings are
validated: only `max_list_items` and `max_str_chars`, each a non-negative int or null. (d) Tools need `name`,
`params` (a list of names) and `description`. Default: as described; cheap to reverse (one module).

### D-018 Lattice and weight details left open by SPEC §6, §8
`src/qet/core/lattice.py`, `weights.py`. (a) Sub-lattice seed: the first 8 bytes of
sha256("qet-sublattice", graph id, epoch, run seed) seed a `random.Random`, which picks uniformly from the ascending
list of ready nodes (the oracle's rule), so S is identical across processes and configurations (D-013). K has no
default; it comes from the run config and goes into the manifest. (b) Explicit state sets are validated: they
contain the empty state and V, consist of downsets, and every state lies on a chain from the empty state to V
inside S (true of the full lattice and of prefix sets of complete orders); so every layer is non-empty and the
finish weight of the flow-based methods is 1. (c) The orders given to `path_sft` and `sampled_paths` must be
topological orders that stay inside S. (d) A state with occupancy 0 sends no flow and is skipped in the frontier
objective, so unreachable states never produce NaN; a reached state with B(I) = 0 (or, for frontier, with zero
mass on every ready action) raises, because the objective is infinite there. (e) logsumexp sums with `math.fsum`.
(f) The facade's `lattice_summary` enumerates the full lattice when no states are given, up to 2^16 states;
`full_lattice` is capped at S_MAX unless a caller passes another cap. (g) In a sub-lattice, configuration 4's π̄
renormalizes over the ready actions whose successor lies inside S (SPEC §8 writes ready(I)); the two coincide on
the exact lattice, and this is what the oracle and the sub-lattice fixtures do. (h) A log pi must be a number
<= 0 (a sum of token log-probabilities, each <= 0): NaN, positive values and +inf raise, which catches a sign slip in
the scorer; -inf (probability 0) is allowed. The facade accepts only canonical state and edge keys, so one edge has
one key. (i) Edge weights of the flow-based methods record the lambda of their start distribution, and the logged
qet objective uses it unless mu is given, so configuration 5 (lambda = 0) logs -log B(empty) (SPEC §8).
Default: as described; cheap to reverse.

### D-019 Repeat numbering per identical call; steps carry an observation
(a) `#k` numbers repeats of the same tool with the same canonical args, as SPEC §2 and §5.1 define identical
calls, and the Reduce key is (tool, canon_json(args), k). The oracle counts repeats of the display name instead.
The two differ only when non-identifier keys make different calls print the same name (`f(a=1, b=2)` for
`{"a": 1, "b": 2}` and `{"a=1, b": 2}`); there the oracle's rule breaks Proposition 1 (two legal orders of the two
independent calls render different bytes; counterexample from the E0.3-E0.6 review, now a test), while ours keeps
it at the price of two entries sharing a display name (Q10). (b) A step without an `obs` field is an error in
Compile and Reduce (use null for "no observation"), as in the oracle (KeyError); it is never read as null.
Default: as described; cheap to reverse (one function each).

### D-020 Guards for later milestones (from the E0.7 review)
(a) Rule 5 (`task_order`) takes only order constraints that are part of the public task, never edges of a
benchmark's ground-truth DAG: it is the one channel through which an adapter could feed the true DAG into Compile,
which SPEC §4 forbids, and core cannot detect it. Proposed for the M1.3 compile audit: check that adapters pass no
ground-truth edges as task constraints. (b) At inference (E1.4), an action whose args cannot be canonicalized
(NaN, infinities, keys that collide after NFC) ends the episode as `invalid_action` (D-006) instead of raising out
of Reduce; adapters make observations canonicalizable before Reduce (for example NaN to null) and record that in
their compile report. (c) E1.3: the trainer logs `logged_objective(edge_weights(...), lat, logp)` without an explicit
`mu`, or checks that a given `mu` equals `start_weights(lat, ew.lam)`; a mismatched `mu` would silently log another
objective (second follow-up review, note 1). (d) The facade `lattice_summary` should reject keys whose node ids
are >= n before building bitmasks (memory) and report keys that name no edge (facade only; note 2). (e) Once torch
is a dependency (E1), GPU tests are selected from what torch sees (`CUDA_VISIBLE_DEVICES` respected) rather than
from `nvidia-smi -L` (note 3). Default: as described; to be implemented and tested in E1 and M1.

### D-021 Compute and access policy until the human sets a budget
On 2026-10-17 the human said: "You do not need to let me check, you goal is to implement the QET and its
evaluations." So review-only stops (GATE E0) are waived and the agent moves from task to task. Spending stays the
human's decision. Every GPU queue open to this account is billed (`gpu_ppu`, project `ccds_siqiang.luo`); the free
queues' access lists do not include the account's group. Hence: (a) E1 jobs use 1 GPU and at most 1 h of walltime
each, one job at a time, at most 10 GPU-hours for E1 in total, and each job is named in the task's evidence or
RUNLOG; (b) anything larger waits for a GPU budget from the human: the M1.4 throughput pilot on Qwen2.5-7B-Instruct
and the M1.5-M1.7 run matrix. The agent asks once, with the pilot's measured projection (Gate A); (c) no paid API
calls: there is no budget, and `OPENAI_API_KEY` is not set; (d) the human's Hugging Face account already has access
to `meta-llama/Llama-3.1-8B-Instruct` and `meta-llama/Llama-2-7b-hf` (checked 2026-10-17 with metadata requests
only), so downloading them accepts no new license. The AppWorld data terms and the ToolBench / DTA-Tool terms
remain the human's (M2). Default: as described.

### D-022 Chat details left open by SPEC §6.7 and D-002
`src/qet/train/chat.py`. (a) Each backbone is pinned to one Hugging Face commit, which fixes weights, tokenizer and
template together: Qwen2.5-0.5B-Instruct `7ae5576`, Qwen2.5-7B-Instruct `a09a354` (the same tokenizer and template
as 0.5B), Llama-3.1-8B-Instruct `0e9e39f`. Every manifest records the fingerprint: the sha256 of the serialized fast
tokenizer and the sha256 of the chat template. (b) Llama 3.1 gets `date_string = "26 Jul 2024"`, the template's own
fallback. D-002 assumed the template inserts today's date, but the pinned revision's template never reads the clock
(no `strftime_now`; Llama 3.2 and later do), so the prompt equals the template's default. The pin keeps it fixed if
a template ever reads the clock, which the cross-process test checks with a faked clock. (c) Decoding never
cleans up tokenization spaces (Llama 3.1's tokenizer would turn `a , b` into `a, b` by default), so
decode(encode(s)) = s for canonical text. (d) Message text and action strings that spell a special token of the
tokenizer (`<|im_end|>`, `<|eot_id|>`, ...) are refused, because the tokenizer would read them as control tokens.
Adapters keep such text out of observations; at inference (E1.4), an output containing one is an invalid action.
(e) Prompts are validated: an explicit system message first, user and assistant turns alternating, a user turn last.
Default: as described; cheap to reverse.

## Open questions for the human
- [ ] Q1 Confirm D-001 (tool first) or choose the method page's literal key order.
- [ ] Q2 Compute: GPUs, scheduler, concurrency, GPU-hour threshold for asking, budget per milestone.
  (2026-10-17: hardware, scheduler and queue filled into AGENTS.md from the machine; threshold and budgets still
  open, interim policy D-021.)
- [ ] Q3 API budget and models: StableToolBench judge (gpt-4-turbo-2024-04-09 or a fixed replacement), any
  strong-model demonstrations or LLM-drafted contracts.
- [ ] Q4 Access: Llama 3.1 / Llama 2 licenses on Hugging Face, AppWorld data terms, ToolBench / DTA-Tool terms.
  (2026-10-17: the human's Hugging Face account already has access to both Llama repositories, D-021d; AppWorld
  and ToolBench / DTA-Tool terms still open.)
- [ ] Q5 Publication: is code, data or checkpoints going to be released (affects license checks and AppWorld
  rules)?
- [ ] Q6 Anything in SPEC.md the human wants to change before E0 starts.
- [ ] Q7 (from the E0 reviews) Align the oracle with the implementation where they knowingly differ; no fixture
  is affected by any of these: `_glob` should match newlines (`re.DOTALL`, SPEC §3, D-015a); reject the contracts
  D-015(b) rejects; print `  (none)` for an empty `[Tools]` list (SPEC §5.2, D-017a); number repeats per
  identical call (SPEC §2, D-019a).
- [ ] Q8 (SPEC §3) Two patterns meet only when one glob-matches the other, so partial globs on both sides never
  meet (`*:7` and `order:*` both cover `order:7`). Acceptable for the AppWorld-style resources of M2, or should
  meeting mean "some string matches both"?
- [ ] Q9 (SPEC §3) Volatile paths cannot descend into lists or name a key that contains a dot, so lists of
  timestamped records will fail the replay audit and fall back to a chain. Extend the path syntax before AppWorld?
- [ ] Q10 (SPEC §2) `show_call` prints keys raw: different calls can share a display name when keys are not
  identifiers (`f(a=1, b=2)` for `{"a": 1, "b": 2}` and `{"a=1, b": 2}`), and a key containing a newline adds a
  line to the prompt. Since D-019 the prompt stays invariant in that case, but two entries can look alike to the
  model. Keep SPEC §2 as written, or quote non-identifier keys as §5.3 does (`["key"]`)?
- [ ] Q11 (from the E0.3-E0.6 review; SPEC §4 vs §5.2) Compile counts the leaves of `initial_observation` as
  available from the start (D-005) and so drops their data edges, but the §5.2 template never renders the initial
  observation: at the empty state the model is trained to emit values that are not in its prompt. The oracle has
  the same gap. Render it (an `[Initial observation]` block, or `[Available values]` lines under a fixed name), or
  have adapters fold it into the request and drop it from D-005? Decide before an adapter (E1.5, M1) uses it;
  until then `compile_demo` refuses a non-null initial observation (D-016f).
- [ ] Q12 (D-005) The empty string always counts as available from the start (oracle and ours), so an argument
  `""` produced by an earlier observation (an empty cursor, say) gets no data edge. Keep?
- [ ] Q13 (from the E0.7 review; D-005, SPEC §4 rule 1) D-005 says a boolean argument "equal to an earlier output
  gets a data edge", but rule 1 does more: any argument value found neither in the request nor in an earlier
  observation (`true`, `false`, `null`, a default such as `limit=10`, a code such as `lang="en"`) depends on every
  earlier call, so calls that carry such constants compile to a chain (`get_weather(city=X, verbose=true)` three
  times gives a chain of 4 states instead of 8). Should adapters declare `true`, `false`, `null` and schema defaults
  and enums as start values by default? The compile report now lists these values (D-016e).
- [ ] Q14 (from the E0.7 review) The request and tool descriptions are used as given (oracle and ours): an NFD
  request misses whole-word start-value matches (spurious but conservative data edges) and is rendered in NFD
  while targets are NFC. Normalize them to NFC in core, or require adapters to?
- [ ] Q15 (from the E0.7 review; SPEC §1 vs §5.2) A task may name its finish tool (SPEC §1; `compile_demo` honours
  `task["finish"]`), but the pinned system text says "Call finish when the task is complete." Require adapters to
  name the finish tool `finish`, or make the system text use the task's finish name?

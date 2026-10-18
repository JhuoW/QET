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
| D-021 | Accepted | Compute and access policy until the human sets a budget (M1 budget set, D-030) |
| D-022 | Proposed | Chat details left open by SPEC §6.7 and D-002 |
| D-023 | Proposed | Trainer details left open by SPEC §6.5, §8 and D-009 |
| D-024 | Proposed | Scorer details left open by SPEC §6.7 |
| D-025 | Proposed | Runner details and what the smoke audit asks of M1 |
| D-026 | Proposed | Inference and evaluation details (end-of-E1 review) |
| D-027 | Proposed | FuncBenchGen adapter details |
| D-028 | Accepted | M1 grid, splits and manifests (test size set at Gate A) |
| D-029 | Proposed | The replay audit adds one hoisted order per node |
| D-030 | Accepted | Gate A decisions for M1 (the human, 2026-10-18) |

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
from `nvidia-smi -L` (note 3). (f) `canon` accepts lone surrogates (`json.loads('"\\ud800"')`), which no
tokenizer can encode; the chat layer refuses them with a clear error (D-022d). Proposed for core: reject them in
`canon`, as it rejects NaN (E1.1 review, note 6).
Default: as described; to be implemented and tested in E1 and M1.

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
remain the human's (M2). Default: as described. Update 2026-10-17: asked to approve adding FuncBenchGen under
third_party/ and to set a GPU budget for the M1.4 pilot and later runs, the human replied "Approve."; read as approving
the submodule and the plan to do M1.1-M1.4 with the M1.4 throughput pilot inside the per-job cap (1 GPU, at most 1
h); the run matrix still waits for a budget at Gate A. Update 2026-10-18: at Gate A the human set the M1 training budget
(D-030a), which is the batch approval for the M1.5 and M1.6 runs; evaluation spending still waits for its own budget.

### D-022 Chat details left open by SPEC §6.7 and D-002
`src/qet/train/chat.py`. (a) Each backbone is pinned to one Hugging Face commit, which fixes weights, tokenizer and
template together: Qwen2.5-0.5B-Instruct `7ae5576`, Qwen2.5-7B-Instruct `a09a354` (the same tokenizer and template
as 0.5B), Llama-3.1-8B-Instruct `0e9e39f`. Every manifest records the fingerprint: the sha256 of the serialized fast
tokenizer without the backend's transient truncation and padding settings, the sha256 of the chat template, the
sha256 of the ids of a fixed probe prompt and two targets, and the transformers, tokenizers and jinja2 versions.
`tests/train/golden_chat_ids.json` pins the ids of two prompts and six targets per backbone, so a library change
that changes ids fails a test; rewrite it only after a deliberate change recorded here (ids on either side of the
change make checkpoints and manifests incompatible), never to silence a failure. A `ChatFormat` refuses a tokenizer of
another backbone (by name: load it with `ChatFormat.load`). `make fetch` downloads the
pinned tokenizers, which the tests read, and the smoke model. (b) Llama 3.1 gets `date_string = "26 Jul 2024"`.
D-002 assumed the template inserts today's date, but the pinned revision's template never reads the clock (no
`strftime_now`): without `date_string` it uses this same fixed date, so the pin changes nothing for this revision.
The pin protects against a template that reads the clock: the cross-process test edits the Llama template to take
its date from the clock (as Llama 3.2's does) and shows, with faked years and different hash seeds, that the prompt
then changes with the year when unpinned and stays fixed when pinned. (c) Decoding never cleans up tokenization
spaces (Llama 3.1's tokenizer would turn `a , b` into `a, b` by default); decode(encode(s)) = s holds for NFC text
(Qwen's tokenizer NFC-normalizes its input). `target_ids` checks each target: it must be exactly the action string
that `action_string` writes for the call it encodes (so D-001 is defined in one place), NFC as a whole, and survive
an exact decode round trip; otherwise it raises. A canonical action
can fail the NFC check when JSON escaping puts an ASCII letter before a combining mark (Q16). (d) Text that matches
`<|...|>` (the form of every special token of the supported backbones) or a special token of the loaded tokenizer is
refused in prompts and targets, the same way on every backbone, so one dataset is accepted or refused everywhere;
`special_text()` gives adapters the same test, to sanitize observations identically before Compile and at
inference (recorded in their compile report). A test checks that every special token of every backbone has the
`<|...|>` form; a backbone with other special tokens (`<s>`, `[INST]`) must extend the pattern first. Of Qwen's
non-special added tokens, `<tool_call>` and `</tool_call>` are allowed (they round-trip and are tokenized the same
in training and inference); the others (`<|fim_prefix|>`, `<|repo_name|>`, ...) match the pattern and are refused. Lone surrogates are refused with a clear
error (see D-020f). (e) Prompts are validated: an explicit system message first, user and assistant turns
alternating, a user turn last. (f) Guards for inference (E1.4): pass `prompt_ids` to the model (vLLM
`prompt_token_ids`), never `prompt_text`, whose default re-tokenization adds a second begin-of-text on Llama; decode
outputs with `ChatFormat.decode`; pass `eos_token_id=[eot_id]` and explicit decoding settings (greedy, or T = 1.0,
top-p 1.0, top-k off, repetition penalty 1.0), because the backbones' generation_config sets other defaults
(Qwen2.5: sampling at T = 0.7, top-p 0.8, top-k 20, repetition penalty 1.05 or 1.1, and two eos ids); load weights at
the backbone's pinned revision. (g) Llama's template trims every message's content and Qwen's does not:
deterministic and identical in training and inference, but on Llama a message that ends in whitespace loses it
(D-017b holds up to that trim). Default: as described; cheap to reverse.

### D-023 Trainer details left open by SPEC §6.5, §8 and D-009
`src/qet/train/methods.py`, `trainer.py`. (a) Ablation a4 samples the sub-lattice for every graph, not only above
S_max, so "exact vs a4" (claim C5) compares the two on the same graphs; configurations 3-6 use the sub-lattice only
above S_max, with the run's K (D-018a). SPEC §8 read literally samples only above S_max: see Q17. (b) a3 takes K from
the run config (default 4, D-011) and resamples its K - 1 random orders every epoch with the sub-lattice seed
(graph id, epoch, run seed), so with equal K they are the orders whose prefix sets form a4's S. (c) The path
configurations (1, 2, a1, a2) train on the prefix states of their order and run no pass 1; the order-sensitive
renderers (1, a1) are allowed only with the demo order. (d) The prompt of a state renders the demo steps of that
state in demo order (a legal history; Proposition 1 makes every legal order give the same quotient prompt). (e) The
learning rate follows tokens, not steps (configurations without pass 1 take more steps): linear warm-up over the
first 3% of the run's own tokens, then cosine decay to 0 at the budget; the first step has learning rate 0, as in the
usual step-based schedule. A run that starts from a warm-up checkpoint (D-010) starts with a fresh optimizer, so it
gets its own warm-up. Gradients are clipped at norm 1.0 for every configuration; AdamW. (f) The budget counts real
tokens (prompts plus targets, padding excluded) of pass 1 and pass 2 (D-024a); the step that crosses the budget
completes; a checkpoint is taken at the first step that reaches 25, 50, 75 and 100% of the budget (EVAL_PLAN §8), and
a final one after an early stop; fractions already passed by the tokens a run starts with (a format warm-up longer
than 25% of the budget) are skipped, and the warm-up checkpoint stands in for them on the compute curves. The
optional B+ diagnostic (QET runs only) scores the actions that leave S outside the budget and logs its tokens and
time separately. (g) Each epoch visits the records in a seeded shuffle; the last batch of
an epoch may be smaller, and pass 2 normalizes by its own number of graphs (D-014). (h) Guard against an update
between the passes: an optimizer step pre-hook raises from the start of pass 1 to the end of pass 2, and each
trainable parameter's storage, in-place version counter and float64 checksum are compared before pass 1 and
before pass 2, and the checksums again after pass 2 (the checksum catches fused optimizers and writes through
`.data`, which leave the version counter alone); non-finite parameters raise their own error before pass 1.
Gradients are cleared at the start of every step. (i) The pass-2 loss value is returned for tests
only; nothing logs it (SPEC §6.5). (j) The pack builder's run seed must be the run's seed, so the three seeds of a
configuration sample different sub-lattices; `train` builds the epoch-0 packs before the first step, so a graph that
cannot be trained on (more than S_max states without K, a refused target) fails before compute is spent. Default: as
described; cheap to reverse.

### D-024 Scorer details left open by SPEC §6.7
`src/qet/train/scoring.py`. (a) Token accounting: scored tokens are the prompt plus target tokens of the pass-1
sequences, gradient tokens the prompt plus target tokens of the pass-2 sequences (forward and backward process whole
sequences); padding counts in neither and is logged separately as padded positions; seconds are each pass's wall
time (the trainer times pass 2 as a whole). (b) Sequences are right-padded and run without an attention mask, so
only causal masking applies. Scores are batch-invariant exactly in float64 with sdpa attention (not with eager
attention, whose softmax runs in float32). In bf16 on an H200 (Qwen2.5-0.5B-Instruct), the no-mask path, where
transformers passes SDPA an explicit causal mask, gave identical scores alone and batched (job 87685.gaas), while an
all-ones mask, which lets SDPA take its flash kernel, moved scores by up to 0.50 nats (job 87769.gaas); the
faster kernel is therefore not used. (c) Only plain LoRA adapters are accepted, because prompt and prefix tuning act inside `PeftModel.forward`,
which the scorer bypasses; dynamic RoPE scaling is refused, because its frequencies depend on the padded width.
(d) Micro-batches cap padded positions and, optionally, target tokens, since each target token needs
vocabulary-sized float32 logits. (e) Deferred to the M1.4 pilot: gradient checkpointing for the 7B and 8B models
(Hugging Face checkpoints only in training mode and both passes run in eval mode, so the scorer will wrap the
decoder layers in `torch.utils.checkpoint` itself; recomputation is exact without dropout), and the bf16 invariance
test on each backbone that will be trained (Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct). Default: as described;
cheap to reverse.
Update 2026-10-18 (M1.4 sizing, `python scripts/size_activations.py --backbone <id>`): with LoRA on all linear
layers and fp32 adapter weights (peft's default for a bf16 model, which also saves each adapter input as an fp32
copy), one decoder layer saves 327.4 KiB per token for backward on Qwen2.5-7B-Instruct (8.95 MiB per token over 28
layers) and 302.4 KiB on Llama-3.1-8B-Instruct (9.45 MiB over 32 layers); measured on CPU with autograd's
saved-tensor hooks, weights and attention matrices excluded. A 16,384-position micro-batch would need 143 GiB (Qwen)
or 151 GiB (Llama) of saved activations, more than one H200 (143,771 MiB), so the M1 pilot uses `max_tokens: 8192`
(72 or 76 GiB, plus about 15 GiB of weights). Training logs record peak GPU memory per step, and the pilot checks
this estimate. Gradient checkpointing stays deferred unless the pilot shows that 8,192-position micro-batches cost
throughput. Engineering sizing, not a result.

### D-025 Runner details and what the smoke audit asks of M1
`src/qet/run.py`, `manifest.py`, `runlog.py`, `scripts/make_tables.py` (E1.6, E1.7; results-auditor notes on the
smoke table). (a) The clean-tree check ignores `docs/RUNLOG.md`, which every run rewrites, so that the second run of a
job can start after the first has added its row; every other tracked change makes a run refuse to start, and the
manifest lists dirty files. (b) Jobs run in the live working tree, so a commit made while a job runs moves HEAD: the
smoke job's QET run recorded c6f32b6 and HEAD moved to 36ac50d (tests only) 51 s later. From M1 on, jobs run from a
pinned checkout (`git worktree add <dir> <commit>`) so the code a run uses is exactly its recorded commit. (c) GPU
training is not bit-reproducible (the two smoke runs' identical warm-ups diverged from step 3); numbers are
re-derived from `results/raw/`, never by re-running. (d) In E1 each run did its own format warm-up; D-010 asks
configurations 2-6 to share one warm-up checkpoint per backbone and seed. Before M1.6 the warm-up becomes its own run
whose checkpoint the configuration runs name as `parent_checkpoint` (its tokens counted through
`tokens_at_start`). (e) Before M1 tables are reported, `make_tables.py` gives error rates per decoding setting, per
call and per task (EVAL_PLAN §7) instead of pooling greedy with sampled episodes, states the episode and task counts of
the Resume@prefix columns, and reports each run's budget overshoot (D-023f; 0.12% and 0.59% in the smoke runs).
Default: as described; (b), (d) and (e) to be implemented before the M1 runs. Update after the end-of-E1 review:
(e) is implemented (greedy outcome shares, the sampled setting named, Resume@prefix episode counts, overshoot,
training GPU-hours). Also implemented: untracked files under src/, configs/, scripts/ and tests/ count as
dirt; the git state is re-read at the end of a run (`git_end`, `code_changed_during_run`); a SIGTERM (walltime,
qdel) marks the run failed, and failed runs record their GPU-hours; GPU-hours are split into load, training
and evaluation; the run log is locked and written through unique temp files; checkpoints record the sha256 of
their files; demonstrations pass the environment's checker before Compile. Still open for M1: (b), (d), an
evaluation-only run that loads a saved checkpoint by path and hash (EVAL_PLAN §8 budget fractions), and the
"no history" condition of the chronological baselines in Resume@prefix (EVAL_PLAN §6.5, M1.7).
Follow-up review: bookkeeping after completion (the end-of-run git check, the RUNLOG update) is recorded in the
manifest and never turns a completed run into a failed one; `make smoke` runs each configuration in its own
process; untracked files under third_party/ and top-level .py files also count as dirt; the dry run counts the
starts the run evaluates. For M1, adapters verify with the environment's checker the exact trace they compile
(the toy env regenerates an identical one).

### D-026 Inference and evaluation details (end-of-E1 review)
`src/qet/infer/`, `src/qet/eval/`, `tests/train/test_scoring.py`. (a) The GPU test's bf16-vs-float32 check keeps the
bound first committed, |delta log pi| < 1.0 nat; bf16 rounding alone reaches 0.899 nats there (2.7% of a mean log pi
of -30.7; jobs 87789.gaas and 87799.gaas), so the margin is thin, and a looser bound is the human's decision (Q18;
AGENTS.md rule 6). The spec's tolerance, batch invariance within 1e-4, is separate and unchanged (worst 0). (b) Decoding
uses an explicit loop rather than Hugging Face `generate`, which fills unset fields from the model's generation_config
(D-022f); it stops only at the end-of-turn id (`<|endoftext|>` does not stop it, and an output without the end of
turn is invalid). A vLLM path for scale must reproduce: stop only at the end of turn, the given T and top-p, top-k off,
no penalties, per-step seeds. (c) Parsing is stricter than SPEC §7's "lenient about spacing and key order" in four
ways, each an invalid action (D-006): duplicate keys at any depth; lone surrogates spelled through JSON escapes;
special-token text inside decoded arguments (D-022d); nesting deeper than Python's recursion limit. (d) Resume@prefix
runs the first decoding setting (greedy, rollout 0); its empty start repeats the standard greedy episode. Its starts
are fixed by an evaluation seed (default 0), the same for every training seed of a comparison, and written to the data
manifest (`resume_starts.json`). (e) Error taxonomy: episodes end in success, wrong answer, invalid action, budget
exhausted or render error; errors are counted per call and per episode by the environment's own reason when it logs
one (FuncBenchGen: value not yet known, incorrect value), else by the step's observation-error flag. Default: as
described; cheap to reverse.

### D-027 FuncBenchGen adapter details
`src/qet/envs/funcbenchgen.py`; the repository is the git submodule `third_party/FuncBenchGen` at `0729e25`
(2026-02-10, BSD-3), never modified. (a) A task is one graph of one grid cell and one trial, built with the
repository's own functions: `_build_graph_for_seed` for the graph seed, `reformat_tree_with_shared_types` (use_types,
the repository's default), `derive_trial_seed` and `_initialize_trial_variables` for the values, `_get_input_prompt`
for the request and `_get_function_schemas` for the tools, plus finish(answer). Where the repository's `run` lets an
unseeded RNG pick seeds (the tree's renaming, the evaluator), the adapter passes seeds derived from (cell, graph seed),
so a task is a function of (cell, graph seed, trial); all of it runs on its own copy of the Python and NumPy global RNG
states, leaving everyone else's untouched. (b) `step` is the repository's `_execute_tool` with on_wrong_inputs =
"Execute", error_detail_level = "None" and no restated values (configuration 1's renderer restates values instead):
an input whose type is not yet known, or a wrong value, returns a random value from the repository's RNG, seeded the
same way at the start of every episode. "Variable x = 123." becomes {"x": 123} and "Error occurred." becomes
{"error": "Error occurred."}; the repository's failure reason goes to the episode's error log. (c) Success is the
repository's rule: the finish answer equals the target value and at least the minimum number of calls was made; calls
are capped at twice the minimum. The agent makes one call per step, where the repository's loop executes every call
of a message. (d) Contracts: every function is pure (SPEC §3); the repository's "known types" gate follows the data
dependencies, which Compile recovers from values, and a hidden value equal to a stated input defeats that and fails
the replay audit (seen on 1 graph of 30 in a 5-function cell), so M1.3 runs the audit and falls back to a chain. (e)
`funcbenchgen.py` imports LiteLLM at module level for API calls the adapter never makes, so a stub module stands in
for it. (f) The repository's generator turns sets of names into lists
before drawing from them, so its graphs depend on Python's per-process string hashing (found when manifests built in
one process differed from another's); the adapter builds graphs under PYTHONHASHSEED=0, in a helper process when
the caller runs under another seed, and caches them, as the repository's runner caches graphs on disk. Everything
after the graph (types, values, prompt, schemas) does not depend on hashing. Default: as described; cheap to
reverse.

### D-028 M1 grid, splits and manifests (draft until Gate A)
`configs/m1/funcbenchgen.yaml` (sha256 9ecc71346472dd8c6a42dea7103e8140f16c42a4837d21101e6c419480b0d7e5) and `scripts/make_m1_manifests.py` build
`configs/m1/manifests/*.json`: every task with its cell, seed, trial, variant and the sha256 of its content (public
text, gold calls, answer, true DAG). (a) The grid is EVAL_PLAN §3.1's in full: core functions 5, 10, 20; every depth
1 .. n-1; irrelevant functions 0, or 10 and 20 each connected, disconnected (their own subgraph) or mixed (half each):
224 cells. (b) Training: 5,000 graphs round-robin over the cells (about 22 per cell), graph seeds 0-4,999, one trial
each. Dev (model selection): 4 graphs per cell, seeds from 10,000,000. Test: 20 graphs per cell, seeds from 20,000,000,
a provisional size that Gate A replaces by power analysis. Held-out larger: 30 core functions (more calls than any
training cell) at depths 1, 2, 4, 8, 16, 29 with 0 or 10 connected irrelevant functions, 20 graphs each, seeds from
30,000,000. Renamed: the test graphs with new tool names, variable names, types and values. Seed ranges are disjoint, so
no graph is in two splits. (c) The stateful extension is the toy accounts-and-orders tasks (`qet.envs.toy`) in six
cells (3, 5, 8 orders; 1-3 accounts): 1,200 training and 120 test tasks, reported separately; it has stateful tools,
effect and decision edges, and payments that truly commute but that Compile cannot certify. Whether this matches the
method page's extension is the human's call at Gate A. (d) The manifests rebuild byte for byte from code, the spec and
the pinned repository under any hash seed (built with PYTHONHASHSEED=7, checked with 99). sha256 of the files:
  `train.json` 35919d61dce815d1ec85f81d015d05053a6c5ff4e9a0378556cb2ead12386427
  `dev.json` c6fb851180d8a4201923ca94111c8572301bc172c7420e381f74ee7df315b31f
  `test.json` 36bec2d1d8ae4eb4e703c88fe5bf3fb3969aaea1600ba2c3ada18af6f7ff1835
  `heldout_larger.json` 7aa0a12ba2450252e13ba79669e3dcde806c1508981ff2f9421a59e5915febab
  `renamed.json` 825d4b0fba0a24599418348c78f06818513ed60b9041f4c9c40e719681941223
  `stateful_train.json` 2e9318bd5ece43a35d8faebd649153575b1e75930c329f4df9f753037d2b8bd8
  `stateful_test.json` 3c8b6cbb96af7f96c0c880af94dfd549ca6c41622936e449ad93a5718ef1b45b
These are drafts: the test manifests are frozen at Gate A (EVAL_PLAN §0), and any change before then is recorded here.
Update 2026-10-18 (Gate A, D-030): the human accepted (c) and set the test size to 10 graphs per cell, the first 10
graphs (lowest graph seeds) of each cell of `test.json`, with `renamed.json` following them; the frozen manifests and
their hashes are added here when built. Built 2026-10-18 (`PYTHONHASHSEED=7 python scripts/make_m1_manifests.py`;
`PYTHONHASHSEED=99 ... --check` identical): the spec now says `frozen: true` and 10 test graphs per cell (sha256
4d14ead668925d67d3fd20806ca1c5052057cfa42c0c2b9017d76b2c09a212e7); test and renamed hold 2,240 tasks each, all in the
drafts above with the same content hashes, and the other splits keep their task lists (their files change only through
the spec hash). Frozen sha256 of the files:
  `train.json` efc6ad9b8210dbb2cf9bbd5a49e6c1098105f6aa1bea681851a4da2270489fb7
  `dev.json` 30e725a915c1539747d7d8fbbb346ab28d674673d29c87cd1bb4ab6dadb33622
  `test.json` f1d28593bcdaf59c18c36c2cfa5034d59e0d1636aba41dce9e8a925147ebb6bf
  `heldout_larger.json` 267b5054c905d29de4ae5549381d762a71836c444f6dddbd71e0fbe4c8f6b9ac
  `renamed.json` c4788095dd72bfcc0c42e99ca8b199d4cfc112368b60d3326ca8c3055d3009d8
  `stateful_train.json` ab964d322ad4decb3e415ecabe7beab9a0fe98430ba3d54e2d9ecf2fb4ae50ed
  `stateful_test.json` 0df26343f48a11f4007aa7ff2f2f3fa1ac2b48113be4dd6e4b9c5e3212c64528

### D-029 The replay audit adds one hoisted order per node
`src/qet/envs/base.py` (`hoisted_orders`, `replay_audit(hoisted=True)`), used by `scripts/compile_m1.py`. SPEC §4's
replay audit executes K random topological orders (K = 8 here). In FuncBenchGen, a hidden value that equals a stated
input value makes Compile take it as available from the start and drop the data edge; a 200-graph sample of the
training split showed such misses and that 8 random orders alone left some of them undetected, so those graphs would
have trained on orders that are illegal in truth. The audit therefore also executes, for every node, its hoisted order:
the node's ancestors in the compiled graph, then the node, then the rest in demo order. That runs each node before
every call Compile left independent of it, so a dependency Compile missed executes its consumer before its producer,
the environment's observation differs, and the graph falls back to a chain. It costs n extra replays per graph, and it
reads only the compiled graph and the trace, never the true DAG. The numbers come from the M1.3 compile run in
results/raw/ (compile report, audits). Default: as described; cheap to reverse.

### D-030 Gate A decisions for M1 (the human, 2026-10-18)
`docs/reports/gate-A.md` (generated by `python scripts/gate_report.py --gate A`; results-auditor: PASS WITH NOTES on the
re-audit, transcribed in `docs/reports/gate-A-audit.md`) gives the numbers; this entry lists the proposals so that each
decision has a record. Nothing below runs until the human decides; the agent starts no M1.5-M1.7 GPU job before then
(D-021b). (a) Training budget: every run gets the same B = 250M scored + gradient tokens (report section 6 gives 125M-1B
with epochs and GPU-hours), recorded as a batch approval for the 33-run matrix. (b) Tuning: no learning-rate grid in M1;
D-009's defaults for every configuration (equal tuning with zero dev trials each; section 10 prices a 3-value grid). (c)
Decoding limit: max_new_tokens = 512 for every system and backbone, twice the longest gold action of any evaluation
split, rounded up to a multiple of 64 (section 7; the pilot's 64 leaves some test tasks unsolvable by construction). (d)
Test size: 10 graphs per cell, the first 10 graphs (lowest graph seeds) of each cell of the provisional test manifest,
with the renamed set following it (section 8), then frozen. (e) EVAL_PLAN §9 wording (section 9), evaluated at each
run's 100% checkpoint so that compared runs spent the same compute; choosing checkpoints by dev success would change "at
matched compute". EVAL_PLAN.md is the human's to edit. (f) Evaluation: the agent builds batched decoding (D-026b) and
measures it in short jobs within the per-job cap; the evaluation budget is set from that measurement. (g) The warm-up
length (D-010) is measured at the start of M1.5, since the pilot ran a fixed 5 steps.

Decided 2026-10-18: the human answered the Gate A questions. (a) Training budget: B = 125M tokens per run, not the 250M
proposed ("125M tokens"); the report's projection is about 127 GPU-hours of training for the 33-run matrix. This is the
batch approval for the M1.5 warm-up runs and the M1.6 training runs at that budget: the agent stops and asks if the
matrix's measured training GPU-hours would pass the projection by more than 10%. Evaluation is not covered; its budget
follows the batched-decoding measurement (f). (b) No grid, as proposed. (c) max_new_tokens = 512, as proposed. (d) 10
graphs per cell, as proposed; the frozen test manifests are built next and their hashes recorded in D-028. (e) The §9
wording accepted as proposed ("Accept as proposed"); EVAL_PLAN.md is human-only, so the human pastes it in, and no test
evaluation runs before EVAL_PLAN §9 carries it. Also decided: Q17, a4 uses the sub-lattice on every graph (D-023a
stands); D-028c, the toy accounts-and-orders tasks are the stateful extension.

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
- [ ] Q16 (from the E1.1 review; SPEC §2) `canon` NFC-normalizes each string, but the serialized action string can
  still be non-NFC: JSON escaping puts an ASCII letter right before a combining mark (the value `"\n\u0303"` is
  written as `\n` followed by U+0303, which NFC composes into `\ñ`). Qwen's tokenizer NFC-normalizes its input, so it
  would be trained to emit invalid JSON; `target_ids` now refuses such strings on every backbone (D-022c), so a
  demonstration carrying such a value cannot be trained on. Change `canon_json` to write a combining mark that
  follows an escape as `\uXXXX` (no fixture is affected), or keep refusing?
- [x] Q17 (resolved 2026-10-18 at Gate A: the human chose "Every graph", so D-023a stands) (from the E1.3 review;
  SPEC §8 a4, EVAL_PLAN §1 C5) D-023a applies the sub-lattice to every graph for
  ablation a4, because otherwise a4 equals configuration 6 on every graph below S_max and "exact vs a4 with K in
  {2, 4, 8}" (claim C5, figure F5) measures nothing there. SPEC §8 read literally ("§6.6 with the given K") samples
  only above S_max. Confirm D-023a before M1.6, since it fixes what a pre-registered comparison means.
- [ ] Q18 (from the end-of-E1 review; D-026a) The GPU test compares bf16 and float32 scores of the same weights with
  the bound |delta log pi| < 1.0 nat, set before any measurement; bf16 rounding alone gives 0.899 nats on its 24 pairs.
  Keep it, or replace it with a bound set from bf16's precision, for example 0.1 nats per target token (2^-8 relative
  precision on logits up to about 30; the measured worst is about 0.045 per token)? The agent does not loosen it.

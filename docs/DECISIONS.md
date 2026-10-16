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

## Open questions for the human
- [ ] Q1 Confirm D-001 (tool first) or choose the method page's literal key order.
- [ ] Q2 Compute: GPUs, scheduler, concurrency, GPU-hour threshold for asking, budget per milestone.
- [ ] Q3 API budget and models: StableToolBench judge (gpt-4-turbo-2024-04-09 or a fixed replacement), any
  strong-model demonstrations or LLM-drafted contracts.
- [ ] Q4 Access: Llama 3.1 / Llama 2 licenses on Hugging Face, AppWorld data terms, ToolBench / DTA-Tool terms.
- [ ] Q5 Publication: is code, data or checkpoints going to be released (affects license checks and AppWorld
  rules)?
- [ ] Q6 Anything in SPEC.md the human wants to change before E0 starts.

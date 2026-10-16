# QET — implementation specification

Status: **normative**. Human-owned: agents propose changes in `docs/DECISIONS.md` and never edit this file.
Source: the QET method page in Notion (page 3f9730d983f2817a960ae11a1c49d870), "the method page" below.
This file restates it for implementation; where it fixes a detail the
page leaves open, the decision is cited as D-xxx. The oracle `reference/qet_ref.py` implements every rule
below literally, and `reference/fixtures/` pins its outputs.

## 0. The method in one paragraph

A successful demonstration is a sequence of tool calls with observations. **Compile** turns it into a DAG G
whose edges are the orders the task truly requires (§4). A progress state I is a set of completed nodes that
contains all predecessors of its members (a downset of G); the states form a lattice from ∅ to V. **Quotient**:
Reduce + Render map any history to a prompt that depends only on the completed set, byte for byte (§5), so
the LLM policy is a function of the state. **Marginalize**: the probability that the model finishes from I in
any legal order, B(I), obeys a one-line recursion over the lattice (§6.2); the loss Σ_I μ_G(I)·(−log B(I))
has a gradient equal to weighted SFT on (state, action) pairs with weights ρ computed from the model's own
success posterior (§6.4). Training is two passes per step: score without gradients, compute ρ, then weighted
SFT. Inference never sees G: it re-renders the real history each step (§7).

## 1. Data objects

**Task x** — everything public at the start: `request` (string), `tools` (list of `{name, params, description}`
plus an optional JSON schema), optional `initial_observation`, the `finish` tool name, and the contracts (§3).
The tool list is part of x: it must not depend on the history.

**Call** — `{tool, args}`; **step** — `{tool, args, obs}`; **demonstration τ** — steps in time order, verified
successful by the environment's checker, followed by a finish call `{tool: "finish", args: {answer: …}}`.

**Graph record** (Compile output, one per demonstration; JSON):

```json
{
  "task_id": "funcbench-000137",
  "request": "…", "tools": [ … ], "initial_observation": null,
  "snapshot": "env@commit-or-state-id",
  "nodes": [{"id": 0, "call": {"tool": "get_flight_price", "args": {"from": "NYC", "to": "SG"}}, "obs": {"price": 1200}}],
  "edges": [{"from": 0, "to": 2, "why": ["data"]}],
  "final": {"tool": "finish", "args": {"answer": 1800}},
  "compile": {"version": "…", "fallback_to_chain": false, "audit": {"orders_replayed": 0, "mismatches": 0}}
}
```

Node ids are demo positions 0..n−1, so every edge goes from a smaller to a larger id. `obs` is the canonical
observation (§2) with volatile fields removed. `why` lists every rule that produced the edge.

**Lattice pack** (built from a graph record at training time; independent of θ, cacheable): the states S;
for each state its prompt and its targets — the action strings of its ready nodes, or the finish action at V.

## 2. Canonical form

| Input | Canonical output |
| --- | --- |
| float with integral value and abs < 2^53 | int (`1200.0` → `1200`); `-0.0` → `0` |
| other float | shortest round-trip repr; NaN and ±inf are errors |
| string, object key | Unicode NFC; nothing else (no case folding, no trimming) |
| object | keys sorted by code point; non-string keys and NFC collisions are errors |
| bool, null, int, list | unchanged (lists keep their order) |

`canon_json(x)` = `json.dumps(canon(x), sort_keys=True, separators=(",", ":"), ensure_ascii=False)`.

**Action string (D-001)**: `{"tool":"<name>","args":<canon_json(args)>}` — tool first, then args with sorted
keys, no spaces. Example: `{"tool":"get_flight_price","args":{"from":"NYC","to":"SG"}}`. The method page sorts
the top-level keys too (args before tool); D-001 records the switch and its status. Every configuration uses the
same strings. The target the model is trained to emit is the action string followed by the end-of-turn token
(D-002); its probability includes that token.

**Call name**: `show_call(tool, args)` = `tool(k1=v1, k2=v2)` with keys sorted and each value as `canon_json`,
e.g. `get_flight_price(from="NYC", to="SG")`, `ping()`. Identical calls (same tool, same canonical args) get
`#2`, `#3`, … in execution order: `get_balance(account=1)#2`. Never name an entry by its step number.

## 3. Tool contracts and the conflict function

A contract is declared once per tool (not per trajectory):

```json
{"update_order": {"reads": ["order:{id}"], "writes": ["order:{id}"], "volatile": ["request_id", "meta.ts"]}}
```

- `reads` / `writes` are resource patterns. `{arg}` is replaced by that argument's value (strings verbatim,
  other values as canon_json); a missing argument makes the resource `*`. `*` inside a pattern is a glob
  (`order:*` meets `order:7`); resources must not contain other glob characters.
- A tool without a contract reads and writes `*` (everything). A tool with empty `reads` and `writes` is pure.
- `volatile` lists observation fields (dotted paths) deleted during canonicalization: timestamps, request ids.
  A field that a later call consumes as an argument must not be volatile (Compile reports such cases).
- Global counters and id generators are shared resources written by every tool that advances them.
- `conflict(u, v)` is the Bernstein condition: `W(u) ∩ (R(v) ∪ W(v)) ≠ ∅` or `W(v) ∩ R(u) ≠ ∅`, where two
  patterns meet if either glob-matches the other. Compile (§4) and Reduce (§5) call **the same function**.

Contract sources: by construction in controlled environments (FuncBenchGen functions are pure); from the REST
method in AppWorld (GET reads, POST/PUT/PATCH/DELETE write, resource = app plus entity and id); GET endpoints
are read-only in StableToolBench after a replay audit through the cache; elsewhere an LLM may draft contracts
from the docs, which are then verified by sandbox differential tests (run two calls in both orders, compare
states). Record each environment's contract source and verification result in its compile report.

## 4. Compile

**Input**: request, initial observation, steps, contracts, optional task constraints (index pairs), optional
adapter-declared start values (D-005). **Output**: edges `[u, v, labels]` with u < v. Cost O(n²) look-ups;
no model calls, no environment calls.

**Values available from the start** (D-005): numbers that appear as standalone numeric tokens in the request;
strings that appear in the request as a whole word sequence (`(?<!\w)value(?!\w)`, case-sensitive); every
scalar leaf of the initial observation; adapter-declared start values. Values are compared by typed key after
canonicalization: `1.0` equals `1`, but `true` never equals `1`.

**Rules**: for every pair u < v (u earlier in the demo), add `u → v` with a label when any rule holds.

1. **data** — for each scalar leaf value of v's args that is not available from the start: every earlier u
   whose canonical observation contains that value as a scalar leaf; if no earlier observation contains it
   (an underived value), every earlier u.
2. **effect** — `conflict(u, v)`.
3. **decision** — v writes anything (`W(v) ≠ ∅`): every earlier u. Reads may move ahead of writes and of each
   other; writes never move ahead of anything.
4. **dup** — u and v have identical action strings.
5. **task** — the pair is in the task's annotated order constraints.

All other pairs are independent. Because edges point forward in demo order, G is acyclic and the demo order
is one of its topological orders. Uncertain cases (no contract, unparseable values) always resolve toward
adding edges. Labels are `data`, `effect`, `decision`, `dup`, `task`, plus `audit` for chain fallback; an edge
carries every label that applies, sorted.

**Replay audit** (optional, environments with snapshots): from the snapshot, execute K random topological orders
and compare every node's canonical observation and the final state with the demo. Any mismatch → the graph
falls back to a chain (edges i → i+1, label `audit`) and is counted. Benchmark ground-truth DAGs (FuncBenchGen)
are used only to audit Compile (edge precision and recall, unsafe-order rate), never as its input.

**Compile report** per dataset: tasks compiled, fallback share, width distribution (largest antichain), share
of incomparable pairs, states per graph, label counts, contract source, audit failures.

## 5. Quotient interface: Reduce and Render

### 5.1 Reduce
Input: task (with contracts) and the raw history in execution order; output: the ordered entry list H.
1. Canonicalize each call and observation (§2), deleting volatile fields.
2. Name each entry by `show_call`, adding `#k` to repeated identical calls in execution order.
3. Order: take the smallest-key topological order of the entries under the conflict constraints only —
   repeatedly place, among the entries whose conflicting predecessors (earlier in the history and
   `conflict(…) = true`) are all placed, the one with the smallest key `(tool, canon_json(args), k)`
   (Python string order). Conflicting entries therefore keep their real order; everything else is sorted.

Reduce never reads G and is defined for any history, including calls never seen in training and error
observations. Examples (fixtures): the seven-step booking history reduces to `get_balance(account=1)`,
`get_flight_price(…)`, `book_flight(…)`, `get_hotel_price(…)`, `get_weather(…)`, `pay(…)`,
`get_balance(account=1)#2`; "update paid → get_flight_price → update shipped → get_order(7)" reduces to
get_flight_price, update paid, update shipped, get_order.

### 5.2 Render (template text pinned by D-003)
Output: `{system, user}`, mapped to chat messages by D-002.

- system: `You are a tool-using agent. Read the task and the completed calls, then output the next action as one
  JSON object {"tool": ..., "args": {...}} and nothing else. Call finish when the task is complete.`
- user, lines joined by `\n`, no trailing newline:

```
[Task] <request>
[Tools]
  <name>(<param>, <param>): <description>
[Completed calls]
  <entry name> -> <canon_json(obs)>
[Available values]
  <entry name><path> = <canon_json(value)>
[Next action]
```

`[Tools]` has one line per tool, sorted by name, parameters in schema order. `[Completed calls]` has one line per
entry of H in order; `[Available values]` one line per leaf (§5.3). An empty section is the single line
`  (none)`. Lines are indented by two spaces exactly as shown.

### 5.3 Available values
Each observation is flattened to its leaves in canonical order: object keys in sorted order as `.key` (or
`["key"]` when the key is not an identifier), list items as `[i]`; scalars and empty containers are leaves; an
empty observation contributes nothing. Example: `get_flight_price(from="NYC", to="SG").price = 1200`,
`list_mail().meta["next page"] = 2`.

### 5.4 Truncation
Content-based only, applied to the canonical observation before both sections: lists keep their first
`max_list_items` items plus a final string item `"<+N more>"`; strings longer than `max_str_chars` keep that
prefix plus `"<+M chars>"`. Limits are per environment, set in the config, identical in training and inference.

### 5.5 Constraints
- The prompt is a function of (x, H) only: no step numbers, timestamps, dates, turn counters, run ids.
- Re-render the whole prompt every step; never append turns.
- No hidden state across steps: no reasoning text carried over; thinking, if used, stays within one step.
- Never iterate sets or unsorted dicts while building text. Chat templates that inject dates (Llama 3.1) get a
  fixed `date_string` (D-002).

### 5.6 Invariance (Proposition 1 of the method page) and its tests
For any two legal histories that complete the same set I, Reduce returns the same H_I, hence identical prompts.
Required tests: (a) for every training graph (or a large sample), several random topological orders: every
prefix renders byte-identically to the demo-order history of the same set; a failure is an implementation bug;
(b) in the replay audit, the same comparison with freshly executed observations; a failure means a wrong
contract or volatile list, and the graph falls back to a chain; (c) the inference agent following any legal
order sees, at each step, exactly the training prompt of that state (E1.4).

### 5.7 Renderers of the baselines (D-004)
- `chronological` (configuration 1): the same template with `[History]` in place of `[Completed calls]`,
  entries in execution order numbered `1.`, `2.`, …, the same canonicalization and naming, and the same
  `[Available values]` block in execution order (the restated-values control). Order-sensitive by design.
- `native_multiturn` (ablation a1, plain chronological SFT): system message as above; first user message = the
  `[Task]` and `[Tools]` blocks; then per step an assistant message with the action string and a user message
  `[Observation] <canon_json(obs)>`; no `[Available values]`.

## 6. Marginalize

### 6.1 States
`ready(I)` = nodes not in I whose predecessors are all in I. Enumerate states breadth-first from ∅, merging
repeats; stop as soon as the count exceeds S_max (default 256) and switch to the sub-lattice (§6.6). A chain has
n+1 states; n independent calls have 2^n; k chains of lengths ℓ_j have at most Π(ℓ_j+1).

### 6.2 Completion probability (eq. 1)
π(a | H_I) is the probability of the full target (action string plus end-of-turn token) under the prompt of I:
the sum of its token log-probabilities, **without** length normalization and **without** renormalizing over the
legal actions (mass on other strings is the failure probability).

    B(V) = π(a_end | H_V)            B(I) = Σ_{v ∈ ready(I)} π(a_v | H_I) · B(I ∪ {v})

Compute in log space with logsumexp, in float64, from the largest states down. The recursion is exact for the
real model because of §5.6. Running example: B({A,B}) = 0.76, B({A}) = 0.38, B({B}) = 0.684, B(∅) = 0.4332.

### 6.3 Start distribution μ_G and objective (eq. 2)
With n = |V|, N_k = number of states with exactly k completed nodes (counted over the states in use), λ ∈ [0, 1),
default 0.5:

    μ(∅) = 1 − λ;   μ(I) = λ / (n − 1) / N_|I|  for 1 ≤ |I| ≤ n − 1;   μ(V) = 0
    n = 1: μ(∅) = 1.    n = 0 (no tool calls): μ(V = ∅) = 1, so the loss is −log π(a_end | H_∅).

    L_QET(θ) = E_(x,G) [ Σ_I μ(I) · (−log B(I)) ]

Running example (λ = 0.5): μ = 0.5, 0.125, 0.125, 0.25, 0; L = 0.6553. On a chain with λ = 0 the loss is SFT.

### 6.4 Gradient as weighted SFT (eq. 3)

    q(v | I) = π(a_v | H_I) · B(I ∪ {v}) / B(I)              success posterior
    m(I)     = μ(I) + Σ_{(J,u): J ∪ {u} = I} ρ(J, u)          occupancy, computed from small states up
    ρ(I, v)  = m(I) · q(v | I);       weight of the finish action at V = m(V) = 1
    ∇L = − Σ_{I ≠ V} Σ_v ρ(I, v) ∇log π(a_v | H_I) − ∇log π(a_end | H_V)

Checks: Σ of tool-call weights = Σ_I μ(I)(n − |I|); ρ = −∂L/∂log π (finite differences). Running example:
ρ = 0.263, 0.237, 0.388, 0.362, 1, 1; the root splits 0.526 : 0.474.

### 6.5 Training step (Algorithm 1)
For a batch of graph records:
1. Build or load lattice packs (exact, or sub-lattice when states > S_max).
2. **Pass 1 (score)**: model in eval mode (dropout inactive), `torch.no_grad()`; log π for every (state, target)
   pair; accumulate in float64.
3. **Recursion**: log B by §6.2; weights by the configuration's rule (§8); log the true objective (for QET
   Σ μ(−log B)) per graph and per batch.
4. **Pass 2 (train)**: same model, same mode, same parameters; loss = − Σ w(I, v) · log π(a_v | H_I) with w
   detached, summed over targets, normalized by the number of graphs in the batch (not by tokens); micro-batch
   and accumulate; one optimizer step per batch. No parameter change between pass 1 and pass 2.
5. The pass-2 loss value equals L plus the μ-weighted entropy of the success posterior; never use it for
   logging, early stopping or model selection.

Equivalent single-pass form (for tests): build L with autograd through logsumexp over the edge log-probs; its
gradient must equal the two-pass gradient (E1.3). Edges with zero weight are not processed in pass 2. Pruning
edges with tiny positive weight is off by default; if enabled, log the pruned mass.

Toy check (fixture `toy_training.json`): a table policy (one softmax per state over its targets plus a "wrong"
bucket) trained by gradient descent with lr 0.5 from the demo-path-only model raises π(A | H_{B}) from 0.10 to
0.956 in 200 steps under QET, while demo SFT leaves it at 0.100.

### 6.6 Sub-lattice (large graphs)
When states exceed S_max: sample K topological orders — the demo order plus K−1 orders from Kahn's algorithm
with uniformly random choice among ready nodes, seeded by (graph id, epoch, run seed) — and let S be all their
prefix sets. Run the same recursion inside S (successors outside S count 0) to get B_S ≤ B; compute μ with N_k
counted inside S (every layer is non-empty); weights as usual. Resample S every epoch. Diagnostic (optional):
score all ready targets of the states in S (same forward) and set successors outside S to 1 to get B⁺ ≥ B;
log(B⁺/B_S) bounds the objective error. With S = the full lattice, B_S = B⁺ = B.

### 6.7 Scoring implementation
- Prompt ids = the tokenizer's chat template applied to [system, user] with `add_generation_prompt=True`;
  target ids = tokenizer(action string, no special tokens) + [end-of-turn id] (D-002). Score target tokens only.
- Use the same template, tokenizer revision and dtype policy in training, scoring and inference; compute
  log-softmax in float32, sums in float64.
- States of one graph share the [System]+[Task]+[Tools] prefix; prefix caching or packing is an allowed
  optimization if it leaves log-probs unchanged (test within 1e-4).
- Token accounting per step: scored tokens (pass 1, prompt plus target), gradient tokens (pass 2), number of
  sequences, wall time; written to the training log (EVAL_PLAN §8 uses them for matched compute).

## 7. Inference (Algorithm 2)

    H ← Reduce(x, [])
    loop until the step budget:
        a ← decode one action from π(· | Render(x, H))
        if a is invalid: stop, failure "invalid_action" (D-006)
        if a is finish: stop and score with the benchmark's checker
        o ← execute a in the real environment;  history ← history + [(a, o)];  H ← Reduce(x, history)

- Decoding: greedy, and sampling at T = 1.0, top-p 1.0 with 8 independent rollouts per task; never best-of-k.
- Parsing is lenient about spacing and key order but requires a JSON object with exactly `tool` and `args`,
  a tool from the task's list and an object for args.
- Inference never touches G, the lattice, B or weights. Log per step: prompt sha256, raw output, parsed action,
  observation, error flags, tokens.

## 8. Training configurations as edge weights

All configurations share demonstrations, contracts, compiled graphs, target strings, format warm-up (D-010),
optimizer and token budget. For the flow-based rows, w(I, v) = m(I) · s(v | I), where the step probability
s sums to 1 over the successors of I and m is the occupancy m(I) = μ(I) + Σ of incoming weights (§6.4).

| # | Configuration | Renderer | Starts | Edge weight w(I, v) | Pass 1 | Logged objective |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Chronological SFT, restated values | chronological | ∅ | 1 on demo-path edges, else 0 | no | demo-path NLL |
| 2 | Interface + demo-order SFT | quotient | ∅ | 1 on demo-path edges, else 0 | no | demo-path NLL |
| 3 | Interface + uniform legal-order augmentation | quotient | μ (λ = 0.5) | m(I) · c(I ∪ {v}) / c(I) | no | Σ w·(−log π) |
| 4 | Interface + frontier supervision | quotient | μ (λ = 0.5) | m(I) · π(a_v) / Σ_{u ∈ ready(I)} π(a_u) | yes | Σ_I m(I)·ℓ(I) + end term |
| 5 | QET, λ = 0 | quotient | ∅ | ρ(I, v) = m(I) · q(v \| I) | yes | −log B(∅) |
| 6 | QET, λ = 0.5 (default) | quotient | μ (λ = 0.5) | ρ(I, v) = m(I) · q(v \| I) | yes | Σ μ·(−log B) |

Here c(V) = 1 and c(I) = Σ_v c(I ∪ {v}) counts legal completions, so configuration 3 is exactly uniform over
complete topological orders from every start state (D-012); choosing uniformly among ready actions is not.
ℓ(I) = −log Σ_{u ∈ ready(I)} π(a_u | H_I) (method page eq. 5); configuration 4's state weights are its occupancy under
the renormalized policy (D-008), which makes its objective the μ-weighted Jensen upper bound of L.
The finish action has weight 1 in every configuration. Configurations 3–6 use the same states and edges.

Ablations (single seed):

| Id | Ablation | Definition |
| --- | --- | --- |
| a1 | Plain chronological SFT | `native_multiturn` renderer, demo-path edges, no restated values |
| a2 | Canonical-order SFT | quotient renderer, weight 1 on the canonical order: repeatedly the ready node with the smallest Reduce key |
| a3 | Sampled-order marginalization | root only; the demo order plus K−1 random orders; objective −log Σ_k P(σ_k); weights = path posterior (D-011) |
| a4 | Sub-lattice QET, K ∈ {2, 4, 8} | §6.6 with the given K, λ = 0.5 |

In sub-lattice mode all lattice configurations (3–6, a4) use the same sampled S per graph and epoch (D-013).
The facade `qet.core.api.lattice_summary` exposes these as methods `path_sft`, `uniform_aug`, `frontier`, `qet`
and `sampled_paths` (§10).

## 9. Identities that must hold (tests)
On random DAGs with random raw policies (failure mass included):
1. Lattice B(I) equals the sum over all legal completions of their path probabilities (rel. error < 1e-12).
2. ρ = −∂L/∂log π for every edge, finish included (finite differences, abs. error < 1e-7 in float64).
3. Finish weight 1; Σ tool weights = Σ_I μ(I)(n − |I|) for configurations 3–6.
4. Surrogate − objective = Σ_I μ(I) · H(success posterior over completions from I).
5. Sub-lattice: B_S ≤ B ≤ B⁺ on S; all equal when S is the full lattice; μ_S sums to 1.
6. Eq. (4): E_{σ∼ω}[−log p(σ)] = −log B(I) + H(ω) + KL(ω ‖ p(· | C)) for ω uniform over completions from I.
7. Eq. (5): −log B(I) ≤ ℓ(I) + E_{v∼π̄}[−log B(I ∪ {v})]; configuration 4's objective equals the μ-weighted
   unrolled bound and is ≥ L.
8. Continuation: if ν ≤ κμ then E_ν[1 − B] ≤ κ · E_μ[−log B].
9. On a chain with λ = 0, configuration 6 = configuration 2.
10. Render invariance (§5.6 a) for random demos with random contracts.

## 10. Test facade `qet.core.api`
Thin JSON-in/JSON-out wrappers used by `reference/acceptance/` (internals are free):

```python
canonical_json(value) -> str
action_string(tool: str, args: dict) -> str
show_call(tool: str, args: dict) -> str
compile_edges(case: dict) -> list[list]        # case: request, steps, contracts, [initial_observation, task_order, start_values]
canonical_order(case: dict) -> list[int]       # §8 a2, on the compiled graph of `case`
reduce(history: list[dict], contracts: dict) -> list[dict]   # [{"name": str, "obs": canonical obs}] in H order
render(task: dict, history: list[dict], contracts: dict, variant="quotient", truncation=None) -> {"system", "user"}
lattice_summary(n, preds, logp, method, lam, order=None, paths=None, states=None, plus=False) -> dict
```

`lattice_summary` takes `preds` as lists of node ids, `logp` keyed `"<state>|<node>"` / `"<state>|end"` (state =
sorted ids joined by commas, `""` = ∅), optional explicit `states` (a sub-lattice), and returns `logB` and `mu`
(by state key), `weights` (by edge key; omitted edges mean 0), `qet_objective` (Σ μ(−log B) with the given λ),
`surrogate` (Σ w·(−log π), finish included), `objective` (the configuration's logged objective) and
`logB_plus` (when `plus`). Method `path_sft` takes `order`; `sampled_paths` takes `paths`.

## 11. Fixtures index (`reference/fixtures/`)
`canonical.json` (§2) · `compile.json` (§4: running example, seven-step booking, update_order, one case per
rule detail) · `reduce.json` (§5.1, including all 24 legal orders of the seven-step graph) · `render.json`
(§5.2–5.7: every running-example state, both orders, the noisy history with `request_id` and `1200.0`,
chronological prompts, truncation) · `lattice.json` (§6, §8: the method page's example model and the demo-path model, the seven-step
graph exact and sub-lattice, random DAGs with n = 0…7) · `identities.json` (method page §5, §7 examples) ·
`toy_training.json` (§6.5).

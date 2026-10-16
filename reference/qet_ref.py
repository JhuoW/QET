"""QET reference oracle.

Pure Python (3.10+), standard library only, deliberately slow and literal. It exists to
generate the golden fixtures in reference/fixtures/ and to cross-check the real
implementation in src/qet. Production code must never import this module.

Normative description: docs/SPEC.md. Section numbers below refer to that file.
"""
from __future__ import annotations

import json
import math
import random
import re
import unicodedata
from collections import Counter, defaultdict

WILDCARD = "*"
ACTION_KEY_ORDER = "tool_first"  # D-001; "sorted" reproduces the method page literally ({"args":...,"tool":...})

SYSTEM_TEXT = (
    "You are a tool-using agent. Read the task and the completed calls, then output the next action "
    'as one JSON object {"tool": ..., "args": {...}} and nothing else. '
    "Call finish when the task is complete."
)  # SPEC §5.2, D-003

LABELS = ("data", "effect", "decision", "dup", "task")


# --------------------------------------------------------------------------- §2 canonical form
def canon_value(x):
    """Canonical JSON value: NFC strings, sorted object keys, integral floats -> int, -0.0 -> 0."""
    if isinstance(x, bool) or x is None:
        return x
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            raise ValueError("non-finite number in a call or observation")
        if x == 0:
            return 0
        if x.is_integer() and abs(x) < 2**53:
            return int(x)
        return x
    if isinstance(x, str):
        return unicodedata.normalize("NFC", x)
    if isinstance(x, dict):
        out = {}
        for k in x:
            if not isinstance(k, str):
                raise TypeError(f"object keys must be strings, got {type(k).__name__}")
        for k in sorted(x, key=lambda s: unicodedata.normalize("NFC", s)):
            nk = unicodedata.normalize("NFC", k)
            if nk in out:
                raise ValueError(f"duplicate key after NFC normalization: {nk!r}")
            out[nk] = canon_value(x[k])
        return out
    if isinstance(x, (list, tuple)):
        return [canon_value(v) for v in x]
    raise TypeError(f"unsupported type {type(x).__name__}")


def canon_json(x) -> str:
    return json.dumps(canon_value(x), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def action_string(tool: str, args: dict, key_order: str | None = None) -> str:
    """The target string the model generates (the end-of-turn token is appended at tokenization, D-002)."""
    order = key_order or ACTION_KEY_ORDER
    if order == "tool_first":
        return '{"tool":' + json.dumps(canon_value(tool), ensure_ascii=False) + ',"args":' + canon_json(args) + "}"
    if order == "sorted":
        return canon_json({"tool": tool, "args": args})
    raise ValueError(order)


def show_call(tool: str, args: dict) -> str:
    a = canon_value(args)
    return f"{tool}(" + ", ".join(f"{k}={canon_json(a[k])}" for k in sorted(a)) + ")"


def scalar_leaves(x):
    if isinstance(x, dict):
        for k in sorted(x):
            yield from scalar_leaves(x[k])
    elif isinstance(x, list):
        for v in x:
            yield from scalar_leaves(v)
    else:
        yield x


def vkey(v):
    """Typed key so that True, 1 and 1.0 never collide (Python treats True == 1)."""
    if isinstance(v, bool):
        return ("bool", v)
    if v is None:
        return ("null", None)
    if isinstance(v, (int, float)):
        return ("num", canon_value(v))
    if isinstance(v, str):
        return ("str", canon_value(v))
    raise TypeError(type(v).__name__)


# --------------------------------------------------------------------------- §3 contracts
_TEMPLATE = re.compile(r"\{(\w+)\}")


def _fmt_resource(template: str, args: dict) -> str:
    def sub(m):
        name = m.group(1)
        if name not in args:
            raise KeyError(name)
        v = args[name]
        return v if isinstance(v, str) else canon_json(v)

    return _TEMPLATE.sub(sub, template)


def effects(tool: str, args: dict, contracts: dict):
    """(reads, writes) as frozensets of resource patterns. No contract -> reads and writes everything."""
    c = contracts.get(tool)
    if c is None:
        return frozenset({WILDCARD}), frozenset({WILDCARD})
    a = canon_value(args)
    out = []
    for field in ("reads", "writes"):
        res = set()
        for t in c.get(field, []):
            try:
                res.add(_fmt_resource(t, a))
            except KeyError:  # template needs an argument that is absent: be conservative
                res.add(WILDCARD)
        out.append(frozenset(res))
    return out[0], out[1]


def _glob(pattern: str, s: str) -> bool:
    rx = "".join(".*" if ch == "*" else re.escape(ch) for ch in pattern)
    return re.fullmatch(rx, s) is not None


def _meets(A, B) -> bool:
    return any(_glob(a, b) or _glob(b, a) for a in A for b in B)


def conflict_eff(e1, e2) -> bool:
    """Bernstein condition on (reads, writes) pairs: W1∩(R2∪W2) or W2∩R1 non-empty."""
    (r1, w1), (r2, w2) = e1, e2
    return _meets(w1, r2 | w2) or _meets(w2, r1)


def conflict(call1, call2, contracts) -> bool:
    return conflict_eff(effects(*call1, contracts), effects(*call2, contracts))


def _delete_path(obj, parts):
    if not isinstance(obj, dict) or not parts:
        return
    if len(parts) == 1:
        obj.pop(parts[0], None)
    else:
        _delete_path(obj.get(parts[0]), parts[1:])


def canon_obs(tool: str, obs, contracts: dict):
    o = canon_value(obs)
    for path in (contracts.get(tool) or {}).get("volatile", []):
        _delete_path(o, path.split("."))
    return o


# --------------------------------------------------------------------------- §4 compile
_NUM = re.compile(r"(?<![0-9A-Za-z_.])-?\d+(?:\.\d+)?(?![0-9A-Za-z_])")


def request_numbers(text: str):
    return {vkey(float(t) if "." in t else int(t)) for t in _NUM.findall(text)}


def string_in_request(s: str, text: str) -> bool:
    if s == "":
        return True
    return re.search(r"(?<!\w)" + re.escape(s) + r"(?!\w)", text) is not None


def compile_edges(request, steps, contracts, *, initial_observation=None, task_order=(), start_values=()):
    """The five ordering rules of SPEC §4.3. Returns sorted [[u, v, sorted labels]] with u < v."""
    n = len(steps)
    calls = [(s["tool"], canon_value(s["args"])) for s in steps]
    obs = [canon_obs(s["tool"], s["obs"], contracts) for s in steps]
    start = set(request_numbers(request)) | {vkey(canon_value(v)) for v in start_values}
    if initial_observation is not None:
        start |= {vkey(v) for v in scalar_leaves(canon_value(initial_observation))}

    def at_start(v):
        k = vkey(v)
        return k in start or (k[0] == "str" and string_in_request(k[1], request))

    out = [{vkey(v) for v in scalar_leaves(o)} for o in obs]
    acts = [action_string(t, a) for t, a in calls]
    eff = [effects(t, a, contracts) for t, a in calls]
    labels = defaultdict(set)
    for j in range(n):
        for v in scalar_leaves(calls[j][1]):  # rule 1: data dependency
            if at_start(v):
                continue
            src = [i for i in range(j) if vkey(v) in out[i]]
            for i in src or range(j):  # underived value: depend on every earlier call
                labels[(i, j)].add("data")
        for i in range(j):
            if conflict_eff(eff[i], eff[j]):  # rule 2: side-effect conflict
                labels[(i, j)].add("effect")
            if eff[j][1]:  # rule 3: decision dependency (v writes something)
                labels[(i, j)].add("decision")
            if acts[i] == acts[j]:  # rule 4: identical calls keep their order
                labels[(i, j)].add("dup")
    for i, j in task_order:  # rule 5: task constraints
        if not 0 <= i < j < n:
            raise ValueError("a task constraint must point from an earlier to a later step")
        labels[(i, j)].add("task")
    return sorted([i, j, sorted(ls)] for (i, j), ls in labels.items())


def preds_from_edges(n, edges):
    p = [set() for _ in range(n)]
    for u, v, _ in edges:
        p[v].add(u)
    return [frozenset(x) for x in p]


def node_keys(steps):
    """Reduce ordering key (tool, canonical args, duplicate index) of each demo step, in demo order."""
    seen, keys = Counter(), []
    for s in steps:
        a = canon_value(s["args"])
        base = show_call(s["tool"], a)
        seen[base] += 1
        keys.append((s["tool"], canon_json(a), seen[base]))
    return keys


def canonical_order(preds, keys):
    """Topological order that always takes the ready node with the smallest Reduce key (SPEC §8)."""
    n, placed, done = len(keys), [], set()
    while len(placed) < n:
        v = min((v for v in range(n) if v not in done and preds[v] <= done), key=lambda v: keys[v])
        placed.append(v)
        done.add(v)
    return placed


# --------------------------------------------------------------------------- §5 reduce + render
def canonical_entries(history, contracts):
    """Reduce steps 1-2: canonicalize each (call, observation) and name it; execution order kept."""
    seen, E = Counter(), []
    for h in history:
        tool = h["tool"]
        a = canon_value(h["args"])
        base = show_call(tool, a)
        seen[base] += 1
        k = seen[base]
        E.append({"tool": tool, "args": a, "obs": canon_obs(tool, h["obs"], contracts),
                  "name": base if k == 1 else f"{base}#{k}", "key": (tool, canon_json(a), k)})
    return E


def reduce_history(history, contracts):
    """Reduce step 3: smallest-key topological order under the conflict constraints only."""
    E = canonical_entries(history, contracts)
    eff = [effects(e["tool"], e["args"], contracts) for e in E]
    pred = [{i for i in range(j) if conflict_eff(eff[i], eff[j])} for j in range(len(E))]
    H, placed = [], set()
    while len(H) < len(E):
        j = min((j for j in range(len(E)) if j not in placed and pred[j] <= placed), key=lambda j: E[j]["key"])
        placed.add(j)
        H.append(E[j])
    return H


def truncate(x, cfg):
    k, m = cfg.get("max_list_items"), cfg.get("max_str_chars")
    if isinstance(x, dict):
        return {key: truncate(v, cfg) for key, v in x.items()}
    if isinstance(x, list):
        items = [truncate(v, cfg) for v in (x if k is None else x[:k])]
        if k is not None and len(x) > k:
            items.append(f"<+{len(x) - k} more>")
        return items
    if isinstance(x, str) and m is not None and len(x) > m:
        return x[:m] + f"<+{len(x) - m} chars>"
    return x


_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def value_lines(name, obs):
    lines = []

    def walk(x, path, root):
        if isinstance(x, dict) and x:
            for key in sorted(x):
                seg = f".{key}" if _IDENT.fullmatch(key) else f"[{json.dumps(key, ensure_ascii=False)}]"
                walk(x[key], path + seg, False)
        elif isinstance(x, list) and x:
            for i, v in enumerate(x):
                walk(v, f"{path}[{i}]", False)
        elif root and isinstance(x, (dict, list)):
            return  # an empty observation contributes no values
        else:
            lines.append(f"  {name}{path} = {canon_json(x)}")

    walk(obs, "", True)
    return lines


def render(task, history, contracts, variant="quotient", truncation=None):
    """Returns {"system", "user"} (SPEC §5.2). variant "chronological" is configuration 1 (D-004)."""
    cfg = dict(truncation or {})
    if variant == "quotient":
        H = reduce_history(history, contracts)
    elif variant == "chronological":
        H = canonical_entries(history, contracts)
    else:
        raise ValueError(variant)
    obs = [truncate(e["obs"], cfg) for e in H]
    lines = [f"[Task] {task['request']}", "[Tools]"]
    for t in sorted(task["tools"], key=lambda t: t["name"]):
        lines.append(f"  {t['name']}({', '.join(t['params'])}): {t['description']}")
    if variant == "quotient":
        lines.append("[Completed calls]")
        lines += [f"  {e['name']} -> {canon_json(o)}" for e, o in zip(H, obs)] or ["  (none)"]
    else:
        lines.append("[History]")
        lines += [f"  {i}. {e['name']} -> {canon_json(o)}" for i, (e, o) in enumerate(zip(H, obs), 1)] or ["  (none)"]
    lines.append("[Available values]")
    lines += [ln for e, o in zip(H, obs) for ln in value_lines(e["name"], o)] or ["  (none)"]
    lines.append("[Next action]")
    return {"system": SYSTEM_TEXT, "user": "\n".join(lines)}


# --------------------------------------------------------------------------- §6 lattice
def full(n):
    return frozenset(range(n))


def ready(I, preds, n):
    return [v for v in range(n) if v not in I and preds[v] <= I]


def sort_states(states):
    return sorted(states, key=lambda I: (len(I), sorted(I)))


def enumerate_states(preds, n, cap=None):
    """All downsets, or None when there are more than `cap`."""
    seen, queue = {frozenset()}, [frozenset()]
    while queue:
        I = queue.pop(0)
        for v in ready(I, preds, n):
            J = I | {v}
            if J not in seen:
                seen.add(J)
                queue.append(J)
                if cap is not None and len(seen) > cap:
                    return None
    return sort_states(seen)


def start_weights(states, n, lam):
    """mu_G (SPEC §6.3). Layer counts N_k are taken over `states`, so this also serves a sub-lattice."""
    mu = {I: 0.0 for I in states}
    if n <= 1:  # n = 1: no intermediate layer; n = 0: the root is the final state (SPEC §6.3)
        mu[frozenset()] = 1.0
        return mu
    N = Counter(len(I) for I in states if 0 < len(I) < n)
    for I in states:
        k = len(I)
        if k == 0:
            mu[I] = 1.0 - lam
        elif k < n:
            mu[I] = lam / (n - 1) / N[k]
    return mu


def logsumexp(xs):
    xs = [x for x in xs if x != -math.inf]
    if not xs:
        return -math.inf
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


def lattice_logB(states, preds, n, logp, outside=None):
    """Eq. (1) in log space over `states`. outside=None: successors outside count 0 (B_S);
    outside=0.0: they count 1 (B+, the diagnostic upper bound of SPEC §6.6)."""
    S, V = set(states), full(n)
    logB = {}
    for I in sorted(states, key=len, reverse=True):
        if I == V:
            logB[I] = logp[(I, "end")]
            continue
        terms = []
        for v in ready(I, preds, n):
            J = I | {v}
            if J in S:
                terms.append(logp[(I, v)] + logB[J])
            elif outside is not None:
                terms.append(logp[(I, v)] + outside)
        logB[I] = logsumexp(terms)
    return logB


def flow(states, preds, n, mu, step_prob):
    """m(I) = mu(I) + inflow; w(I, v) = m(I) * step_prob(I, v); w(V, end) = m(V)."""
    S, V = set(states), full(n)
    m = {I: mu.get(I, 0.0) for I in states}
    w = {}
    for I in sorted(states, key=len):
        if I == V:
            continue
        for v in ready(I, preds, n):
            J = I | {v}
            if J in S:
                w[(I, v)] = m[I] * step_prob(I, v)
                m[J] += w[(I, v)]
    w[(V, "end")] = m[V]
    return w, m


def completion_counts(states, preds, n):
    S, V = set(states), full(n)
    c = {}
    for I in sorted(states, key=len, reverse=True):
        c[I] = 1 if I == V else sum(c[I | {v}] for v in ready(I, preds, n) if (I | {v}) in S)
    return c


def all_edges(states, preds, n):
    S, V = set(states), full(n)
    E = [(I, v) for I in sort_states(states) if I != V for v in ready(I, preds, n) if (I | {v}) in S]
    return E + [(V, "end")]


def path_logp(order, logp, start=frozenset()):
    I, s = start, 0.0
    for v in order:
        s += logp[(I, v)]
        I = I | {v}
    return s + logp[(I, "end")]


def path_edges(order, start=frozenset()):
    I, out = start, []
    for v in order:
        out.append((I, v))
        I = I | {v}
    return out + [(I, "end")]


def method_weights(method, *, states, preds, n, logp, lam=0.5, order=None, paths=None):
    """Edge weights of the training configurations (SPEC §8). Returns (weights, state_mass)."""
    S = set(states)
    if method in ("qet", "uniform_aug", "frontier"):
        mu = start_weights(states, n, lam)
        if method == "qet":
            logB = lattice_logB(states, preds, n, logp)

            def sp(I, v):
                return math.exp(logp[(I, v)] + logB[I | {v}] - logB[I])
        elif method == "uniform_aug":
            c = completion_counts(states, preds, n)

            def sp(I, v):
                return c[I | {v}] / c[I]
        else:
            def sp(I, v):
                z = sum(math.exp(logp[(I, u)]) for u in ready(I, preds, n) if (I | {u}) in S)
                return math.exp(logp[(I, v)]) / z
        return flow(states, preds, n, mu, sp)
    if method == "path_sft":
        w = {e: 1.0 for e in path_edges(order)}
        return w, None
    if method == "sampled_paths":
        uniq = []
        for p in paths:
            if list(p) not in uniq:
                uniq.append(list(p))
        lps = [path_logp(p, logp) for p in uniq]
        z = logsumexp(lps)
        w = defaultdict(float)
        for p, lp in zip(uniq, lps):
            for e in path_edges(p):
                w[e] += math.exp(lp - z)
        return dict(w), None
    raise ValueError(method)


def summary(method, *, states, preds, n, logp, lam=0.5, order=None, paths=None, plus=False):
    """Everything the acceptance tests compare (SPEC §10)."""
    w, mass = method_weights(method, states=states, preds=preds, n=n, logp=logp, lam=lam, order=order, paths=paths)
    mu = start_weights(states, n, lam)
    logB = lattice_logB(states, preds, n, logp)
    qet_obj = sum(mu[I] * -logB[I] for I in states if mu[I] > 0)
    edges = all_edges(states, preds, n)
    weights = {e: w.get(e, 0.0) for e in edges}
    surrogate = sum(wt * -logp[e] for e, wt in weights.items() if wt != 0.0)
    S, V = set(states), full(n)
    if method == "qet":
        objective = qet_obj
    elif method == "frontier":
        def local(I):
            return -math.log(sum(math.exp(logp[(I, u)]) for u in ready(I, preds, n) if (I | {u}) in S))
        objective = sum(mass[I] * local(I) for I in states if I != V) + mass[V] * -logp[(V, "end")]
    elif method == "sampled_paths":
        uniq = []
        for p in paths:
            if list(p) not in uniq:
                uniq.append(list(p))
        objective = -logsumexp([path_logp(p, logp) for p in uniq])
    else:  # path_sft, uniform_aug: the weighted negative log-likelihood itself
        objective = surrogate
    out = {"logB": logB, "mu": mu, "weights": weights, "qet_objective": qet_obj,
           "surrogate": surrogate, "objective": objective, "logB_plus": None}
    if plus:
        out["logB_plus"] = lattice_logB(states, preds, n, logp, outside=0.0)
    return out


# --------------------------------------------------------------------------- §6.6 sub-lattice
def random_topological_order(preds, n, rng):
    I, order = frozenset(), []
    while len(order) < n:
        v = rng.choice(sorted(ready(I, preds, n)))
        order.append(v)
        I = I | {v}
    return order


def sublattice_states(orders):
    S = {frozenset()}
    for o in orders:
        I = frozenset()
        for v in o:
            I = I | {v}
            S.add(I)
    return sort_states(S)


# --------------------------------------------------------------------------- path enumeration
def all_orders(preds, n, start=frozenset()):
    def rec(I, prefix):
        if len(I) == n:
            yield list(prefix)
            return
        for v in ready(I, preds, n):
            yield from rec(I | {v}, prefix + [v])

    yield from rec(start, [])


# --------------------------------------------------------------------------- toy table policy (SPEC §6.5 test)
def table_train(preds, n, p0, method, steps, lr, lam=0.5, order=None, record=(0, 1, 5, 20, 50, 200)):
    """Each state's prompt is an independent softmax over its ready targets plus one 'wrong' bucket.
    One step = gradient descent on the weighted SFT loss with weights from `method` (held constant)."""
    states = enumerate_states(preds, n)
    V = full(n)
    targets = {I: (["end"] if I == V else ready(I, preds, n)) for I in states}
    z = {}
    for I in states:
        ps = [p0[(I, a)] for a in targets[I]]
        z[I] = [math.log(p) for p in ps] + [math.log(1.0 - sum(ps))]
    hist = []
    for t in range(steps + 1):
        probs, logp = {}, {}
        for I in states:
            mx = max(z[I])
            e = [math.exp(x - mx) for x in z[I]]
            s = sum(e)
            pv = [x / s for x in e]
            probs[I] = pv
            for k, a in enumerate(targets[I]):
                logp[(I, a)] = math.log(pv[k])
        w, _ = method_weights(method, states=states, preds=preds, n=n, logp=logp, lam=lam, order=order)
        logB = lattice_logB(states, preds, n, logp)
        mu = start_weights(states, n, lam)
        if t in record:
            hist.append({"step": t,
                         "pi": {f"{skey(I)}|{a}": math.exp(logp[(I, a)]) for I in states for a in targets[I]},
                         "B": {skey(I): math.exp(logB[I]) for I in states},
                         "qet_objective": sum(mu[I] * -logB[I] for I in states if mu[I] > 0)})
        for I in states:
            wv = [w.get((I, a), 0.0) for a in targets[I]] + [0.0]
            W = sum(wv)
            z[I] = [zi - lr * (W * pi - wi) for zi, pi, wi in zip(z[I], probs[I], wv)]
    return hist


# --------------------------------------------------------------------------- JSON keys
def skey(I):
    return ",".join(str(v) for v in sorted(I))


def ekey(I, v):
    return f"{skey(I)}|{v}"


def parse_skey(s):
    return frozenset(int(x) for x in s.split(",")) if s else frozenset()


def parse_ekey(s):
    st, v = s.split("|")
    return parse_skey(st), (v if v == "end" else int(v))


# --------------------------------------------------------------------------- random instances (tests)
def random_dag(n, p_edge, rng):
    return [frozenset(i for i in range(j) if rng.random() < p_edge) for j in range(n)]


def random_policy(states, preds, n, rng, fail=(0.0, 0.5)):
    """Raw model probabilities: each state's legal actions share 1 - f of the mass (f = failure mass)."""
    V, logp = full(n), {}
    for I in states:
        if I == V:
            logp[(I, "end")] = math.log(rng.uniform(0.5, 0.99))
            continue
        R = ready(I, preds, n)
        ws = [rng.random() + 0.05 for _ in R]
        legal = 1.0 - rng.uniform(*fail)
        for v, x in zip(R, ws):
            logp[(I, v)] = math.log(legal * x / sum(ws))
    return logp


def random_demo(rng, n_steps, n_tools=5, n_res=3):
    """A random demonstration with random contracts (for invariance and compile property tests)."""
    res = [f"r{i}" for i in range(n_res)]
    contracts = {}
    for t in range(n_tools):
        kind = rng.random()
        reads = rng.sample(res, rng.randint(0, 2))
        writes = rng.sample(res, rng.randint(1, 2)) if kind < 0.3 else []
        if kind > 0.9:
            continue  # no contract: reads and writes everything
        contracts[f"tool{t}"] = {"reads": reads, "writes": writes, "volatile": ["ts"]}
    steps, produced = [], []
    for _ in range(n_steps):
        tool = f"tool{rng.randrange(n_tools)}"
        if produced and rng.random() < 0.5:
            a = rng.choice(produced)
        else:
            a = rng.choice([7, 11, 13, rng.randint(100, 999)])
        out = rng.randint(100, 999)
        produced.append(out)
        steps.append({"tool": tool, "args": {"x": a}, "obs": {"y": out, "ts": rng.randint(0, 10**6)}})
    return "Use the values 7, 11 and 13.", steps, contracts

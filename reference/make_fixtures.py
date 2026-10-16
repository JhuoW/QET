"""Generate the golden fixtures in reference/fixtures/ from the oracle (qet_ref.py).

    python reference/make_fixtures.py          # (re)write fixtures
    python reference/make_fixtures.py --check  # exit 1 if any fixture on disk differs

Fixtures are normative (docs/SPEC.md §11). Changing one needs an accepted entry in docs/DECISIONS.md.
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import qet_ref as R  # noqa: E402

OUT = HERE / "fixtures"

# ----------------------------------------------------------------------------- shared examples
RUN_REQUEST = ("Find the price of a flight from NYC to SG and the price of 3 nights at a hotel in SG, "
               "then compute the total cost.")
RUN_TOOLS = [
    {"name": "get_flight_price", "params": ["from", "to"], "description": "Look up a flight price"},
    {"name": "get_hotel_price", "params": ["city", "nights"], "description": "Look up a hotel price"},
    {"name": "calculate_total", "params": ["flight_price", "hotel_price"], "description": "Compute the total cost"},
    {"name": "finish", "params": ["answer"], "description": "Submit the final answer"},
]
RUN_CONTRACTS = {
    "get_flight_price": {"reads": ["flights"], "writes": [], "volatile": ["request_id"]},
    "get_hotel_price": {"reads": ["hotels"], "writes": [], "volatile": ["request_id"]},
    "calculate_total": {"reads": [], "writes": []},
}
A = {"tool": "get_flight_price", "args": {"from": "NYC", "to": "SG"}, "obs": {"price": 1200}}
B = {"tool": "get_hotel_price", "args": {"city": "SG", "nights": 3}, "obs": {"price": 600}}
C = {"tool": "calculate_total", "args": {"flight_price": 1200, "hotel_price": 600}, "obs": {"total": 1800}}
RUN_STEPS = [A, B, C]
RUN_FINAL = {"tool": "finish", "args": {"answer": 1800}}

SEVEN_REQUEST = ("Check the flight price from NYC to SG, the hotel price for 3 nights in SG and the weather in SG; "
                 "pay for the flight from account 1, check the balance of account 1 before and after paying, "
                 "and finally book the flight.")
SEVEN_STEPS = [
    {"tool": "get_flight_price", "args": {"from": "NYC", "to": "SG"}, "obs": {"price": 1200}},
    {"tool": "get_weather", "args": {"city": "SG"}, "obs": {"temp": 31}},
    {"tool": "get_balance", "args": {"account": 1}, "obs": {"balance": 2000}},
    {"tool": "get_hotel_price", "args": {"city": "SG", "nights": 3}, "obs": {"price": 600}},
    {"tool": "pay", "args": {"account": 1, "amount": 1200}, "obs": {"ok": True}},
    {"tool": "get_balance", "args": {"account": 1}, "obs": {"balance": 800}},
    {"tool": "book_flight", "args": {"from": "NYC", "to": "SG"}, "obs": {"status": "booked"}},
]
SEVEN_CONTRACTS = {
    "get_flight_price": {"reads": ["flights"], "writes": []},
    "book_flight": {"reads": ["flights"], "writes": ["flights"]},
    "get_balance": {"reads": ["acct:{account}"], "writes": []},
    "pay": {"reads": ["acct:{account}"], "writes": ["acct:{account}"]},
    "get_weather": {"reads": ["weather"], "writes": []},
    "get_hotel_price": {"reads": ["hotels"], "writes": []},
}

ORDER_REQUEST = "Mark order 7 as paid, check the NYC to SG flight price, mark order 7 as shipped, then read order 7."
ORDER_STEPS = [
    {"tool": "update_order", "args": {"id": 7, "status": "paid"}, "obs": {"ok": True}},
    {"tool": "get_flight_price", "args": {"from": "NYC", "to": "SG"}, "obs": {"price": 1200}},
    {"tool": "update_order", "args": {"id": 7, "status": "shipped"}, "obs": {"ok": True}},
    {"tool": "get_order", "args": {"id": 7}, "obs": {"status": "shipped"}},
]
ORDER_CONTRACTS = {
    "update_order": {"reads": ["order:{id}"], "writes": ["order:{id}"]},
    "get_order": {"reads": ["order:{id}"], "writes": []},
    "get_flight_price": {"reads": ["flights"], "writes": []},
}


def S(*xs):
    return frozenset(xs)


def enc_logp(logp):
    return {R.ekey(I, v): x for (I, v), x in logp.items()}


def enc_states(d):
    return {R.skey(I): x for I, x in d.items()}


def enc_weights(d):
    return {R.ekey(I, v): x for (I, v), x in d.items()}


def summary_json(out):
    return {"logB": enc_states(out["logB"]), "mu": enc_states(out["mu"]), "weights": enc_weights(out["weights"]),
            "qet_objective": out["qet_objective"], "surrogate": out["surrogate"], "objective": out["objective"],
            "logB_plus": enc_states(out["logB_plus"]) if out["logB_plus"] is not None else None}


def lattice_run(preds, n, logp, method, lam, *, order=None, paths=None, states=None, plus=False):
    st = states if states is not None else R.enumerate_states(preds, n)
    out = R.summary(method, states=st, preds=preds, n=n, logp=logp, lam=lam, order=order, paths=paths, plus=plus)
    run = {"method": method, "lam": lam}
    if order is not None:
        run["order"] = order
    if paths is not None:
        run["paths"] = paths
    if states is not None:
        run["states"] = [sorted(I) for I in states]
    if plus:
        run["plus"] = True
    run["expected"] = summary_json(out)
    return run


def table_logp(table):
    return {k: math.log(v) for k, v in table.items()}


DOC_PI = {(S(), 0): 0.60, (S(), 1): 0.30, (S(0), 1): 0.50, (S(1), 0): 0.90, (S(0, 1), 2): 0.80, (S(0, 1, 2), "end"): 0.95}
DEMO_PATH_PI = {(S(), 0): 0.90, (S(), 1): 0.05, (S(0), 1): 0.90, (S(1), 0): 0.10, (S(0, 1), 2): 0.90,
                (S(0, 1, 2), "end"): 0.95}
RUN_PREDS = [S(), S(), S(0, 1)]


# ----------------------------------------------------------------------------- fixture builders
def build_canonical():
    values = [1200.0, -0.0, 2.5, 1e16, 7, True, None, "é", {"to": "SG", "from": "NYC"},
              {"b": [1.0, {"d": True, "c": None}], "a": "x"}, [3, 2.0, "1"]]
    calls = [
        ("get_flight_price", {"to": "SG", "from": "NYC"}),
        ("calculate_total", {"hotel_price": 600.0, "flight_price": 1200}),
        ("finish", {"answer": 1800}),
        ("ping", {}),
        ("search", {"q": {"z": 1, "a": [2, 3.0]}, "limit": 10}),
        ("send_email", {"to": "me", "body": "Total: 1800 SGD\nThanks"}),
    ]
    return {
        "description": "Canonical JSON values, action strings (D-001 tool-first) and call names (SPEC §2).",
        "values": [{"input": v, "canonical_json": R.canon_json(v)} for v in values],
        "calls": [{"tool": t, "args": a, "action_string": R.action_string(t, a),
                   "action_string_sorted_variant": R.action_string(t, a, "sorted"), "show_call": R.show_call(t, a)}
                  for t, a in calls],
    }


def compile_case(name, request, steps, contracts, note, **kw):
    edges = R.compile_edges(request, steps, contracts, **kw)
    n = len(steps)
    preds = R.preds_from_edges(n, edges)
    states = R.enumerate_states(preds, n)
    inp = {"request": request, "steps": steps, "contracts": contracts}
    for k in ("initial_observation", "task_order", "start_values"):
        if k in kw:
            inp[k] = [list(x) for x in kw[k]] if k == "task_order" else kw[k]
    return {"name": name, "note": note, "input": inp, "expected_edges": edges,
            "expected_canonical_order": R.canonical_order(preds, R.node_keys(steps)),
            "expected_n_states": len(states), "expected_n_orders": sum(1 for _ in R.all_orders(preds, n))}


def build_compile():
    f = {"tool": "f", "args": {}, "obs": {"x": 5}}
    g = {"tool": "g", "args": {}, "obs": {"y": 5}}
    pure = {"f": {"reads": [], "writes": []}, "g": {"reads": [], "writes": []}, "h": {"reads": [], "writes": []}}
    cases = [
        compile_case("running", RUN_REQUEST, RUN_STEPS, RUN_CONTRACTS, "Method page §4 worked example: A->C, B->C (data)."),
        compile_case("seven_step", SEVEN_REQUEST, SEVEN_STEPS, SEVEN_CONTRACTS,
                     "Method page §5 seven-step booking example; pay and book_flight are writes (rule 3)."),
        compile_case("update_order", ORDER_REQUEST, ORDER_STEPS, ORDER_CONTRACTS,
                     "Two writes to order:7 keep their order (rule 2); the read after them too."),
        compile_case("data_multi_match", "Compute h.", [f, g, {"tool": "h", "args": {"a": 5}, "obs": {"z": 1}}], pure,
                     "A value produced by two earlier calls creates an edge from each of them."),
        compile_case("data_underived", "Compute h.", [f, g, {"tool": "h", "args": {"a": 99}, "obs": {"z": 1}}], pure,
                     "A value in neither the request nor any earlier output depends on every earlier call."),
        compile_case("data_from_request", "Compute h of 99.", [f, g, {"tool": "h", "args": {"a": 99}, "obs": {"z": 1}}],
                     pure, "A value found in the request is available from the start: no edge."),
        compile_case("start_values", "Compute h.", [f, g, {"tool": "h", "args": {"a": 99}, "obs": {"z": 1}}], pure,
                     "Adapter-declared start values (D-005) behave like request values.", start_values=[99]),
        compile_case("initial_observation", "Compute h.", [f, g, {"tool": "h", "args": {"a": 42}, "obs": {"z": 1}}],
                     pure, "Leaves of the initial observation are available from the start.",
                     initial_observation={"inbox": [{"id": 42}]}),
        compile_case("decision", "Read r1 then write r2.",
                     [{"tool": "read1", "args": {}, "obs": {"v": 1}}, {"tool": "write2", "args": {}, "obs": {"ok": True}}],
                     {"read1": {"reads": ["r1"], "writes": []}, "write2": {"reads": [], "writes": ["r2"]}},
                     "A write never moves ahead of any earlier call (rule 3), even without a conflict."),
        compile_case("unknown_tool", "Go.",
                     [{"tool": "read1", "args": {}, "obs": {"v": 1}}, {"tool": "mystery", "args": {}, "obs": {"v": 2}},
                      {"tool": "read3", "args": {}, "obs": {"v": 3}}],
                     {"read1": {"reads": ["r1"], "writes": []}, "read3": {"reads": ["r3"], "writes": []}},
                     "A tool without a contract reads and writes everything."),
        compile_case("pure_after_unknown", "Go with 4.",
                     [{"tool": "mystery", "args": {}, "obs": {"v": 2}}, {"tool": "calc", "args": {"a": 4}, "obs": {"v": 8}}],
                     {"calc": {"reads": [], "writes": []}},
                     "A pure tool (no reads, no writes) commutes even with an unknown tool."),
        compile_case("glob_resource", "List orders, then update order 7.",
                     [{"tool": "list_orders", "args": {}, "obs": {"ids": [7, 8]}},
                      {"tool": "update_order", "args": {"id": 7}, "obs": {"ok": True}}],
                     {"list_orders": {"reads": ["order:*"], "writes": []},
                      "update_order": {"reads": ["order:{id}"], "writes": ["order:{id}"]}},
                     "Resource patterns use * as a glob: order:* meets order:7."),
        compile_case("volatile_is_not_a_source", "Go.",
                     [{"tool": "f", "args": {}, "obs": {"id": 41, "ts": "2026-10-16T10:00:00Z"}},
                      {"tool": "g", "args": {}, "obs": {"id": 42}},
                      {"tool": "h", "args": {"t": "2026-10-16T10:00:00Z"}, "obs": {"ok": True}}],
                     {"f": {"reads": [], "writes": [], "volatile": ["ts"]}, "g": {"reads": [], "writes": []},
                      "h": {"reads": [], "writes": []}},
                     "Volatile fields are dropped before matching, so a value taken from one is underived."),
        compile_case("task_constraint", "Back up, then edit.",
                     [{"tool": "backup", "args": {}, "obs": {"ok": True}}, {"tool": "lookup", "args": {}, "obs": {"v": 1}}],
                     {"backup": {"reads": ["files"], "writes": []}, "lookup": {"reads": ["db"], "writes": []}},
                     "Rule 5: an annotated task order adds an edge.", task_order=[(0, 1)]),
        compile_case("duplicate_reads", "Read x=1 twice.",
                     [{"tool": "get", "args": {"x": 1}, "obs": {"v": 1}}, {"tool": "get", "args": {"x": 1}, "obs": {"v": 1}}],
                     {"get": {"reads": ["r"], "writes": []}}, "Rule 4: identical calls keep their order."),
        compile_case("bool_is_matched", "Check, then act.",
                     [{"tool": "check", "args": {}, "obs": {"ok": True}}, {"tool": "act", "args": {"flag": True}, "obs": {"v": 1}}],
                     {"check": {"reads": [], "writes": []}, "act": {"reads": [], "writes": []}},
                     "Booleans are values too: true produced earlier creates a data edge (conservative, D-005)."),
    ]
    return {"description": "Compile rules (SPEC §4). Edges are [u, v, sorted labels] with u earlier than v.",
            "cases": cases}


def build_reduce():
    preds = R.preds_from_edges(7, R.compile_edges(SEVEN_REQUEST, SEVEN_STEPS, SEVEN_CONTRACTS))
    orders = list(R.all_orders(preds, 7))
    seven_names = [e["name"] for e in R.reduce_history(SEVEN_STEPS, SEVEN_CONTRACTS)]
    for o in orders:
        assert [e["name"] for e in R.reduce_history([SEVEN_STEPS[i] for i in o], SEVEN_CONTRACTS)] == seven_names
    noisy = [{"tool": "get_hotel_price", "args": {"city": "SG", "nights": 3}, "obs": {"price": 600, "request_id": "r-18"}},
             {"tool": "get_flight_price", "args": {"to": "SG", "from": "NYC"}, "obs": {"price": 1200.0, "request_id": "r-93"}}]
    hist_order = [ORDER_STEPS[0], ORDER_STEPS[1], ORDER_STEPS[2], ORDER_STEPS[3]]
    cases = [
        {"name": "seven_step_demo", "history": SEVEN_STEPS, "contracts": SEVEN_CONTRACTS, "expected_names": seven_names},
        {"name": "update_order", "history": hist_order, "contracts": ORDER_CONTRACTS,
         "expected_names": [e["name"] for e in R.reduce_history(hist_order, ORDER_CONTRACTS)]},
        {"name": "noisy_inference_history", "history": noisy, "contracts": RUN_CONTRACTS,
         "expected_names": [e["name"] for e in R.reduce_history(noisy, RUN_CONTRACTS)],
         "expected_obs": [e["obs"] for e in R.reduce_history(noisy, RUN_CONTRACTS)]},
    ]
    return {"description": "Reduce (SPEC §5.1): entry names in H order. Every legal order of the seven-step "
                           "graph must give the same names.",
            "cases": cases,
            "seven_step_legal_orders": {"steps": SEVEN_STEPS, "contracts": SEVEN_CONTRACTS, "orders": orders,
                                        "expected_names": seven_names}}


def build_render():
    task = {"request": RUN_REQUEST, "tools": RUN_TOOLS}
    noisy = [{"tool": "get_hotel_price", "args": {"city": "SG", "nights": 3}, "obs": {"price": 600, "request_id": "r-18"}},
             {"tool": "get_flight_price", "args": {"to": "SG", "from": "NYC"}, "obs": {"price": 1200.0, "request_id": "r-93"}}]
    hists = {"empty": [], "A": [A], "B": [B], "A,B": [A, B], "B,A": [B, A], "A,B,C": [A, B, C], "B,A,C": [B, A, C],
             "noisy_B,A": noisy}
    cases = []
    for name, h in hists.items():
        cases.append({"name": f"quotient:{name}", "history": h, "variant": "quotient",
                      "expected": R.render(task, h, RUN_CONTRACTS)})
    for name in ("empty", "A", "A,B", "B,A", "A,B,C"):
        cases.append({"name": f"chronological:{name}", "history": hists[name], "variant": "chronological",
                      "expected": R.render(task, hists[name], RUN_CONTRACTS, variant="chronological")})
    ttask = {"request": "Summarize the inbox.", "tools": [{"name": "list_mail", "params": [], "description": "List mail"}]}
    th = [{"tool": "list_mail", "args": {}, "obs": {"items": ["a", "b", "c", "d", "e"], "note": "abcdefghij",
                                                     "meta": {"next page": 2}}}]
    tcfg = {"max_list_items": 2, "max_str_chars": 5}
    cases.append({"name": "truncation", "task": ttask, "contracts": {"list_mail": {"reads": ["mail"], "writes": []}},
                  "history": th, "variant": "quotient", "truncation": tcfg,
                  "expected": R.render(ttask, th, {"list_mail": {"reads": ["mail"], "writes": []}}, truncation=tcfg)})
    targets = {}
    preds = RUN_PREDS
    for I in R.enumerate_states(preds, 3):
        if I == R.full(3):
            targets[R.skey(I)] = [R.action_string(RUN_FINAL["tool"], RUN_FINAL["args"])]
        else:
            targets[R.skey(I)] = [R.action_string(RUN_STEPS[v]["tool"], RUN_STEPS[v]["args"]) for v in R.ready(I, preds, 3)]
    return {"description": "Render (SPEC §5.2-5.7). Cases without their own task/contracts use the running example's. "
                           "Pairs that reach the same completed set must be byte-identical in the quotient variant.",
            "task": task, "contracts": RUN_CONTRACTS, "cases": cases,
            "same_prompt_groups": [["quotient:A,B", "quotient:B,A", "quotient:noisy_B,A"], ["quotient:A,B,C", "quotient:B,A,C"]],
            "running_targets_by_state": targets}


def build_lattice():
    cases = []
    doc = table_logp(DOC_PI)
    cases.append({"name": "running_doc_model", "note": "Method page §6.1-6.3 numbers: B(empty)=0.4332, L=0.6553 at lambda=0.5.",
                  "n": 3, "preds": [sorted(p) for p in RUN_PREDS], "demo_order": [0, 1, 2], "logp": enc_logp(doc),
                  "runs": [lattice_run(RUN_PREDS, 3, doc, "qet", 0.5), lattice_run(RUN_PREDS, 3, doc, "qet", 0.0),
                           lattice_run(RUN_PREDS, 3, doc, "uniform_aug", 0.5),
                           lattice_run(RUN_PREDS, 3, doc, "frontier", 0.5),
                           lattice_run(RUN_PREDS, 3, doc, "path_sft", 0.5, order=[0, 1, 2]),
                           lattice_run(RUN_PREDS, 3, doc, "sampled_paths", 0.0, paths=[[0, 1, 2], [1, 0, 2]])]})
    dp = table_logp(DEMO_PATH_PI)
    cases.append({"name": "running_demo_path_model", "note": "A model that learned only the demo path A->B->C.",
                  "n": 3, "preds": [sorted(p) for p in RUN_PREDS], "demo_order": [0, 1, 2], "logp": enc_logp(dp),
                  "runs": [lattice_run(RUN_PREDS, 3, dp, "qet", 0.5), lattice_run(RUN_PREDS, 3, dp, "qet", 0.0),
                           lattice_run(RUN_PREDS, 3, dp, "path_sft", 0.5, order=[0, 1, 2])]})
    # seven-step graph with a random policy, exact and sub-lattice
    rng = random.Random(7)
    sp = R.preds_from_edges(7, R.compile_edges(SEVEN_REQUEST, SEVEN_STEPS, SEVEN_CONTRACTS))
    st = R.enumerate_states(sp, 7)
    lp = R.random_policy(st, sp, 7, rng)
    orders = [list(range(7))] + [R.random_topological_order(sp, 7, rng) for _ in range(2)]
    sub = R.sublattice_states(orders)
    canon = R.canonical_order(sp, R.node_keys(SEVEN_STEPS))
    cases.append({"name": "seven_step_random_policy", "note": "19 states; the sub-lattice run uses 3 orders (demo + 2).",
                  "n": 7, "preds": [sorted(p) for p in sp], "demo_order": list(range(7)), "logp": enc_logp(lp),
                  "runs": [lattice_run(sp, 7, lp, "qet", 0.5), lattice_run(sp, 7, lp, "uniform_aug", 0.5),
                           lattice_run(sp, 7, lp, "frontier", 0.5), lattice_run(sp, 7, lp, "path_sft", 0.5, order=canon),
                           lattice_run(sp, 7, lp, "qet", 0.5, states=sub, plus=True),
                           lattice_run(sp, 7, lp, "uniform_aug", 0.5, states=sub)]})
    for n, p_edge, seed in [(0, 0.0, 1), (1, 0.0, 2), (2, 0.0, 3), (5, 0.3, 4), (6, 0.15, 5), (7, 0.25, 6)]:
        rng = random.Random(1000 + seed)
        preds = R.random_dag(n, p_edge, rng)
        st = R.enumerate_states(preds, n)
        lp = R.random_policy(st, preds, n, rng)
        runs = [lattice_run(preds, n, lp, "qet", 0.5), lattice_run(preds, n, lp, "qet", 0.0),
                lattice_run(preds, n, lp, "uniform_aug", 0.5), lattice_run(preds, n, lp, "frontier", 0.5),
                lattice_run(preds, n, lp, "path_sft", 0.5, order=list(range(n)))]
        if n >= 2:
            paths = [list(range(n))] + [R.random_topological_order(preds, n, rng) for _ in range(2)]
            runs.append(lattice_run(preds, n, lp, "sampled_paths", 0.0, paths=paths))
        if n >= 5:
            orders = [list(range(n))] + [R.random_topological_order(preds, n, rng) for _ in range(2)]
            runs.append(lattice_run(preds, n, lp, "qet", 0.5, states=R.sublattice_states(orders), plus=True))
        cases.append({"name": f"random_n{n}_seed{seed}", "note": f"{len(st)} states", "n": n,
                      "preds": [sorted(p) for p in preds], "demo_order": list(range(n)), "logp": enc_logp(lp), "runs": runs})
    return {"description": "Lattice quantities and edge weights of every training configuration (SPEC §6, §8, §10). "
                           "State keys are sorted node ids joined by commas ('' = empty set); edge keys are "
                           "'<state>|<node>' or '<state>|end'. Edges absent from 'weights' have weight 0.",
            "cases": cases}


def build_identities():
    pref = table_logp({(S(), 0): 0.5, (S(), 1): 0.5, (S(0), 1): 0.9, (S(1), 0): 0.1, (S(0, 1), 2): 1.0, (S(0, 1, 2), "end"): 1.0})
    bal = table_logp({(S(), 0): 0.5, (S(), 1): 0.5, (S(0), 1): 0.5, (S(1), 0): 0.5, (S(0, 1), 2): 1.0, (S(0, 1, 2), "end"): 1.0})
    st = R.enumerate_states(RUN_PREDS, 3)
    out = {}
    for name, lp in (("prefers_ABC", pref), ("balanced", bal)):
        q = R.summary("qet", states=st, preds=RUN_PREDS, n=3, logp=lp, lam=0.0)
        a = R.summary("uniform_aug", states=st, preds=RUN_PREDS, n=3, logp=lp, lam=0.0)
        p1 = math.exp(R.path_logp([0, 1, 2], lp))
        p2 = math.exp(R.path_logp([1, 0, 2], lp))
        post = [p1 / (p1 + p2), p2 / (p1 + p2)]
        kl = sum(0.5 * math.log(0.5 / x) for x in post)
        out[name] = {"logp": enc_logp(lp), "qet_root_objective": q["objective"], "uniform_aug_objective": a["objective"],
                     "entropy_omega": math.log(2), "kl_omega_posterior": kl}
    doc = table_logp(DOC_PI)
    q = R.summary("qet", states=st, preds=RUN_PREDS, n=3, logp=doc, lam=0.0)
    lb = q["logB"]
    zA = math.exp(doc[(S(), 0)]) + math.exp(doc[(S(), 1)])
    local_root = -math.log(zA) + (math.exp(doc[(S(), 0)]) / zA) * -lb[S(0)] + (math.exp(doc[(S(), 1)]) / zA) * -lb[S(1)]
    p = DOC_PI
    true = p[(S(), 0)] * p[(S(0), 1)] * 0.8 * 0.95 + p[(S(), 1)] * p[(S(1), 0)] * 0.6 * 0.95
    return {"description": "Numerical examples of the method page §5 and §7 (eq. 4 and eq. 5).",
            "augmentation_eq4": out,
            "local_bound_eq5": {"neglogB_root": -lb[S()], "local_upper_bound_root": local_root},
            "chronological_counterexample": {
                "pi_C_after_A_then_B": 0.8, "pi_C_after_B_then_A": 0.6, "true_completion_probability": true,
                "merged_with_0.8": p[(S(), 0)] * p[(S(0), 1)] * 0.8 * 0.95 + p[(S(), 1)] * p[(S(1), 0)] * 0.8 * 0.95,
                "merged_with_0.6": p[(S(), 0)] * p[(S(0), 1)] * 0.6 * 0.95 + p[(S(), 1)] * p[(S(1), 0)] * 0.6 * 0.95}}


def build_toy_training():
    runs = []
    for init_name, table in (("doc_model", DOC_PI), ("demo_path_model", DEMO_PATH_PI)):
        for method, kw in (("qet", {"lam": 0.5}), ("path_sft", {"lam": 0.5, "order": [0, 1, 2]})):
            hist = R.table_train(RUN_PREDS, 3, table, method, 200, 0.5, **kw)
            runs.append({"init": init_name, "init_pi": {R.ekey(I, v): x for (I, v), x in table.items()},
                         "method": method, **kw, "lr": 0.5, "steps": 200, "history": hist})
    return {"description": "Table policy (one softmax per state over its targets plus a 'wrong' bucket) trained by "
                           "gradient descent on the weighted SFT loss with weights recomputed every step (SPEC §6.5). "
                           "QET learns the off-demo state {B}; demo SFT never touches it.",
            "preds": [sorted(p) for p in RUN_PREDS], "runs": runs}


BUILDERS = {"canonical.json": build_canonical, "compile.json": build_compile, "reduce.json": build_reduce,
            "render.json": build_render, "lattice.json": build_lattice, "identities.json": build_identities,
            "toy_training.json": build_toy_training}


def dumps(obj, name):
    return json.dumps(obj, indent=1, ensure_ascii=(name == "canonical.json")) + "\n"


def main(argv):
    check = "--check" in argv
    OUT.mkdir(exist_ok=True)
    bad = []
    for name, fn in BUILDERS.items():
        text = dumps(fn(), name)
        path = OUT / name
        if check:
            if not path.exists() or path.read_text(encoding="utf-8") != text:
                bad.append(name)
        else:
            path.write_text(text, encoding="utf-8")
    if check and bad:
        print("fixtures out of date:", ", ".join(bad))
        return 1
    print("fixtures", "up to date" if check else "written", "->", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

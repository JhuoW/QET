"""Evaluation pieces (EVAL_PLAN §6-§7): Resume@prefix starts and restoration, and the metrics of episode rows."""

import random

import pytest

from qet.core.canonical import action_string
from qet.core.lattice import node_masks
from qet.envs.toy import ToyEnv, compile_toy, make_tasks
from qet.eval.metrics import pass_k, summarize
from qet.eval.resume_prefix import evaluate_resume, restore, starts
from qet.infer.agent import ScriptedPolicy, run_episode
from qet.train.chat import ChatFormat


@pytest.fixture(scope="module")
def fmt():
    return ChatFormat.load("Qwen/Qwen2.5-0.5B-Instruct")


def toys():
    return make_tasks("functions", range(30), n_core=5, n_given=2) + make_tasks("stateful", range(30))


def test_starts_are_balanced_legal_and_off_path():
    for toy in toys():
        record, _ = compile_toy(toy)
        masks = node_masks(record.n, record.preds())
        ss = starts(record, seed=0)
        assert ss[0].condition == "empty" and ss[0].order == ()
        by_depth = {}
        for s in ss[1:]:
            by_depth.setdefault(s.depth, {})[s.condition] = s
            done = 0
            for v in s.order:  # a legal order of a legal set
                assert masks[v] & ~done == 0
                done |= 1 << v
            assert len(s.order) == s.depth
        for d, pair in by_depth.items():
            assert set(pair) == {"on_path", "off_path"}  # balanced across depths
            assert pair["on_path"].order == tuple(range(d)) and set(pair["off_path"].order) != set(range(d))
        assert starts(record, seed=0) == ss  # seeded


def test_a_chain_has_only_the_empty_start():
    toy = make_tasks("functions", [3], n_core=1, n_given=1, n_distractors=0)[0]
    record, _ = compile_toy(toy)
    assert [s.condition for s in starts(record, seed=0)] == ["empty"]


def test_every_start_is_restored_and_a_legal_continuation_succeeds(fmt):
    rng = random.Random(0)
    for toy in toys()[:6] + toys()[30:36]:
        record, _ = compile_toy(toy)
        masks = node_masks(record.n, record.preds())
        for s in starts(record, seed=1):
            env = ToyEnv(toy)
            prefix = restore(env, record, s)
            assert [(p["tool"], p["args"]) for p in prefix] == [(record.nodes[v].tool, record.nodes[v].args)
                                                                 for v in s.order]  # fmt: skip
            done, rest = sum(1 << v for v in s.order), []
            while done != (1 << record.n) - 1:  # a random legal continuation of the prefix
                v = rng.choice([u for u in range(record.n) if not done >> u & 1 and masks[u] & ~done == 0])
                rest.append(v)
                done |= 1 << v
            actions = [action_string(record.nodes[v].tool, record.nodes[v].args) for v in rest]
            actions.append(action_string(record.final["tool"], record.final["args"]))
            ep = run_episode(env, ScriptedPolicy(fmt, actions), fmt, prefix=prefix)
            assert ep.success and ep.prefix == s.depth and ep.calls == record.n - s.depth


def test_evaluate_resume_writes_one_row_per_start(fmt):
    toy = toys()[2]
    record, _ = compile_toy(toy)
    answer = action_string(record.final["tool"], record.final["args"])

    class Answer:  # submits the known answer at once; function tasks check only the answer
        def generate(self, prompt_ids, seed):
            return ScriptedPolicy(fmt, [answer]).generate(prompt_ids, seed)

    rows = []
    evaluate_resume([record], {toy.task_id: toy}, ToyEnv, Answer(), fmt, decoding="greedy", renderer="quotient",
                    truncation=None, run_seed=5, on_episode=rows.append, start_seed=1)  # fmt: skip
    ss = starts(record, seed=1)
    assert [(r["condition"], r["depth"], tuple(r["prefix_order"])) for r in rows] == [
        (s.condition, s.depth, s.order) for s in ss
    ]
    assert all(r["success"] and r["prefix"] == r["depth"] and r["decoding"] == "greedy" for r in rows)


def test_pass_k():
    assert pass_k(8, 8, 4) == 1.0 and pass_k(3, 8, 4) == 0.0 and pass_k(4, 8, 4) == pytest.approx(1 / 70)
    assert pass_k(5, 8, 4) == pytest.approx(5 / 70) and pass_k(1, 2, 4) is None


def row(task, decoding, success, end="finish", condition="standard", depth=0, calls=2, failures=(), errors=()):
    steps = [{"error": x} for x in errors]
    failures = [{"reason": r} for r in failures]
    return {"task_id": task, "decoding": decoding, "success": success, "end": end, "condition": condition,
            "depth": depth, "calls": calls, "steps": steps, "env_failures": failures}  # fmt: skip


def test_summarize():
    rows = [row("a", "greedy", True), row("b", "greedy", False, "invalid_action", calls=0)]
    rows += [row("a", "s", i < 6) for i in range(8)] + [row("b", "s", i < 1, calls=1) for i in range(8)]
    rows += [row("a", "greedy", True, condition="on_path", depth=1), row("a", "greedy", False, condition="off_path",
             depth=1), row("a", "greedy", True, condition="empty")]  # fmt: skip
    m = summarize(rows)
    g, s = m["standard"]["greedy"], m["standard"]["s"]
    assert (g["tasks"], g["success"], g["pass^4"], g["ends"]) == (2, 0.5, None, {"finish": 1, "invalid_action": 1})
    assert g["outcomes"] == {"invalid_action": 1, "success": 1} and s["outcomes"] == {"success": 7, "wrong_answer": 9}
    errs = summarize([row("a", "g", False, calls=4, failures=["Variable value not yet known"] * 2),
                      row("b", "g", True, calls=4, errors=["observation error"])])["standard"]["g"]  # fmt: skip
    assert errs["calls"] == 8 and errs["errors_per_call"] == {"Variable value not yet known": 0.25,
                                                               "observation error": 0.125}  # fmt: skip
    assert errs["episodes_with_error"] == {"Variable value not yet known": 0.5, "observation error": 0.5}
    assert s["success"] == pytest.approx((6 / 8 + 1 / 8) / 2) and s["rollouts_per_task"] == 8
    assert s["pass^4"] == pytest.approx((15 / 70 + 0) / 2)  # C(6,4) / C(8,4) = 15/70
    r = m["resume_prefix"]
    assert r["by_condition"]["on_path"]["success"] == 1.0 and r["by_condition"]["off_path"]["success"] == 0.0
    assert r["off_path_gap"] == 1.0 and r["by_condition_and_depth"]["on_path@1"]["episodes"] == 1
    with pytest.raises(ValueError, match="different numbers of rollouts"):
        summarize([row("a", "s", True), row("a", "s", True), row("b", "s", True)])


class Oracle:
    """A stateless policy: the demonstration's next action at every state on the demo path, invalid text elsewhere.
    The batch flavour answers several prompts at once with the same function (D-026b's batched evaluation)."""

    def __init__(self, fmt, tasks):
        from qet.core.render import render_messages
        from qet.infer.decoding import Generation

        self.fmt, self.Generation, self.table = fmt, Generation, {}
        for toy in tasks:
            record, _ = compile_toy(toy)
            steps, x = record.steps(), toy.public()
            for k in range(record.n + 1):
                ids = fmt.prompt_ids(render_messages(x, steps[:k], toy.contracts, "quotient"))
                node = record.nodes[k] if k < record.n else record.final
                tool, args = (node.tool, node.args) if k < record.n else (node["tool"], node["args"])
                self.table[tuple(ids)] = action_string(tool, args)

    def generate(self, prompt_ids, seed):
        text = self.table.get(tuple(prompt_ids), "not an action")
        return self.Generation(self.fmt.encode(text) + [self.fmt.eot_id], True, 0.0)


class BatchOracle(Oracle):
    def generate_batch(self, prompts, seeds):
        return [self.generate(p, s) for p, s in zip(prompts, seeds, strict=True)]


def test_batched_evaluation_writes_the_same_rows(fmt):
    from qet.config import EvalSettings
    from qet.eval.runner import evaluate_tasks

    tasks = make_tasks("functions", range(5), n_core=4, n_given=2) + make_tasks("stateful", range(3))
    settings = EvalSettings(({"mode": "greedy"}, {"mode": "sample", "samples": 3}), max_new_tokens=64)
    rows = {}
    for kind, cls in (("one", Oracle), ("batch", BatchOracle)):
        policy, out = cls(fmt, tasks), []
        evaluate_tasks(tasks, ToyEnv, lambda d, p=policy: p, fmt, settings, renderer="quotient", truncation=None,
                       run_seed=0, on_episode=out.append, width=4)  # fmt: skip
        records = [compile_toy(t)[0] for t in tasks]
        evaluate_resume(records, {t.task_id: t for t in tasks}, ToyEnv, policy, fmt, decoding="greedy",
                        renderer="quotient", truncation=None, run_seed=0, on_episode=out.append, width=3)  # fmt: skip
        rows[kind] = out
    assert rows["batch"] == rows["one"] and len(rows["one"]) > 4 * len(tasks)
    assert {r["end"] for r in rows["one"]} >= {"finish", "invalid_action"}  # off-path starts leave the demo path

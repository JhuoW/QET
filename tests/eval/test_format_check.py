"""D-010: the format-error rate at sampled states along each task's demonstration."""

from qet.config import load_config
from qet.core.canonical import action_string
from qet.datasets import build_dataset
from qet.eval.format_check import format_errors, reason_kind
from qet.infer.agent import ScriptedPolicy
from qet.infer.decoding import Generation
from qet.manifest import REPO
from qet.train.chat import ChatFormat

SMOKE = REPO / "configs" / "smoke" / "qet.yaml"


def setup(n):
    cfg = load_config(SMOKE)
    ds = build_dataset(cfg.data, limit=n)
    return cfg, ds, ChatFormat.load(cfg.backbone)


def test_invalid_actions_are_counted_by_kind():
    cfg, ds, fmt = setup(3)
    good = [action_string(r.nodes[0].tool, r.nodes[0].args) for r in ds.test_records]
    outputs = [good[0], "hello", good[1], '{"tool":"nope","args":{}}', good[2], good[2]]  # 3 tasks x 2 states
    res = format_errors(ds.test_tasks, ds.test_records, ScriptedPolicy(fmt, outputs), fmt, ds.contracts,
                        renderer="quotient", states_per_task=2)  # fmt: skip
    assert res["states"] == 6 and res["errors"] == 2 and abs(res["rate"] - 2 / 6) < 1e-12
    assert res["kinds"] == {"not one JSON object": 1, "unknown tool": 1}


class Rambling:
    """Never emits the end of turn: every output is cut at the token limit."""

    def __init__(self, fmt):
        self.fmt, self.prompts = fmt, []

    def generate(self, prompt_ids, seed):
        self.prompts.append(tuple(prompt_ids))
        return Generation(self.fmt.encode('{"tool":'), False, 0.0)


def test_cut_outputs_are_errors_and_states_repeat_across_checks():
    cfg, ds, fmt = setup(4)
    a, b = Rambling(fmt), Rambling(fmt)
    res = format_errors(ds.test_tasks, ds.test_records, a, fmt, ds.contracts, renderer="quotient", states_per_task=3)
    format_errors(ds.test_tasks, ds.test_records, b, fmt, ds.contracts, renderer="quotient", states_per_task=3)
    assert res == {"states": 12, "errors": 12, "rate": 1.0, "kinds": {"no end of turn within the token limit": 12}}
    assert a.prompts == b.prompts  # the same states at every check
    assert len(set(a.prompts[::3])) == 4  # each task's first state is its empty state


def test_reason_kinds_drop_details():
    assert reason_kind("not one JSON object: Expecting value: line 1 column 1 (char 0)") == "not one JSON object"
    assert reason_kind("unknown tool 'x'") == "unknown tool"
    assert reason_kind("args is not an object") == "args is not an object"

"""ToolHop adapter (EVAL_PLAN §3.2, D-035): the official tool execution and answer check, in a sandbox."""

import json
import os

import pytest

from qet.envs.toolhop import DATA, Sandbox, ToolHopEnv, ToolHopTask, literal, load_tasks, official_correct


def test_tool_calls_follow_the_official_execution():
    box = Sandbox()
    f = 'def add(a, b=2):\n    print("noise")\n    return {"sum": a + b}\nprint(add(1))\n'
    assert box.call([f], "add", {"a": 3}) == {"content": json.dumps({"sum": 5})}  # printing stays off the channel
    uses_json = "def dump(x):\n    return json.dumps(x)\n"  # the evaluator's globals (json, os, datetime) are there
    assert box.call([uses_json], "dump", {"x": [1]}) == {"content": json.dumps("[1]")}
    own_import = "import math\ndef root(x):\n    return math.sqrt(x)\n"  # but not the source's module-level imports
    assert "name 'math' is not defined" in box.call([own_import], "root", {"x": 4})["error"]
    assert "boom" in box.call(['def bad():\n    raise ValueError("boom")\n'], "bad", {})["error"]
    assert "not JSON serializable" in box.call(["def s():\n    return {1, 2}\n"], "s", {})["error"]


def test_a_slow_tool_times_out_and_the_sandbox_recovers():
    box = Sandbox()
    slow = "def wait():\n    import time\n    time.sleep(30)\n"
    assert "timed out" in box.call([slow], "wait", {}, timeout=1.0)["error"]
    assert box.call(["def one():\n    return 1\n"], "one", {}) == {"content": "1"}


def test_tool_writes_land_in_the_sandbox_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    box = Sandbox()
    w = "def w():\n    with open('error_log.txt', 'a') as f:\n        f.write('x')\n    return 'ok'\n"
    assert box.call([w], "w", {}) == {"content": json.dumps("ok")}
    assert not (tmp_path / "error_log.txt").exists() and os.path.exists(os.path.join(box.workdir, "error_log.txt"))


def test_the_official_answer_check():
    assert literal("4") == 4 and literal("-2.5") == -2.5 and literal("[1, 'a']") == [1, "a"]
    with pytest.raises(ValueError):
        literal("__import__('os')")
    assert official_correct("4", "4", None) and official_correct(4, "4", None)  # equal after eval
    assert not official_correct("5", "4", None)
    assert official_correct("Mr Thomas Mawson Jr", "Thomas Mawson", None)  # names: substring, case-insensitive
    assert official_correct("1985-03-02", "1985-03-02", None)  # a leading zero is no literal: substring rule
    assert not official_correct("1,234", "1234", None)  # eval-equality path: 1234 vs the tuple (1, 234)
    assert official_correct("1,234 people", "1234 people", None)  # substring path: commas removed from the answer
    assert official_correct("no idea", "4", json.dumps({"count": 4}))  # the last tool output holds the answer
    assert not official_correct("", "abc", None)


def task_with(functions, tools, answer="4"):
    return ToolHopTask("toolhop-x", "q?", answer, tuple(tools) + ({"name": "finish", "params": ["answer"],
                       "description": "Submit the final answer"},), tuple(functions), 2)  # fmt: skip


def test_the_environment_records_tool_output_for_the_check():
    t = task_with(["def count(word):\n    return len(word) - 2\n"],
                  [{"name": "count", "params": ["word"], "description": "letters"}])  # fmt: skip
    env = ToolHopEnv(t)
    assert env.step_budget() == 8 and env.contracts() == {"count": {"reads": [], "writes": []}}
    assert env.step("count", {"word": "Thomas"}) == 4 and env.check("wrong")  # the last tool output says 4
    assert env.step("count", {}) == {"error": env.last_tool_content} and env.failures


@pytest.mark.skipif(not DATA.exists(), reason="the ToolHop data is fetched, not committed (make fetch)")
def test_the_pinned_data_loads():
    tasks = load_tasks()
    assert len(tasks) == 995 and all(t.tools[-1]["name"] == "finish" for t in tasks)
    assert sum(t.chain for t in tasks) == 3912 and "Question: " in tasks[0].public()["request"]
    assert "<answer>" not in tasks[0].public()["request"]


def test_toolhop_configs_are_evaluation_only(tmp_path):
    import yaml

    from qet.config import load_config

    data = {"milestone": "M1", "name": "toolhop", "configuration": "2", "seed": 0,
            "backbone": "Qwen/Qwen2.5-0.5B-Instruct",
            "data": {"env": "toolhop", "kind": "mandatory", "train": {"start": 0, "count": 0},
                     "test": {"start": 0, "count": 2}},
            "eval": {"decoding": [{"mode": "greedy"}], "max_new_tokens": 8}}  # fmt: skip
    path = tmp_path / "t.yaml"
    path.write_text(yaml.safe_dump(data))
    assert load_config(path).data.env == "toolhop"
    data["data"]["train"]["count"] = 5
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="evaluation only"):
        load_config(path)


@pytest.mark.slow  # a tiny-model evaluation of two ToolHop queries, end to end
@pytest.mark.skipif(not DATA.exists(), reason="the ToolHop data is fetched, not committed (make fetch)")
def test_a_toolhop_evaluation_runs(tmp_path):
    import sys

    import yaml

    sys.path.insert(0, str(DATA.parents[3] / "tests" / "eval"))
    from test_run import fresh_runlog, tiny_model

    from qet import run as runner
    from qet.config import load_config

    data = {"milestone": "M1", "name": "toolhop", "configuration": "2", "seed": 0,
            "backbone": "Qwen/Qwen2.5-0.5B-Instruct",
            "data": {"env": "toolhop", "kind": "mandatory", "train": {"start": 0, "count": 0},
                     "test": {"start": 0, "count": 2}},
            "eval": {"decoding": [{"mode": "greedy"}], "max_new_tokens": 8}}  # fmt: skip
    path = tmp_path / "t.yaml"
    path.write_text(yaml.safe_dump(data))
    out = runner.run(load_config(path), allow_dirty=True, results_root=tmp_path / "raw",
                     checkpoints_root=tmp_path / "ckpt", runlog_path=fresh_runlog(tmp_path),
                     model_factory=tiny_model)  # fmt: skip
    manifest = json.loads((out / "manifest.json").read_text())
    rows = [json.loads(x) for x in (out / "predictions" / "greedy.jsonl").read_text().splitlines()]
    assert manifest["status"] == "completed" and manifest["benchmark"]["name"] == "toolhop" and len(rows) == 2

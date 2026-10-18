"""ToolHop adapter, evaluation only (EVAL_PLAN §3.2, D-035).

The 995 queries of bytedance-research/ToolHop (Apache-2.0; data/toolhop/data/ToolHop.json at dataset commit b439d72,
sha256 pinned below), in the mandatory tool-use setting of the official evaluator (code/evaluation_open.py):
- the same instructions in the request, minus the "<answer>" line (the answer goes into our finish action);
- at most 9 turns: 8 tool calls and the answer (one call per turn here, where the official loop allows several);
- each call runs the dataset's code as the official `excute_func` does: `exec` of the matching source (the first
  `def` name) with the evaluator's module globals, in a separate locals scope, then the call, with a 10 s limit;
  an error becomes the official text "an error occured when call <name>: <error>";
- success follows the official check on the finish answer and on the last tool output.
Tool code runs in a sandbox process (its own working directory, a memory limit, output kept off its reply channel),
because two tools exec or eval strings taken from the model's arguments. The official check `eval`s the gold answer
and the model's answer; here a literal-and-arithmetic evaluator stands in for `eval`, so model text is never
executed (answers that only `eval` itself could read, such as names, fall back to the substring rule as they would).
"""

from __future__ import annotations

import ast
import hashlib
import json
import operator
import os
import select
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qet.core.canonical import JSON, canon

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "data" / "toolhop" / "data" / "ToolHop.json"
# data/toolhop/data/ToolHop.json at DATASET_COMMIT (D-035)
DATA_SHA256 = "0a51f71a44b7025645e452123af3caf2e348301922af91778e268db0188a7fab"
DATASET_COMMIT = "b439d7279af359fda46e8117ae4f0245b75f5c6b"
TIMEOUT_S = 10.0  # the official func_set_timeout(10)
MAX_TURNS = 9  # the official --max_turns default
REQUEST = (  # the official mandatory-scenario user message, verbatim but for the <answer> line
    "You will be asked a question with some tools, and should provide a short final answer.\n"
    "Please note that you must call the tool at every step, you must not use your own knowledge. Your final answer "
    "must also be returned from the tool.\n"
    "If the final answer is a date, format is as follows: YYYY-MM-DD (ISO standard)\n"
    "If the final nswer is a name, format it as follows: Firstname Lastname\n"
    "If the final answer contains any number, format it as a number, not a word, and only output that number. Do not "
    "include leading 0s.\n\n"
    "Answer as short as possible.\n"
    "Question: {question}"
)
FINISH = {"name": "finish", "params": ["answer"], "description": "Submit the final answer"}

# The worker: the official excute_func (evaluation_open.py lines 74-78) in a loop, reading one JSON request per line.
WORKER = r"""
import argparse, builtins, contextlib, datetime, io, json, os, resource, sys
from hashlib import sha256
resource.setrlimit(resource.RLIMIT_AS, (8 << 30, 8 << 30))
EVALUATOR_GLOBALS = {"__builtins__": builtins, "__name__": "__main__", "argparse": argparse, "json": json, "os": os,
                     "sha256": sha256, "datetime": datetime}
reply = sys.stdout
for line in sys.stdin:
    req = json.loads(line)
    sink = io.StringIO()
    try:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            scope_globals, scope_locals = dict(EVALUATOR_GLOBALS), {}
            for func in req["functions"]:
                if func.split("def")[1].split("(")[0].strip() == req["name"]:
                    exec(func, scope_globals, scope_locals)
            feedback = eval(req["name"], scope_globals, scope_locals)(**req["args"])
            out = {"content": json.dumps(feedback, ensure_ascii=False)}
    except Exception as e:  # what the official get_feedback catches
        out = {"error": str(e)}
    reply.write(json.dumps(out) + "\n")
    reply.flush()
"""


class Sandbox:
    """One worker process for the tool calls of this process; restarted after a timeout or a crash."""

    def __init__(self) -> None:
        self.proc: subprocess.Popen | None = None
        self.workdir = tempfile.mkdtemp(prefix="toolhop-")  # error_log.txt and other writes land here

    def _start(self) -> None:
        env = {"PATH": os.environ.get("PATH", ""), "HOME": self.workdir, "PYTHONDONTWRITEBYTECODE": "1"}
        self.proc = subprocess.Popen([sys.executable, "-I", "-c", WORKER], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, cwd=self.workdir, env=env,
                                     text=True, bufsize=1)  # fmt: skip

    def call(self, functions: Sequence[str], name: str, args: Mapping[str, Any], timeout: float = TIMEOUT_S) -> dict:
        if self.proc is None or self.proc.poll() is not None:
            self._start()
        assert self.proc is not None and self.proc.stdin is not None and self.proc.stdout is not None
        try:
            self.proc.stdin.write(json.dumps({"functions": list(functions), "name": name, "args": dict(args)}) + "\n")
            self.proc.stdin.flush()
            ready, _, _ = select.select([self.proc.stdout], [], [], timeout)
            line = self.proc.stdout.readline() if ready else ""
        except (BrokenPipeError, OSError):
            line = ""
        if not line:  # a timeout or a dead worker: the official FunctionTimedOut, then a fresh worker
            self.proc.kill()
            self.proc.wait()
            self.proc = None
            return {"error": f"Function excute_func timed out after {timeout:.6f} seconds."}
        return json.loads(line)


_SANDBOX: Sandbox | None = None


def sandbox() -> Sandbox:
    global _SANDBOX
    if _SANDBOX is None:
        _SANDBOX = Sandbox()
    return _SANDBOX


def describe(tool: Mapping[str, Any]) -> tuple[list[str], str]:
    """Parameter names in schema order and a description with the parameters folded in (our renderer shows names and
    one description per tool, D-035)."""
    params = (tool.get("parameters") or {}).get("properties") or {}
    required = set((tool.get("parameters") or {}).get("required") or [])
    parts = []
    for p, spec in params.items():
        kind = spec.get("type")
        kind = "/".join(kind) if isinstance(kind, list) else kind
        if kind == "array" and isinstance(spec.get("items"), dict) and spec["items"].get("type"):
            kind = f"array of {spec['items']['type']}"
        bits = [b for b in (kind, "required" if p in required else None) if b]
        text = f"{p} ({', '.join(bits)}): {spec.get('description', '').strip()}"
        items = spec.get("items") if isinstance(spec.get("items"), dict) else {}
        enum = spec.get("enum") or items.get("enum")
        if enum:
            text += f" One of {json.dumps(enum, ensure_ascii=False)}."
        if "default" in spec:
            text += f" Default {json.dumps(spec['default'], ensure_ascii=False)}."
        parts.append(text)
    desc = tool.get("description", "").strip()
    return list(params), desc + (" Parameters: " + " ".join(parts) if parts else "")


@dataclass(frozen=True)
class ToolHopTask:
    task_id: str
    question: str
    answer: str
    tools: tuple[dict[str, Any], ...]  # name, params, description; finish included
    functions: tuple[str, ...]  # the dataset's tool code
    chain: int  # the number of sub-questions

    def public(self) -> dict[str, Any]:
        return {"request": REQUEST.format(question=self.question), "tools": [dict(t) for t in self.tools],
                "initial_observation": None, "finish": "finish"}  # fmt: skip


def load_tasks(path: Path = DATA) -> list[ToolHopTask]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != DATA_SHA256:
        raise ValueError(f"{path} is not the pinned ToolHop data (D-035)")
    tasks = []
    for q in json.loads(raw):
        tools = []
        for t in q["tools"].values():
            params, desc = describe(t)
            tools.append({"name": t["name"], "params": params, "description": desc})
        tasks.append(ToolHopTask(f"toolhop-{q['id']}", q["question"], str(q["answer"]),
                                 tuple(tools) + (FINISH,), tuple(q["functions"]), len(q["sub_task"])))  # fmt: skip
    return tasks


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}  # fmt: skip


def literal(text: str) -> Any:
    """What `eval` returns for a literal or arithmetic expression (raises otherwise, as `eval` would for most text)."""

    def ev(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            items = [ev(e) for e in node.elts]
            return {ast.List: list, ast.Tuple: tuple, ast.Set: set}[type(node)](items)
        if isinstance(node, ast.Dict):
            return {ev(k): ev(v) for k, v in zip(node.keys, node.values, strict=True)}
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            v = ev(node.operand)
            return -v if isinstance(node.op, ast.USub) else +v
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 1000:
                raise ValueError("exponent too large")
            return _OPS[type(node.op)](left, right)
        raise ValueError(f"not a literal: {type(node).__name__}")

    return ev(ast.parse(text.strip(), mode="eval"))


def official_correct(answer: JSON, gold: str, last_tool_content: str | None) -> bool:
    """evaluation_open.py lines 176-199, with `literal` for `eval`."""
    correct = False
    solution = answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False)
    if solution:  # the official checks the final assistant message only when it has content
        try:
            truth = literal(gold)
        except Exception:
            if str(gold).removesuffix(".0").lower() in str(solution).removesuffix(".0").replace(",", "").lower():
                correct = True
        else:
            try:
                value = literal(solution)
            except Exception:
                pass
            else:
                correct = correct or truth == value
    if (
        last_tool_content is not None
        and gold.removesuffix(".0").lower() in last_tool_content.removesuffix(".0").replace(",", "").lower()
    ):
        correct = True
    return correct


class ToolHopEnv:
    """One ToolHop query (evaluation only: no compile, no snapshots)."""

    contract_source = "pure functions (ToolHop's tools are deterministic code without side effects on the episode)"

    def __init__(self, task: ToolHopTask):
        self.toolhop, self.task_id = task, task.task_id
        self.last_tool_content: str | None = None
        self.failures: list[dict[str, Any]] = []

    def task(self) -> dict[str, Any]:
        return self.toolhop.public()

    def contracts(self) -> dict[str, Any]:
        return {t["name"]: {"reads": [], "writes": []} for t in self.toolhop.tools if t["name"] != "finish"}

    def step_budget(self) -> int:
        return MAX_TURNS - 1  # 8 tool calls, then the answer

    def step(self, tool: str, args: Mapping[str, JSON]) -> JSON:
        res = sandbox().call(self.toolhop.functions, tool, args)
        if "error" in res:
            text = f"an error occured when call {tool}: {res['error']}"  # the official wording
            self.last_tool_content = text
            self.failures.append({"tool": tool, "reason": res["error"][:200]})
            return {"error": text}
        self.last_tool_content = res["content"]
        try:
            value = json.loads(res["content"])
            canon(value)  # the prompt shows canonical JSON; NaN or infinity cannot be shown
            return value
        except (ValueError, TypeError):
            return res["content"]

    def snapshot(self) -> Any:
        raise NotImplementedError("ToolHop is evaluation only")

    def restore(self, snap: Any) -> None:
        raise NotImplementedError("ToolHop is evaluation only")

    def check(self, answer: JSON) -> bool:
        return official_correct(answer, self.toolhop.answer, self.last_tool_content)

    def final_state(self) -> JSON:
        return None

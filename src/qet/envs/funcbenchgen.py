"""FuncBenchGen adapter (EVAL_PLAN §3.1, M1.1): the pinned repository's graph generator and evaluator, wrapped and
never modified (third_party/FuncBenchGen).

A task is one graph of one grid configuration and one trial, built exactly as the repository's `run` builds it: the
hidden function DAG for the graph seed, dependencies re-expressed through shared types (`use_types`, the repository's
default), and variable values drawn for the trial seed. The request is the repository's input prompt; the tools are
its function schemas plus finish(answer). `step` runs the repository's `_execute_tool` with on_wrong_inputs="Execute":
an input whose type is not yet known, or a wrong value, returns a random value from the repository's global RNG, which
the adapter seeds per episode and keeps apart from every other user of the global RNG. Its text result "Variable x =
123." becomes the observation {"x": 123}; "Error occurred." becomes {"error": "Error occurred."}; the repository's
failure reason (value not yet known, incorrect value, ...) is kept for the error taxonomy, out of the model's sight.
Success follows the repository: the finish answer equals the target value and at least the minimum number of calls
was made; calls are capped at twice the minimum. Functions are pure (SPEC §3: contracts by construction).
"""

from __future__ import annotations

import copy
import json
import random
import re
import sys
import types
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from qet.core.canonical import JSON, canon

REPO_DIR = Path(__file__).resolve().parents[3] / "third_party" / "FuncBenchGen"
_RESULT = re.compile(r"Variable (\S+) = (-?\d+)\.\Z")


def _repo():
    """The repository's modules; `funcbenchgen.py` imports LiteLLM for API calls this adapter never makes, so a stub
    stands in for it."""
    src = str(REPO_DIR / "src")
    if not (REPO_DIR / "src" / "evaluator.py").exists():
        raise RuntimeError(f"{REPO_DIR} is missing: run `git submodule update --init third_party/FuncBenchGen`")
    if src not in sys.path:
        sys.path.insert(0, src)
    for name in ("evaluator", "funcbenchgen", "function_tree", "utils"):  # the repository imports these by bare name
        loaded = sys.modules.get(name)
        if loaded is not None and not str(getattr(loaded, "__file__", "")).startswith(src):
            raise RuntimeError(f"another module named {name!r} is already imported ({loaded!r})")
    sys.modules.setdefault("litellm", types.SimpleNamespace(completion=None))
    import evaluator
    import funcbenchgen
    import function_tree
    import utils

    return funcbenchgen, function_tree, evaluator, utils


@contextmanager
def _global_rng(state: dict[str, Any]):
    """Run repository code on its own Python and NumPy global RNG states, leaving everyone else's untouched."""
    saved = (random.getstate(), np.random.get_state())
    random.setstate(state["py"])
    np.random.set_state(state["np"])
    try:
        yield
    finally:
        state["py"], state["np"] = random.getstate(), np.random.get_state()
        random.setstate(saved[0])
        np.random.set_state(saved[1])


def _seeded_state(py_seed: int, np_seed: int) -> dict[str, Any]:
    saved = (random.getstate(), np.random.get_state())
    random.seed(py_seed)
    np.random.seed(np_seed)
    state = {"py": random.getstate(), "np": np.random.get_state()}
    random.setstate(saved[0])
    np.random.set_state(saved[1])
    return state


@dataclass(frozen=True)
class GridConfig:
    """One cell of the grid (EVAL_PLAN §3.1): core functions, depth, irrelevant functions."""

    core_nodes: int
    max_critical_path: int
    num_total_nodes: int  # core + connected + disconnected functions
    num_disconnected_nodes: int = 0
    use_types: bool = True
    show_variable_names_in_description: bool = False
    rename_variables: bool = False
    use_subtypes: bool = False
    num_supertypes: int = 5
    function_shuffle_seed: int = 42

    @property
    def min_calls(self) -> int:
        return max(self.core_nodes, self.max_critical_path + 1)

    def experiment_config(self):
        fbg, *_ = _repo()
        return fbg.ExperimentConfig(
            num_total_nodes=self.num_total_nodes,
            max_critical_path_length=self.max_critical_path,
            num_disconnected_nodes=self.num_disconnected_nodes,
            min_calls=self.min_calls,
            show_variable_names_in_description=self.show_variable_names_in_description,
            rename_variables=self.rename_variables,
            use_subtypes=self.use_subtypes,
            num_supertypes=self.num_supertypes,
        )

    def tag(self) -> str:
        fbg, *_ = _repo()
        return fbg._config_tag(self.experiment_config())


@dataclass(frozen=True)
class FuncTask:
    task_id: str
    config: GridConfig
    graph_seed: int
    trial: int
    request: str
    tools: tuple[dict[str, Any], ...]  # name, params (schema order), description; finish included
    calls: tuple[dict[str, Any], ...]  # the gold calls of the needed (core) functions, in a topological order
    true_preds: tuple[tuple[int, ...], ...]  # the generator's DAG over `calls` (audits only, never Compile's input)
    answer: int
    min_calls: int
    graph: dict[str, Any] = field(repr=False, compare=False)  # the repository's node-link payload
    _proto: Any = field(default=None, repr=False, compare=False)  # (evaluator, known types, var_to_type)

    def public(self) -> dict[str, Any]:
        return {"request": self.request, "tools": [dict(t) for t in self.tools], "initial_observation": None,
                "finish": "finish"}  # fmt: skip


FINISH = {"name": "finish", "params": ["answer"], "description": "Submit the final answer"}


def build_task(cfg: GridConfig, graph_seed: int, trial: int = 0, task_id: str | None = None) -> FuncTask:
    """One graph and trial, built as the repository builds them, under seeds derived from (config, graph, trial)."""
    fbg, ft, ev, utils = _repo()
    tag = cfg.tag()
    state = _seeded_state(graph_seed, graph_seed + 7)
    with _global_rng(state):
        payload = fbg._build_graph_for_seed(cfg.experiment_config(), graph_seed)  # seeds the global RNG itself
        tree = ft.FunctionDependencyTree(seed=utils.stable_int_from(tag, str(graph_seed), "types"))
        tree.graph = utils.graph_from_node_link_data(copy.deepcopy(payload["graph_data"]))
        if cfg.use_types:
            tree.reformat_tree_with_shared_types(
                show_variable_names_in_description=cfg.show_variable_names_in_description,
                rename_variables=cfg.rename_variables, use_subtypes=cfg.use_subtypes,
                num_supertypes=cfg.num_supertypes,
            )  # fmt: skip
        evaluator = ev.ToolCallingEvaluator(
            tree, seed=utils.stable_int_from(tag, str(graph_seed), "evaluator"), num_noise_inputs=0,
            function_shuffle_seed=cfg.function_shuffle_seed, error_detail_level="None", on_wrong_inputs="Execute",
            repeat_known_variable_values=False,
        )  # fmt: skip
        trial_seed = fbg.derive_trial_seed(tag, graph_seed, trial)
        utils.set_global_rng(trial_seed, trial_seed + 11)
        input_vars, known, var_to_type, _ = fbg._initialize_trial_variables(evaluator, tree, cfg.use_types)
        target_node = payload["target_node"]
        target_var = tree.graph.nodes[target_node]["function"].output
        request = evaluator._get_input_prompt(target_var, input_vars)
        schemas = evaluator._get_function_schemas()
    g = tree.graph
    needed = sorted(set(nx.ancestors(g, target_node)) | {target_node})
    order = list(nx.lexicographical_topological_sort(g.subgraph(needed)))  # one topological order
    index = {name: i for i, name in enumerate(order)}
    values = evaluator.variable_values
    calls = tuple({"tool": n, "args": {x: values[x] for x in g.nodes[n]["function"].inputs}} for n in order)
    true_preds = tuple(tuple(sorted(index[p] for p in g.predecessors(n) if p in index)) for n in order)
    tools = [{"name": s["function"]["name"], "params": list(s["function"]["parameters"]["required"]),
              "description": s["function"]["description"]} for s in schemas]  # fmt: skip
    task = FuncTask(
        task_id=task_id or f"fbg-{tag}-g{graph_seed}-t{trial}",
        config=cfg,
        graph_seed=graph_seed,
        trial=trial,
        request=request,
        tools=tuple(sorted(tools + [FINISH], key=lambda t: t["name"])),
        calls=calls,
        true_preds=true_preds,
        answer=int(values[target_var]),
        min_calls=cfg.min_calls,
        graph=payload,
    )
    object.__setattr__(task, "_proto", (evaluator, frozenset(known), dict(var_to_type)))
    return task


class _Call:
    """The shape of a tool call the repository's evaluator reads: .function.name and .function.arguments (JSON)."""

    def __init__(self, name: str, args: Mapping[str, JSON]):
        self.function = types.SimpleNamespace(name=name, arguments=json.dumps(dict(args)))


class FuncBenchGenEnv:
    contract_source = "by construction (FuncBenchGen functions are pure)"

    def __init__(self, task: FuncTask):
        self.toy = task  # the task object (same attribute name as the toy env, for the evaluation code)
        self.task_id = task.task_id
        evaluator, known, var_to_type = task._proto
        self._evaluator = copy.deepcopy(evaluator)
        self._known = set(known)
        self._var_to_type = var_to_type
        _, _, _, utils = _repo()
        seed = utils.stable_int_from(task.task_id, "episode")
        self._rng = _seeded_state(seed, seed + 7)  # the same wrong values for the same calls in every episode
        self.calls = 0
        self.failures: list[dict[str, Any]] = []

    def task(self) -> dict[str, Any]:
        return self.toy.public()

    def contracts(self) -> dict[str, Any]:
        return {t["name"]: {"reads": [], "writes": []} for t in self.toy.tools if t["name"] != "finish"}

    def step_budget(self) -> int:
        return 2 * self.toy.min_calls  # the repository's cap: twice the minimum number of calls

    def step(self, tool: str, args: Mapping[str, JSON]) -> JSON:
        self.calls += 1
        try:
            canon(args)
        except (TypeError, ValueError) as e:
            self.failures.append({"call": self.calls, "tool": tool, "reason": "Tool input was invalid"})
            return {"error": f"arguments are not canonical JSON: {e}"}
        with _global_rng(self._rng):
            try:
                result, reason = self._evaluator._execute_tool(_Call(tool, args), self._known,
                                                                use_types=self.toy.config.use_types,
                                                                var_to_type=self._var_to_type)  # fmt: skip
            except Exception:  # the repository's own loop treats any exception as invalid input
                result, reason = "Error occurred.", "Tool input was invalid"
        if reason:
            self.failures.append({"call": self.calls, "tool": tool, "reason": reason})
        m = _RESULT.match(result)
        return {m.group(1): int(m.group(2))} if m else {"error": result}

    def snapshot(self) -> Any:
        return copy.deepcopy((self._evaluator.revealed_vars, self._known, self._rng, self.calls, self.failures))

    def restore(self, snap: Any) -> None:
        revealed, self._known, self._rng, self.calls, self.failures = copy.deepcopy(snap)
        self._evaluator.revealed_vars = revealed

    def check(self, answer: JSON) -> bool:
        try:
            answer = canon(answer)
        except (TypeError, ValueError):
            return False
        is_int = isinstance(answer, int) and not isinstance(answer, bool)
        return is_int and answer == self.toy.answer and self.calls >= self.toy.min_calls

    def final_state(self) -> JSON:
        return {}  # pure functions: no state to compare beyond the observations

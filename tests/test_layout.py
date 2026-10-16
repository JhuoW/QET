"""The repository layout of AGENTS.md exists and every module imports."""

import importlib

import pytest

MODULES = [
    "qet",
    "qet.run",
    "qet.manifest",
    "qet.runlog",
    *(f"qet.core.{m}" for m in ("canonical", "contracts", "compile", "reduce", "render", "lattice", "weights", "api")),
    *(f"qet.train.{m}" for m in ("chat", "scoring", "methods", "trainer")),
    *(f"qet.infer.{m}" for m in ("agent", "decoding")),
    *(
        f"qet.envs.{m}"
        for m in ("toy", "funcbenchgen", "toolhop", "appworld", "stabletoolbench", "bfcl", "tinyagent", "nestful")
    ),
    *(f"qet.eval.{m}" for m in ("runner", "resume_prefix", "metrics", "stats")),
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(name)

"""M1.4 groundwork: runs on FuncBenchGen read the records of an M1.3 compile run (checked against its hashes) and
evaluate tasks rebuilt from the manifest it compiled."""

import json
import sys
from pathlib import Path

import pytest
import yaml
from test_run import fresh_runlog, tiny_model

from qet import run as runner
from qet.config import load_config
from qet.datasets import build_dataset
from qet.manifest import FIELDS

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def compile_run(tmp_path_factory):
    sys.path.insert(0, str(REPO / "scripts"))
    import compile_m1

    tmp = tmp_path_factory.mktemp("compile")
    log = fresh_runlog(tmp)
    return compile_m1.main(["--splits", "train", "dev", "--limit", "3", "--allow-dirty", "--out", str(tmp / "raw"),
                            "--runlog", str(log)])  # fmt: skip


def fbg_config(tmp_path, compile_run, **changes):
    data = yaml.safe_load((REPO / "configs" / "smoke" / "qet.yaml").read_text())
    data["data"] = {"env": "funcbenchgen", "kind": "grid", "train": {"start": 0, "count": 3},
                    "test": {"start": 0, "count": 3},
                    "source": {"compile_run": str(compile_run), "train": "train", "eval": "dev"}}  # fmt: skip
    data["train"].update(token_budget=40_000, warmup_steps=1, batch_graphs=2, checkpoint_fractions=[1.0])
    data["eval"] = {"decoding": [{"mode": "greedy"}], "max_new_tokens": 6, "resume_prefix_tasks": 1}
    data.update(changes)
    path = tmp_path / "fbg.yaml"
    path.write_text(yaml.safe_dump(data))
    return load_config(path)


def test_the_dataset_comes_from_the_compile_run(tmp_path, compile_run):
    cfg = fbg_config(tmp_path, compile_run)
    ds = build_dataset(cfg.data)
    assert len(ds.train_records) == 3 and len(ds.test_tasks) == 3
    assert [t.task_id for t in ds.test_tasks] == [r.task_id for r in ds.test_records]
    assert not {r.task_id for r in ds.train_records} & {t.task_id for t in ds.test_tasks}
    assert all(c == {"reads": [], "writes": []} for c in ds.contracts.values())
    assert ds.benchmark["name"] == "funcbenchgen" and len(ds.benchmark["commit"]) == 40


def test_a_tampered_compile_run_is_refused(tmp_path, compile_run):
    import shutil

    copy = tmp_path / "copy"
    shutil.copytree(compile_run, copy)
    records = copy / "train" / "records.jsonl"
    records.write_text(records.read_text().replace('"answer"', '"answer" ', 1))
    with pytest.raises(ValueError, match="does not match its compile manifest"):
        build_dataset(fbg_config(tmp_path, copy).data)


def test_configs_refuse_evaluating_on_the_training_split(tmp_path, compile_run):
    data = yaml.safe_load((REPO / "configs" / "smoke" / "qet.yaml").read_text())
    data["data"] = {"env": "funcbenchgen", "kind": "grid", "train": {"start": 0, "count": 3},
                    "test": {"start": 0, "count": 3},
                    "source": {"compile_run": str(compile_run), "train": "train", "eval": "train"}}  # fmt: skip
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="training split"):
        load_config(path)


@pytest.mark.slow  # a tiny-model run end to end on FuncBenchGen tasks
def test_a_run_on_funcbenchgen(tmp_path, compile_run):
    cfg = fbg_config(tmp_path, compile_run)
    out = runner.run(cfg, allow_dirty=True, results_root=tmp_path / "raw", checkpoints_root=tmp_path / "ckpt",
                     runlog_path=fresh_runlog(tmp_path), model_factory=tiny_model)  # fmt: skip
    manifest = json.loads((out / "manifest.json").read_text())
    assert all(f in manifest for f in FIELDS) and manifest["status"] == "completed"
    assert manifest["benchmark"]["name"] == "funcbenchgen"
    rows = [json.loads(x) for x in (out / "predictions" / "greedy.jsonl").read_text().splitlines()]
    assert len(rows) == 3 and all(r["end"] in ("finish", "invalid_action", "budget_exhausted") for r in rows)

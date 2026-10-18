"""E1.6: run configs, the dry run, manifests (every EVAL_PLAN §10 field) and the run log. Real runs here use a tiny
random model and temporary folders; the GPU runs are E1.7's."""

import dataclasses
import json
from pathlib import Path

import pytest
import torch
import yaml

from qet import run as runner
from qet import runlog
from qet.config import load_config
from qet.manifest import FIELDS, REPO, sha256_file

SMOKE = REPO / "configs" / "smoke" / "qet.yaml"


def tiny_model(cfg):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    sizes = {"hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4}
    config = Qwen2Config(vocab_size=151936, num_key_value_heads=2, attn_implementation="sdpa", **sizes)
    return Qwen2ForCausalLM(config).eval()


def small_config(tmp_path, **changes):
    data = yaml.safe_load(SMOKE.read_text())
    data["data"]["train"]["count"], data["data"]["test"]["count"] = 4, 3
    data["train"].update(token_budget=60_000, warmup_steps=1, batch_graphs=2, checkpoint_fractions=[0.5, 1.0])
    data["eval"] = {"decoding": [{"mode": "greedy"}, {"mode": "sample", "samples": 2}], "max_new_tokens": 6,
                    "resume_prefix_tasks": 2}  # fmt: skip
    data.update(changes)
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(data))
    return load_config(path)


def fresh_runlog(tmp_path):
    path = tmp_path / "RUNLOG.md"
    path.write_text(f"# Run log\n\n{runlog.HEADER}\n| {' | '.join('---' for _ in runlog.COLUMNS)} |\n")
    return path


def test_configs_are_validated(tmp_path):
    cfg = load_config(SMOKE)
    assert (cfg.configuration, cfg.method.name, cfg.train.k, cfg.data.train.count) == ("6", "qet", 4, 256)
    assert cfg.sha256 == sha256_file(SMOKE)
    good = yaml.safe_load(SMOKE.read_text())
    bad = [
        {**good, "extra": 1},
        {**good, "configuration": "7"},
        {**good, "backbone": "gpt2"},
        {**good, "data": {**good["data"], "test": {"start": 100, "count": 10}}},  # overlaps the training seeds
        {**good, "train": {**good["train"], "learning_rate": 1.0}},
        {**good, "eval": {"decoding": [{"mode": "beam"}]}},
        {**good, "eval": {"decoding": [{"mode": "greedy", "samples": 8}]}},  # caught at load, not mid-run
        {**good, "eval": {"decoding": [{"mode": "sample", "temperature": 0}]}},
        {**good, "eval": {"decoding": []}},
        {**good, "name": "a/b"},  # names a folder
    ]
    for i, data in enumerate(bad):
        path = tmp_path / f"bad{i}.yaml"
        path.write_text(yaml.safe_dump(data))
        with pytest.raises(ValueError):
            load_config(path)


def test_the_dry_run_prints_the_plan_and_writes_nothing(capsys, monkeypatch):
    """Every way of writing a file fails during the dry run, so any attempted write fails the test."""
    import builtins
    import os
    import pathlib

    # Library import side effects are not the dry run's writes: torch makes a temporary folder when transformers'
    # auto classes first import it, which used to fail this test when it ran before any other test imported them.
    from transformers import AutoTokenizer  # noqa: F401

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    real_open = builtins.open
    attempts = []

    def guarded_open(file, mode="r", *args, **kwargs):
        if any(c in mode for c in "wax+") and "__pycache__" not in str(file):
            attempts.append(str(file))
            raise PermissionError(f"the dry run tried to write {file}")
        return real_open(file, mode, *args, **kwargs)

    def refuse(*args, **kwargs):
        attempts.append(str(args[:1]))
        raise PermissionError("the dry run tried to write")

    monkeypatch.setattr(builtins, "open", guarded_open)
    for name in ("write_text", "write_bytes", "mkdir", "touch", "rename", "replace", "unlink"):
        monkeypatch.setattr(pathlib.Path, name, refuse)
    for name in ("makedirs", "mkdir", "replace", "rename", "remove", "unlink"):
        monkeypatch.setattr(os, name, refuse)
    runner.main(["--config", str(SMOKE), "--dry-run", "--limit", "6"])
    out = capsys.readouterr().out
    assert attempts == []
    plan = {line.split(": ", 1)[0]: line.split(": ", 1)[1] for line in out.splitlines() if ": " in line}
    train, est = json.loads(plan["train"]), json.loads(plan["estimate"])
    assert train["token_budget"] == 4_000_000 and train["tokens_per_epoch"]["scored"] > 0 and train["steps"] > 0
    assert est["gpu_hours"] > 0 and "estimated GPU-hours" in out


@pytest.mark.slow  # trains and evaluates a tiny model end to end: about 30 s here
def test_a_real_run_writes_the_manifest_the_outputs_and_the_run_log(tmp_path):
    cfg = small_config(tmp_path)
    log_path = fresh_runlog(tmp_path)
    out = runner.run(cfg, allow_dirty=True, results_root=tmp_path / "raw", checkpoints_root=tmp_path / "ckpt",
                     runlog_path=log_path, model_factory=tiny_model)  # fmt: skip
    manifest = json.loads((out / "manifest.json").read_text())
    assert all(f in manifest for f in FIELDS) and manifest["status"] == "completed"
    assert manifest["config"]["sha256"] == cfg.sha256 and manifest["seed"] == 0 and manifest["lambda"] == 0.5
    assert manifest["data_manifest"]["sha256"] == sha256_file(out / "data" / "data_manifest.json")
    assert manifest["tokenizer"]["revision"] == manifest["backbone"]["revision"]
    assert manifest["tokens"]["scored"] > 0 and manifest["tokens"]["gradient"] > 0 and manifest["gpu_hours"] == 0
    assert manifest["tokens"]["budget_used"] >= 60_000
    rows = runlog.read_rows(log_path)
    assert [(r["run_id"], r["status"], r["GPU-h"]) for r in rows] == [(out.name, "completed", "0.000")]
    preds = {p.name: [json.loads(x) for x in p.read_text().splitlines()] for p in (out / "predictions").iterdir()}
    assert sorted(preds) == ["greedy.jsonl", "resume_prefix.jsonl", "sample-t1-p1.jsonl"]
    assert len(preds["greedy.jsonl"]) == 3 and len(preds["sample-t1-p1.jsonl"]) == 6
    assert all(r["steps"] and r["steps"][0]["prompt_sha256"] for rs in preds.values() for r in rs)
    metrics = json.loads((out / "metrics.json").read_text())
    assert set(metrics["standard"]) == {"greedy", "sample-t1-p1"} and metrics["resume_prefix"] is not None
    phases = [json.loads(x)["phase"] for x in (out / "train_log.jsonl").read_text().splitlines()]
    assert phases[0] == "warmup" and phases.count("warmup") == 1 and "train" in phases
    ckpts = sorted(p.name for p in (tmp_path / "ckpt" / out.name).iterdir())
    assert ckpts == ["frac-0.5", "frac-1"]
    meta = json.loads((tmp_path / "ckpt" / out.name / "frac-1" / "qet_checkpoint.json").read_text())
    assert meta["files"] and all(len(h) == 64 for h in meta["files"].values())  # adapter weights hashed
    assert manifest["outputs"]["checkpoints"][str(tmp_path / "ckpt" / out.name / "frac-1")] == meta["files"]
    assert set(manifest["gpu_hours_by_phase"]) == {"load", "train", "eval"} and "git_end" in manifest
    starts = json.loads((out / "data" / "resume_starts.json").read_text())
    data_manifest = json.loads((out / "data" / "data_manifest.json").read_text())
    assert data_manifest["files"]["resume_starts.json"] == sha256_file(out / "data" / "resume_starts.json")
    assert [(s["task_id"], s["condition"], s["depth"], s["order"]) for s in starts] == [
        (r["task_id"], r["condition"], r["depth"], r["prefix_order"]) for r in preds["resume_prefix.jsonl"]
    ]  # the frozen starts are the ones evaluated
    assert all(o["outcomes"] for o in metrics["standard"].values())


def test_a_failed_run_is_recorded_as_failed(tmp_path):
    cfg = small_config(tmp_path)
    log_path = fresh_runlog(tmp_path)

    def broken(cfg):
        raise RuntimeError("no model")

    with pytest.raises(RuntimeError, match="no model"):
        runner.run(cfg, allow_dirty=True, results_root=tmp_path / "raw", checkpoints_root=tmp_path / "ckpt",
                   runlog_path=log_path, model_factory=broken)  # fmt: skip
    (out,) = (tmp_path / "raw").iterdir()
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "failed" and "no model" in manifest["error"] and manifest["gpu_hours"] == 0
    assert runlog.read_rows(log_path)[0]["status"] == "failed"


def test_a_terminated_run_is_recorded_as_failed(tmp_path):
    """The scheduler's SIGTERM becomes SystemExit (run._terminate), which must still mark the run failed."""
    cfg = small_config(tmp_path)
    log_path = fresh_runlog(tmp_path)

    import os
    import signal

    def killed(cfg):
        os.kill(os.getpid(), signal.SIGTERM)  # what PBS sends at the walltime or on qdel

    before = signal.getsignal(signal.SIGTERM)
    with pytest.raises(SystemExit) as stop:
        runner.run(cfg, allow_dirty=True, results_root=tmp_path / "raw", checkpoints_root=tmp_path / "ckpt",
                   runlog_path=log_path, model_factory=killed)  # fmt: skip
    assert stop.value.code == 128 + signal.SIGTERM and signal.getsignal(signal.SIGTERM) == before  # handler restored
    (out,) = (tmp_path / "raw").iterdir()
    assert json.loads((out / "manifest.json").read_text())["status"] == "failed"
    assert runlog.read_rows(log_path)[0]["status"] == "failed"


@pytest.mark.slow  # a tiny-model run end to end
def test_bookkeeping_after_completion_never_fails_a_run(tmp_path, monkeypatch):
    cfg = small_config(tmp_path)
    log_path = fresh_runlog(tmp_path)
    calls = []
    real_state = runner.git_state

    def flaky_state():
        calls.append(1)
        if len(calls) > 1:
            raise OSError("git went away")
        return real_state()

    def broken_update(path, run_id, **fields):
        raise OSError("run log unwritable")

    monkeypatch.setattr(runner, "git_state", flaky_state)
    monkeypatch.setattr(runner.runlog, "update_row", broken_update)
    out = runner.run(cfg, allow_dirty=True, results_root=tmp_path / "raw", checkpoints_root=tmp_path / "ckpt",
                     runlog_path=log_path, model_factory=tiny_model)  # fmt: skip
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "completed" and manifest["outputs"] and manifest["tokens"]
    assert "git went away" in manifest["git_end"]["error"] and manifest["code_changed_during_run"] is None
    assert "run log unwritable" in manifest["runlog_error"]


def test_the_dry_run_counts_the_starts_the_run_evaluates(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    cfg = load_config(SMOKE)
    other = dataclasses.replace(cfg, seed=3)  # starts follow eval.eval_seed, not the training seed
    assert runner.plan(other, limit=8)["eval"] == runner.plan(cfg, limit=8)["eval"]


def test_the_dry_run_estimate_stops_at_the_wall_clock(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    cfg = load_config(SMOKE)
    full = runner.plan(cfg, limit=8)
    capped = runner.plan(dataclasses.replace(cfg, train=dataclasses.replace(cfg.train, max_minutes=0.5)), limit=8)
    tps = capped["estimate"]["assumed_tokens_per_second"]
    expected = (capped["train"]["warmup_tokens_estimate"] / tps["train"] + 30) / 3600
    expected += capped["eval"]["tokens_estimate"] / tps["eval"] / 3600
    assert capped["estimate"]["gpu_hours"] == pytest.approx(expected) and expected < full["estimate"]["gpu_hours"]
    assert capped["train"]["max_minutes"] == 0.5 and full["train"]["max_minutes"] is None


def test_untracked_code_counts_as_dirt(tmp_path):
    import subprocess

    from qet.manifest import git_state

    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    (tmp_path / "a.txt").write_text("a")
    git("add", "a.txt")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "x")
    (tmp_path / "notes.md").write_text("notes")  # untracked, not code
    assert git_state(tmp_path)["dirty"] is False
    (tmp_path / "sitecustomize.py").write_text("")  # a top-level module can shadow imports
    (tmp_path / "third_party").mkdir()
    (tmp_path / "third_party" / "lib.py").write_text("")
    state = git_state(tmp_path)
    assert state["dirty"] and state["dirty_files"] == ["untracked: sitecustomize.py", "untracked: third_party/lib.py"]


def test_a_bad_setting_fails_before_the_run_folder_exists(tmp_path, monkeypatch):
    cfg = small_config(tmp_path)

    def no_data(*args, **kwargs):
        raise RuntimeError("no data")

    monkeypatch.setattr("qet.datasets.build_dataset", no_data)
    with pytest.raises(RuntimeError, match="no data"):
        runner.run(cfg, allow_dirty=True, results_root=tmp_path / "raw", runlog_path=fresh_runlog(tmp_path))
    assert not (tmp_path / "raw").exists()  # no orphan folder without a manifest


def test_runs_need_a_committed_config_and_a_clean_tree(tmp_path, monkeypatch):
    cfg = small_config(tmp_path)  # a config outside the repository is not committed
    with pytest.raises(SystemExit, match="committed config"):
        runner.run(cfg, results_root=tmp_path / "raw", runlog_path=fresh_runlog(tmp_path))
    monkeypatch.setattr(runner, "git_state", lambda: {"commit": "0" * 40, "dirty": True, "dirty_files": ["x.py"]})
    monkeypatch.setattr(runner, "is_committed", lambda path: True)
    with pytest.raises(SystemExit, match="clean working tree"):
        runner.run(load_config(SMOKE), results_root=tmp_path / "raw", runlog_path=fresh_runlog(tmp_path))
    assert not (tmp_path / "raw").exists()


_WRITER = """
import sys
from qet import runlog
path, prefix = sys.argv[1], sys.argv[2]
for i in range(15):
    runlog.append_row(path, {"run_id": f"{prefix}{i}", "status": "running"})
    runlog.update_row(path, f"{prefix}{i}", status="completed")
"""


def test_concurrent_writers_lose_no_rows(tmp_path):
    import subprocess
    import sys

    path = fresh_runlog(tmp_path)
    procs = [subprocess.Popen([sys.executable, "-c", _WRITER, str(path), p]) for p in "abc"]
    assert [proc.wait(timeout=120) for proc in procs] == [0, 0, 0]
    rows = runlog.read_rows(path)
    assert sorted(r["run_id"] for r in rows) == sorted(f"{p}{i}" for p in "abc" for i in range(15))
    assert {r["status"] for r in rows} == {"completed"}


def test_the_run_log_appends_and_updates_rows(tmp_path):
    path = fresh_runlog(tmp_path)
    runlog.append_row(path, {"run_id": "r1", "status": "running", "note": "a | b"})
    runlog.append_row(path, {"run_id": "r2", "status": "running"})
    runlog.update_row(path, "r1", status="completed", **{"GPU-h": "0.5"})
    rows = runlog.read_rows(path)
    assert [(r["run_id"], r["status"], r["GPU-h"], r["note"]) for r in rows] == [
        ("r1", "completed", "0.5", "a | b"), ("r2", "running", "", "")
    ]  # fmt: skip
    with pytest.raises(ValueError, match="already"):
        runlog.append_row(path, {"run_id": "r1"})
    with pytest.raises(ValueError, match="unknown"):
        runlog.append_row(path, {"run_id": "r3", "result": 0.9})
    assert Path(path).read_text().splitlines()[2] == runlog.HEADER
    assert "| a \\| b |" in Path(path).read_text()  # stored escaped


def warm_configs(tmp_path, **train):
    """A warm-up run (D-010, D-025d) and the train section its configuration runs share."""
    base = {"token_budget": 60_000, "batch_graphs": 2, "checkpoint_fractions": [1.0]}
    warm = small_config(tmp_path, name="warmup", train={**base, "warmup_steps": 2, "warmup_only": True, **train},
                        eval={"decoding": [{"mode": "greedy"}], "max_new_tokens": 6})  # fmt: skip
    return warm, base


def test_warm_up_and_parent_settings_are_validated(tmp_path):
    base = {"token_budget": 60_000}
    bad = [
        {**base, "warmup_only": True},  # no warm-up steps
        {**base, "warmup_only": True, "warmup_steps": 2, "parent": "x"},
        {**base, "warmup_steps": 2, "parent": "x"},  # a parent's run does no warm-up of its own
        {**base, "warmup_steps": 2, "format_check": {"at": [0], "tasks": 2}},  # checks belong to warm-up runs
        {**base, "warmup_steps": 2, "warmup_only": True, "format_check": {"at": [0, 3], "tasks": 2}},  # past the end
        {**base, "warmup_steps": 2, "warmup_only": True, "format_check": {"at": [2, 0], "tasks": 2}},  # unsorted
    ]
    for train in bad:
        with pytest.raises(ValueError):
            small_config(tmp_path, train=train)


@pytest.mark.slow  # three tiny-model runs
def test_a_shared_warm_up_equals_the_inline_one(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "c").mkdir()
    warm, base = warm_configs(tmp_path / "a", format_check={"at": [0, 2], "tasks": 2, "states_per_task": 2})
    log = fresh_runlog(tmp_path)
    common = {"allow_dirty": True, "results_root": tmp_path / "raw", "checkpoints_root": tmp_path / "ckpt",
              "runlog_path": log, "model_factory": tiny_model}  # fmt: skip
    w = runner.run(warm, **common)
    wm = json.loads((w / "manifest.json").read_text())
    metrics = json.loads((w / "metrics.json").read_text())
    assert wm["status"] == "completed" and wm["tokens"]["warmup"]["steps"] == 2 and wm["outputs"]["predictions"] == []
    assert [c["step"] for c in metrics["format_check"]] == [0, 2] and metrics["format_check"][0]["states"] == 4
    ((ckpt, files),) = wm["outputs"]["checkpoints"].items()
    assert ckpt.endswith("/warmup") and "adapter_model.safetensors" in files

    child = small_config(tmp_path / "b", name="child", train={**base, "parent": str(w)})
    inline = small_config(tmp_path / "c", name="inline", train={**base, "warmup_steps": 2})
    c, i = runner.run(child, **common), runner.run(inline, **common)
    cm, im = (json.loads((x / "manifest.json").read_text()) for x in (c, i))
    assert cm["parent_checkpoint"]["run_id"] == wm["run_id"] and cm["parent_checkpoint"]["files"] == files
    assert cm["tokens"]["budget_used"] == im["tokens"]["budget_used"]
    assert cm["tokens"]["gradient"] == im["tokens"]["gradient"] and cm["tokens"]["parent"]["gradient"] > 0

    from safetensors.torch import load_file

    def final(m):
        (path,) = [p for p in m["outputs"]["checkpoints"] if p.endswith("frac-1")]
        return load_file(str(Path(path) / "adapter_model.safetensors"))

    a, b = final(cm), final(im)
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)  # D-025d changes nothing on CPU


@pytest.mark.slow  # two tiny-model runs
def test_a_parent_of_another_seed_is_refused(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    warm, base = warm_configs(tmp_path / "a")
    log = fresh_runlog(tmp_path)
    common = {"allow_dirty": True, "results_root": tmp_path / "raw", "checkpoints_root": tmp_path / "ckpt",
              "runlog_path": log, "model_factory": tiny_model}  # fmt: skip
    w = runner.run(warm, **common)
    other = small_config(tmp_path / "b", name="child", seed=1, train={**base, "parent": str(w)})
    with pytest.raises(ValueError, match="has seed 0, this run 1"):
        runner.run(other, **common)
    assert len(list((tmp_path / "raw").iterdir())) == 1  # refused before its folder existed

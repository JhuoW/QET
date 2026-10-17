"""E1.6: run configs, the dry run, manifests (every EVAL_PLAN §10 field) and the run log. Real runs here use a tiny
random model and temporary folders; the GPU runs are E1.7's."""

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
    ]
    for i, data in enumerate(bad):
        path = tmp_path / f"bad{i}.yaml"
        path.write_text(yaml.safe_dump(data))
        with pytest.raises(ValueError):
            load_config(path)


def snapshot():
    roots = [REPO / "results", REPO / "checkpoints"]
    files = sorted(str(p) for r in roots if r.exists() for p in r.rglob("*"))
    return files, (REPO / "docs" / "RUNLOG.md").read_bytes()


def test_the_dry_run_prints_the_plan_and_writes_nothing(capsys, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    before = snapshot()
    runner.main(["--config", str(SMOKE), "--dry-run", "--limit", "6"])
    out = capsys.readouterr().out
    assert snapshot() == before
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
    assert (tmp_path / "ckpt" / out.name / "frac-1" / "qet_checkpoint.json").exists()


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
    assert manifest["status"] == "failed" and "no model" in manifest["error"]
    assert runlog.read_rows(log_path)[0]["status"] == "failed"


def test_runs_need_a_committed_config_and_a_clean_tree(tmp_path, monkeypatch):
    cfg = small_config(tmp_path)  # a config outside the repository is not committed
    with pytest.raises(SystemExit, match="committed config"):
        runner.run(cfg, results_root=tmp_path / "raw", runlog_path=fresh_runlog(tmp_path))
    monkeypatch.setattr(runner, "git_state", lambda: {"commit": "0" * 40, "dirty": True, "dirty_files": ["x.py"]})
    monkeypatch.setattr(runner, "is_committed", lambda path: True)
    with pytest.raises(SystemExit, match="clean working tree"):
        runner.run(load_config(SMOKE), results_root=tmp_path / "raw", runlog_path=fresh_runlog(tmp_path))
    assert not (tmp_path / "raw").exists()


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

"""The pre-registered M1 criteria (EVAL_PLAN §9, D-030e) on synthetic evaluation runs."""

import json
import random

import pytest

from qet.eval.criteria import EvalRun, by_seed, criterion_1, criterion_2, criterion_3, load_runs

QWEN = "Qwen/Qwen2.5-7B-Instruct"
TASKS = [f"t{i}" for i in range(300)]


def runs_with(rates: dict[str, float], seed0: int = 0) -> list[EvalRun]:
    """Three seeds per configuration with the given greedy success rates, drawn per task and seed."""
    out = []
    for c, p in rates.items():
        for s in range(3):
            rng = random.Random(f"{c}-{s}-{seed0}")
            out.append(EvalRun(f"run-{c}-{s}", c, QWEN, s, "test", "frac-1", {t: rng.random() < p for t in TASKS}))
    return out


def test_criterion_1_holds_when_6_beats_3_and_4_on_wide_graphs():
    widths = {t: (5 if i % 2 else 1) for i, t in enumerate(TASKS)}
    res = criterion_1(runs_with({"6": 0.6, "3": 0.3, "4": 0.3}), widths, QWEN, resamples=2000)
    assert res["claim"] == "holds" and res["pool"] == 150 and all(d["low"] > 0 for d in res["differences"])
    res = criterion_1(runs_with({"6": 0.4, "3": 0.4, "4": 0.2}), widths, QWEN, resamples=2000)
    assert res["claim"] == "fails"  # 6 - 3 straddles 0
    assert set(res["differences"][0]["mcnemar_per_seed"]) == {0, 1, 2}


def test_criterion_2_reads_the_interface_condition():
    res = criterion_2(runs_with({"1": 0.2, "2": 0.5, "6": 0.5}), set(TASKS), QWEN, resamples=2000)
    assert res["condition_met"] and "interface" in res["reading"]
    res = criterion_2(runs_with({"1": 0.2, "2": 0.4, "6": 0.7}), set(TASKS), QWEN, resamples=2000)
    assert not res["condition_met"]


def resume_rows(gap: float, seed: int, c: str) -> list[dict]:
    rng = random.Random(f"resume-{c}-{seed}")
    rows = []
    for t in TASKS:
        rows.append({"task_id": t, "condition": "on_path", "history": "full", "success": rng.random() < 0.7})
        rows.append({"task_id": t, "condition": "off_path", "history": "full", "success": rng.random() < 0.7 - gap})
    return rows


def test_criterion_3_compares_off_path_gaps():
    def runs(gaps):
        out = []
        for c, g in gaps.items():
            for s in range(3):
                out.append(EvalRun(f"r{c}{s}", c, QWEN, s, "test", "frac-1", resume=resume_rows(g, s, c)))
        return out

    assert criterion_3(runs({"3": 0.4, "5": 0.4, "6": 0.05}), QWEN, resamples=2000)["claim"] == "holds"
    assert criterion_3(runs({"3": 0.1, "5": 0.4, "6": 0.1}), QWEN, resamples=2000)["claim"] == "fails"


def test_runs_load_from_their_folders_and_seeds_are_unique(tmp_path):
    folder = tmp_path / "M1-eval-x"
    (folder / "data").mkdir(parents=True)
    (folder / "predictions").mkdir()
    (folder / "manifest.json").write_text(json.dumps({
        "run_id": "M1-eval-x", "status": "completed", "configuration": {"id": "6"}, "backbone": {"id": QWEN},
        "seed": 1, "evaluated_checkpoint": {"label": "frac-1"}}))  # fmt: skip
    (folder / "data" / "data_manifest.json").write_text(json.dumps({"spec": {"eval": {"split": "test"}}}))
    rows = [{"task_id": "a", "decoding": "greedy", "condition": "standard", "success": True},
            {"task_id": "a", "decoding": "sample-t1-p1", "condition": "standard", "success": False},
            {"task_id": "a", "decoding": "greedy", "condition": "on_path", "success": True}]  # fmt: skip
    (folder / "predictions" / "greedy.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    (run,) = load_runs([folder])
    assert run.greedy == {"a": True} and run.samples == {"a": [False]} and len(run.resume) == 1
    with pytest.raises(ValueError, match="two evaluation runs"):
        by_seed([run, run], "6", QWEN)

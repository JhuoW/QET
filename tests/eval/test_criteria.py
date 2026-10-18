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
        "seed": 1,
        "evaluated_checkpoint": {"label": "frac-1", "run_id": "M1-c6-qwen7b-s1-20261018T060956Z"}}))  # fmt: skip
    (folder / "data" / "data_manifest.json").write_text(json.dumps({"spec": {"eval": {"split": "test"}}}))
    rows = [{"task_id": "a", "decoding": "greedy", "condition": "standard", "success": True},
            {"task_id": "a", "decoding": "sample-t1-p1", "condition": "standard", "success": False},
            {"task_id": "a", "decoding": "greedy", "condition": "on_path", "success": True}]  # fmt: skip
    (folder / "predictions" / "greedy.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    (run,) = load_runs([folder])
    assert run.greedy == {"a": True} and run.samples == {"a": [False]} and len(run.resume) == 1
    with pytest.raises(ValueError, match="two evaluation runs"):
        by_seed([run, run], "6", QWEN)


def test_variants_keep_the_extras_apart_from_the_matrix():
    from qet.eval.criteria import variant_of

    assert variant_of("M1-c6-qwen7b-s0-20261018T045428Z") == ""
    assert variant_of("M1-c6-qwen7b-n1250-s0-20261018T090000Z") == "n1250"
    assert variant_of("M1-a4k2-qwen7b-s0-20261018T060959Z") == "k2" and variant_of(None) == "base"
    main = EvalRun("m", "6", QWEN, 0, "test", "frac-1", {"t": True})
    extra = EvalRun("e", "6", QWEN, 0, "test", "frac-1", {"t": False}, variant="fi")
    assert by_seed([main, extra], "6", QWEN) == {0: main} and by_seed([main, extra], "6", QWEN, variant="fi") == {
        0: extra
    }


def test_the_gate_b_report_quotes_the_criteria_and_computes_them():
    import sys

    from qet.manifest import REPO

    sys.path.insert(0, str(REPO / "scripts"))
    import gate_report

    runs = runs_with({"1": 0.2, "2": 0.4, "3": 0.3, "4": 0.3, "5": 0.35, "6": 0.6})
    for r in runs:
        r.samples = {t: [r.greedy[t]] * 8 for t in TASKS}
        if r.configuration in ("3", "5", "6"):
            r.resume = resume_rows({"3": 0.3, "5": 0.3, "6": 0.05}[r.configuration], r.seed, r.configuration)
    widths = {t: (5 if i % 2 else 2) for i, t in enumerate(TASKS)}
    section = "## 9. Falsification criteria\nCommon terms: ...\n1. Greedy success on graphs of measured width >= 3: ..."
    for r in runs:
        r.trained_from = f"M1-c{r.configuration}-qwen7b-s{r.seed}-20261018T000000Z"
    log = [{"phase": "train", "graphs": ["g1", "g2"]}, {"phase": "train", "graphs": ["g2", "g3"]}]
    one = gate_report.exposure({"budgets": {"train": {"token_budget": 100}}, "tokens": {"budget_used": 101}}, log,
                               {"spec": {"train": {"count": 8}}})  # fmt: skip
    assert one == {"budget": 100, "used": 101, "graphs": 8, "seen": 3, "visits": 4}
    training = {r.trained_from: one for r in runs}
    text = gate_report.gate_b(runs, widths, section, [("x", "0" * 64)], resamples=500, training=training)
    assert section in text and "| 1 | 6 - 3 |" in text and "| 2 | 2 - 1 |" in text and "| 3 | gap 3 - gap 6 |" in text
    assert "Criterion 1: the objective claim holds" in text and "Criterion 3: claim C4 holds" in text
    assert "## 4. Resume@prefix" in text and "**STOP.**" in text and "does not carry" not in text
    assert "| Qwen2.5-7B-Instruct | main configuration 6 | 3 | 1.0100-1.0100 | 1.00% | 3-3 | 8 | 0.50-0.50 |" in text

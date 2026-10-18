"""EVAL_PLAN §8 statistics: clustered paired bootstrap and the exact McNemar test."""

import random

import pytest

from qet.eval.stats import cluster_bootstrap, mcnemar, paired_difference, task_means


def test_mcnemar_matches_the_binomial_tail():
    a = {f"t{i}": i < 12 for i in range(40)}  # 12 successes
    b = {f"t{i}": 10 <= i < 12 or i >= 30 for i in range(40)}  # shares 2, adds 10
    res = mcnemar(a, b)
    assert (res["a_only"], res["b_only"]) == (10, 10) and res["p_value"] == 1.0
    a = {f"t{i}": i < 10 for i in range(20)}
    b = {f"t{i}": i in (10, 11) for i in range(20)}
    assert mcnemar(a, b)["p_value"] == pytest.approx(2 * (1 + 12 + 66) / 4096)  # 10 vs 2 discordant
    assert mcnemar({"x": True}, {"x": True})["p_value"] == 1.0


def test_units_are_averaged_within_tasks_before_resampling():
    # one task with many units must count once: its eight rollouts are not eight observations
    values = {"big": [1.0] * 8, "a": [0.0], "b": [0.0], "c": [0.0]}
    assert task_means(values)["big"] == 1.0
    res = cluster_bootstrap(task_means(values), resamples=2000, seed=1)
    assert res["estimate"] == 0.25 and 0.0 <= res["low"] <= 0.25 <= res["high"] <= 1.0


def test_the_interval_covers_the_truth_and_is_deterministic():
    rng = random.Random(0)
    per_task = {f"t{i}": rng.random() < 0.6 for i in range(400)}
    one = cluster_bootstrap({t: float(v) for t, v in per_task.items()}, resamples=2000, seed=3)
    two = cluster_bootstrap({t: float(v) for t, v in per_task.items()}, resamples=2000, seed=3)
    assert one == two and one["low"] < 0.6 < one["high"] and one["high"] - one["low"] < 0.12


def test_paired_differences_need_the_same_tasks_and_flag_significance():
    a = {f"t{i}": [1.0, 1.0] for i in range(100)}
    b = {f"t{i}": [0.0, 1.0] for i in range(100)}
    res = paired_difference(a, b, resamples=1000)
    assert res["estimate"] == 0.5 and res["low"] == res["high"] == 0.5 and res["significant"]
    with pytest.raises(ValueError, match="different tasks"):
        paired_difference(a, {"t0": [1.0]})

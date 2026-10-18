"""Run one config end to end (AGENTS.md "Running experiments"):

    python -m qet.run --config configs/<milestone>/<name>.yaml [--dry-run] [--limit N]

`--dry-run` prints the plan, the scored and gradient token counts and a GPU-hour estimate, and writes nothing. A real
run starts from a committed config and a clean working tree, writes `results/raw/<run_id>/` (manifest, data, training
log, predictions, metrics) and `checkpoints/<run_id>/`, and appends a row to docs/RUNLOG.md that it updates at the end.
A run trains (format warm-up of D-010, then the configuration) when the config has a `train` section, and evaluates the
final model: every test task under every decoding setting, and Resume@prefix on the first test tasks.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from qet import runlog
from qet.config import RunConfig, load_config
from qet.manifest import REPO, Manifest, git_state, hardware, is_committed, libraries, sha256_json, utc_now

RESULTS = REPO / "results" / "raw"
CHECKPOINTS = REPO / "checkpoints"
RUNLOG_PATH = REPO / "docs" / "RUNLOG.md"


def _rel(path: str | Path) -> str:
    """A repository-relative path when the path is inside the repository. Symlinks are kept, not resolved: a pinned job
    checkout (D-025b) links results/raw and checkpoints to the main repository's folders."""
    p = Path(os.path.abspath(path))
    return str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p)


def warmup_method(cfg: RunConfig):
    """D-010: demo-order SFT in the configuration's own renderer."""
    from qet.train.methods import Method

    m = cfg.method
    return Method(f"warmup-{m.config_id}", "format_warmup", m.renderer, "path_sft", order="demo")


# ----------------------------------------------------------------------------- the plan (dry run)
class DryRunCache:
    """What the dry runs of several configs share in one process: datasets, chat formats, and rendered prompts and
    targets. A prompt depends only on the task, the state, the renderer, the truncation, the contracts and the
    tokenizer (SPEC §5.6), and a target only on the action and the tokenizer, so pack builders that agree on those
    share them (the builders' own caches, swapped for shared ones)."""

    def __init__(self) -> None:
        self.datasets: dict[str, Any] = {}
        self.formats: dict[str, Any] = {}
        self.prompts: dict[tuple, dict] = {}
        self.targets: dict[str, dict] = {}

    def dataset(self, data: Any, limit: int | None) -> Any:
        from qet.datasets import build_dataset

        key = repr((data, limit))
        if key not in self.datasets:
            self.datasets[key] = build_dataset(data, limit=limit)
        return self.datasets[key]

    def format(self, model_id: str) -> Any:
        from qet.train.chat import ChatFormat

        if model_id not in self.formats:
            self.formats[model_id] = ChatFormat.load(model_id)
        return self.formats[model_id]

    def share(self, builder: Any, cfg: RunConfig, renderer: str) -> None:
        key = (cfg.backbone, renderer, json.dumps(cfg.data.truncation, sort_keys=True), repr(cfg.data))
        builder._prompts = self.prompts.setdefault(key, {})
        builder._targets = self.targets.setdefault(cfg.backbone + repr(cfg.data), {})


def plan(cfg: RunConfig, *, limit: int | None = None, cache: DryRunCache | None = None) -> dict[str, Any]:
    """What the run would do and cost, from the real data and prompts; nothing is written."""
    from qet.core.render import render_messages
    from qet.eval.resume_prefix import starts
    from qet.eval.runner import decoding_settings
    from qet.train.trainer import PackBuilder, pass1_edges

    cache = cache or DryRunCache()
    ds = cache.dataset(cfg.data, limit)
    fmt = cache.format(cfg.backbone)
    m = cfg.method
    out: dict[str, Any] = {
        "run": {"milestone": cfg.milestone, "name": cfg.name, "configuration": f"{m.config_id} ({m.name})",
                "seed": cfg.seed, "backbone": cfg.backbone, "config": cfg.path, "config_sha256": cfg.sha256},
        "data": {"train_graphs": len(ds.train_records), "test_tasks": len(ds.test_tasks), **ds.spec},
    }  # fmt: skip
    train_tokens = 0
    if cfg.train:
        t = cfg.train

        def graph_tokens(method, with_pass1: bool) -> tuple[list[int], list[int], list[int]]:
            b = PackBuilder(fmt, ds.contracts, method, run_seed=cfg.seed, truncation=cfg.data.truncation, k=t.k,
                            s_max=t.s_max)  # fmt: skip
            cache.share(b, cfg, method.renderer)
            packs = [b.build(r, 0) for r in ds.train_records]
            edges = [pass1_edges(pk, method) for pk in packs]  # the edges pass 2 can weight, too
            p2 = [sum(len(pk.pair(e)) for e in es) for pk, es in zip(packs, edges, strict=True)]
            p1 = p2 if with_pass1 else [0] * len(packs)
            return p1, p2, [len(pk.lat.states) for pk in packs]

        # only the packs this run trains on: a warm-up run has no configuration phase, a run with a parent no warm-up
        p1, p2, states = ([], [], []) if t.warmup_only else graph_tokens(m, m.pass1)
        per_epoch = sum(p1) + sum(p2)
        w2 = graph_tokens(warmup_method(cfg), False)[1] if t.warmup_steps else []
        warmup = round(t.warmup_steps * t.batch_graphs * sum(w2) / max(1, len(w2)))
        if t.parent:  # the parent's warm-up tokens count toward this run's budget (D-010)
            try:
                warmup = parent_record(cfg, ds)["tokens_used"]
            except FileNotFoundError:  # the warm-up run has not run yet: assume this config's own warm-up length
                pass
        steps_per_epoch = math.ceil(len(p2) / t.batch_graphs)
        epochs = 0.0 if t.warmup_only else max(0, t.token_budget - warmup) / per_epoch if per_epoch else 0.0
        train_tokens = warmup if t.warmup_only else t.token_budget - (warmup if t.parent else 0)
        out["train"] = {
            "token_budget": t.token_budget,
            "warmup_tokens_estimate": warmup,
            "tokens_per_epoch": {"scored": sum(p1), "gradient": sum(p2)},
            "tokens_per_graph": {
                "mean": per_epoch / max(1, len(p2)),
                "max": max((a + b for a, b in zip(p1, p2, strict=True)), default=0),
            },
            "states_per_graph": {"mean": sum(states) / max(1, len(states)), "max": max(states, default=0)},
            "epochs": epochs,
            "steps": math.ceil(epochs * steps_per_epoch) + t.warmup_steps,
            "warmup_only": t.warmup_only,
            "parent": t.parent,
            "max_minutes": t.max_minutes,  # epochs and steps above assume the budget, not this stop, ends training
        }  # fmt: skip
    episodes = eval_tokens = 0
    task_x = {t.task_id: t.public() for t in ds.test_tasks}
    per_task = {}
    for rec in ds.test_records:  # prompt tokens along the demo path, one generation per step
        steps = rec.steps()
        prompts = [len(fmt.prompt_ids(render_messages(task_x[rec.task_id], steps[:k], ds.contracts, m.renderer,
                                                      cfg.data.truncation))) for k in range(rec.n + 1)]  # fmt: skip
        per_task[rec.task_id] = (prompts, sum(prompts) + (rec.n + 1) * cfg.eval.max_new_tokens)
    if cfg.train and cfg.train.warmup_only:  # no episodes; each format check acts once per sampled state (D-010)
        fc = cfg.train.format_check or {"at": [], "tasks": 0}
        checked = list(per_task.values())[: int(fc["tasks"])]
        per_state = sum(sum(v[0]) / len(v[0]) for v in checked) / max(1, len(checked)) + cfg.eval.max_new_tokens
        eval_tokens = round(len(fc["at"]) * len(checked) * int(fc.get("states_per_task", 2)) * per_state)
        out["eval"] = {"episodes": 0, "format_checks": len(fc["at"]), "tokens_estimate": eval_tokens}
    skip = (cfg.train and cfg.train.warmup_only) or not cfg.eval.standard
    for _, _, rollouts in decoding_settings(cfg.eval) if not skip else ():
        episodes += rollouts * len(per_task)
        eval_tokens += rollouts * sum(v[1] for v in per_task.values())
    for rec in ds.test_records[: cfg.eval.resume_prefix_tasks]:
        prompts, _ = per_task[rec.task_id]
        for s in starts(rec, cfg.eval.eval_seed, per_task=cfg.eval.resume_starts):  # the same starts the run evaluates
            episodes += 1
            eval_tokens += sum(prompts[s.depth :]) + (rec.n + 1 - s.depth) * cfg.eval.max_new_tokens
    out.setdefault("eval", {"episodes": episodes, "tokens_estimate": eval_tokens})
    tps_train, tps_eval = cfg.estimate.get("train_tokens_per_second"), cfg.estimate.get("eval_tokens_per_second")
    hours = None
    if (tps_train or not train_tokens) and tps_eval:
        train_s = train_tokens / tps_train if train_tokens else 0.0
        if cfg.train and cfg.train.max_minutes is not None:  # the wall-clock stop ends training after the warm-up
            train_s = min(train_s, out["train"]["warmup_tokens_estimate"] / tps_train + 60 * cfg.train.max_minutes)
        hours = train_s / 3600 + eval_tokens / tps_eval / 3600
    out["estimate"] = {"gpu_hours": hours, "assumed_tokens_per_second": {"train": tps_train, "eval": tps_eval},
                       "note": "throughputs are the config's assumptions until a pilot measures them"}  # fmt: skip
    return out


def format_plan(p: dict[str, Any]) -> str:
    lines = [f"{k}: {json.dumps(v, ensure_ascii=False)}" for k, v in p.items()]
    est = p["estimate"]["gpu_hours"]
    lines.append(
        f"estimated GPU-hours: {'unknown' if est is None else f'{est:.3f}'} (assumed throughput, not measured)"
    )
    return "\n".join(lines)


# ----------------------------------------------------------------------------- a real run
def load_model(cfg: RunConfig):
    """The backbone at its pinned revision, on the GPU when there is one, in eval mode."""
    import torch
    from transformers import AutoModelForCausalLM

    from qet.train.chat import backbone

    b = backbone(cfg.backbone)
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[cfg.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        b.model_id, revision=b.revision, dtype=dtype, attn_implementation="sdpa"
    )
    return model.to("cuda" if torch.cuda.is_available() else "cpu").eval()


def _train_data(spec: dict[str, Any]) -> dict[str, Any]:
    """The part of a data spec that fixes the training records (everything but the evaluation split)."""
    return {k: v for k, v in spec.items() if k not in ("eval", "test")}


def parent_record(cfg: RunConfig, ds: Any) -> dict[str, Any] | None:
    """The warm-up run a config starts from (D-010, D-025d), checked: a completed warm-up run with the same backbone,
    seed, renderer, LoRA shape and training data, whose checkpoint files match the hashes in its manifest."""
    from qet.manifest import sha256_file
    from qet.train.chat import backbone

    t = cfg.train
    if t is None or not t.parent:
        return None
    folder = Path(t.parent) if Path(t.parent).is_absolute() else REPO / t.parent
    pm = json.loads((folder / "manifest.json").read_text())
    pt = (pm.get("budgets") or {}).get("train") or {}
    dm = (pm.get("data_manifest") or {}).get("path")
    data = json.loads((Path(dm) if Path(dm).is_absolute() else REPO / dm).read_text()) if dm else {}
    problems = [msg for bad, msg in [
        (pm.get("status") != "completed", "did not complete"),
        (not pt.get("warmup_only"), "is not a warm-up run"),
        (pm["backbone"]["id"] != cfg.backbone or pm["backbone"]["revision"] != backbone(cfg.backbone).revision,
         "has another backbone"),
        (pm["seed"] != cfg.seed, f"has seed {pm['seed']}, this run {cfg.seed}"),
        (pm["configuration"]["renderer"] != cfg.method.renderer, "used another renderer"),
        ((pt.get("lora_rank"), pt.get("lora_alpha")) != (t.lora_rank, t.lora_alpha), "has another LoRA shape"),
        (_train_data(data.get("spec", {})) != _train_data(ds.spec), "trained on other data"),
    ] if bad]  # fmt: skip
    if problems:
        raise ValueError(f"parent {folder}: " + "; ".join(problems))
    (rel, files), *more = pm["outputs"]["checkpoints"].items()
    if more:
        raise ValueError(f"parent {folder} has more than one checkpoint")
    path = Path(rel) if Path(rel).is_absolute() else REPO / rel
    for name, digest in files.items():
        if sha256_file(path / name) != digest:
            raise ValueError(f"{path / name} does not match the parent's manifest")
    tok = pm["tokens"]
    return {"run_id": pm["run_id"], "path": rel, "files": files, "steps": tok["warmup"]["steps"],
            "scored": tok["scored"], "gradient": tok["gradient"], "tokens_used": tok["budget_used"]}  # fmt: skip


def checkpoint_record(cfg: RunConfig) -> dict[str, Any] | None:
    """The trained checkpoint an evaluation-only config evaluates (M1.7), checked: a completed training run of the
    same configuration, backbone and seed, whose checkpoint files match the hashes in its manifest."""
    from qet.manifest import sha256_file
    from qet.train.chat import backbone

    if not cfg.checkpoint:
        return None
    folder = Path(cfg.checkpoint["run"])
    folder = folder if folder.is_absolute() else REPO / folder
    tm = json.loads((folder / "manifest.json").read_text())
    problems = [msg for bad, msg in [
        (tm.get("status") != "completed", "did not complete"),
        (tm["configuration"]["id"] != cfg.configuration, f"trained configuration {tm['configuration']['id']}"),
        (tm["backbone"]["id"] != cfg.backbone or tm["backbone"]["revision"] != backbone(cfg.backbone).revision,
         "has another backbone"),
        (tm["seed"] != cfg.seed, f"has seed {tm['seed']}, this run {cfg.seed}"),
    ] if bad]  # fmt: skip
    if problems:
        raise ValueError(f"checkpoint run {folder}: " + "; ".join(problems))
    label = cfg.checkpoint["label"]
    hits = [(rel, files) for rel, files in tm["outputs"]["checkpoints"].items() if Path(rel).name == label]
    if len(hits) != 1:
        raise ValueError(f"checkpoint run {folder} has no single checkpoint labelled {label!r}")
    ((rel, files),) = hits
    path = Path(rel) if Path(rel).is_absolute() else REPO / rel
    for name, digest in files.items():
        if sha256_file(path / name) != digest:
            raise ValueError(f"{path / name} does not match its training run's manifest")
    meta = json.loads((path / "qet_checkpoint.json").read_text())
    return {"run_id": tm["run_id"], "label": label, "path": rel, "files": files,
            **{k: meta.get(k) for k in ("step", "tokens_used", "budget_fraction")}}  # fmt: skip


def _terminate(signum: int, frame: Any) -> None:
    """A scheduler's SIGTERM (walltime, qdel) becomes SystemExit, so the run is recorded as failed, not left running."""
    raise SystemExit(128 + signum)


def _log_safely(manifest: Manifest, path: str | Path, run_id: str, **fields: Any) -> None:
    """Update the run's RUNLOG row; a failure there is recorded in the manifest and never changes the run's status."""
    try:
        runlog.update_row(path, run_id, **fields)
    except Exception as e:  # noqa: BLE001 - recorded, not raised
        manifest.update(runlog_error=f"{type(e).__name__}: {e}")


def run(
    cfg: RunConfig,
    *,
    limit: int | None = None,
    allow_dirty: bool = False,
    results_root: str | Path = RESULTS,
    checkpoints_root: str | Path = CHECKPOINTS,
    runlog_path: str | Path = RUNLOG_PATH,
    model_factory: Callable[[RunConfig], Any] | None = None,
) -> Path:
    import torch

    from qet.datasets import build_dataset, write_dataset
    from qet.eval.resume_prefix import starts
    from qet.eval.runner import decoding_settings
    from qet.train.chat import ChatFormat, backbone

    git = git_state()
    if not allow_dirty and (git["dirty"] or not is_committed(cfg.path)):
        raise SystemExit(f"a run needs a committed config and a clean working tree (dirty: {git['dirty_files']})")
    t0 = time.perf_counter()
    gpus = 1 if torch.cuda.is_available() else 0
    # Whatever can fail on the config or the data fails here, before the run folder exists.
    fmt = ChatFormat.load(cfg.backbone)
    ds = build_dataset(cfg.data, limit=limit)
    decodings = decoding_settings(cfg.eval)
    parent = parent_record(cfg, ds)
    evaluated = checkpoint_record(cfg)
    resume = [{"task_id": rec.task_id, "condition": s.condition, "depth": s.depth, "order": list(s.order)}
              for rec in ds.test_records[: cfg.eval.resume_prefix_tasks]
              for s in starts(rec, cfg.eval.eval_seed, per_task=cfg.eval.resume_starts)]  # fmt: skip
    created = utc_now()
    run_id = f"{cfg.milestone}-{cfg.name}-s{cfg.seed}-{created:%Y%m%dT%H%M%SZ}"
    out = Path(results_root) / run_id
    m, t = cfg.method, cfg.train
    fields = {
        "run_id": run_id,
        "created_at": created.isoformat(),
        "git": git,
        "config": {"path": _rel(cfg.path), "sha256": cfg.sha256},
        "seed": cfg.seed,
        "configuration": {"id": m.config_id, "name": m.name, "renderer": m.renderer, "weights": m.weights,
                          "order": m.order, "paths": m.paths, "force_sublattice": m.force_sublattice},
        "lambda": m.lam,
        "s_max": t.s_max if t else None,
        "k": t.k if t else None,
        "backbone": {"id": cfg.backbone, "revision": backbone(cfg.backbone).revision, "dtype": cfg.dtype},
        "tokenizer": fmt.fingerprint(),
        "data_manifest": None,  # written below
        "benchmark": {"commit": git["commit"], **ds.benchmark, "contract_source": ds.contract_source,
                      "contracts_sha256": sha256_json(ds.contracts)},
        "judge": None,
        "decoding": [{"name": n, "mode": d.mode, "temperature": d.temperature, "top_p": d.top_p, "rollouts": r,
                      "max_new_tokens": d.max_new_tokens} for n, d, r in decodings],
        "budgets": {"train": None if t is None else {k: getattr(t, k) for k in t.__dataclass_fields__},
                    "step_budget": ds.benchmark.get("step_budget"),
                    "resume_prefix_tasks": cfg.eval.resume_prefix_tasks, "eval_seed": cfg.eval.eval_seed,
                    "limit": limit},
        "hardware": hardware(),
        "libraries": libraries(),
        "tokens": None,
        "gpu_hours": None,
        "parent_checkpoint": parent,  # the shared warm-up this run starts from, or None (D-010, D-025d)
        "evaluated_checkpoint": evaluated,  # evaluation-only runs: the trained checkpoint they evaluate (M1.7)
        "status": "running",
    }  # fmt: skip
    out.mkdir(parents=True)
    manifest = Manifest(out / "manifest.json", fields)
    try:
        previous = signal.signal(signal.SIGTERM, _terminate)
    except ValueError:  # not the main thread: no handler
        previous = None
    row_added = False
    try:
        try:
            runlog.append_row(runlog_path, {
                "run_id": run_id, "submitted": created.isoformat(), "milestone": cfg.milestone,
                "config": _rel(cfg.path), "method": f"{m.config_id} {m.name}", "backbone": cfg.backbone,
                "seed": cfg.seed, "commit": git["commit"][:7] + ("+dirty" if git["dirty"] else ""),
                "status": "running", "GPU-h": "", "path": _rel(out), "note": f"limit={limit}" if limit else "",
            })  # fmt: skip
            row_added = True
            data_path, data_sha = write_dataset(ds, out / "data", resume_starts=resume)
            manifest.update(data_manifest={"path": _rel(data_path), "sha256": data_sha})
            tokens, outputs, seconds = _execute(
                cfg, ds, fmt, out, Path(checkpoints_root) / run_id, model_factory, parent, evaluated
            )
            total = (time.perf_counter() - t0) / 3600 * gpus
            phases = {k: v / 3600 * gpus for k, v in seconds.items()}
            manifest.update(tokens=tokens, gpu_hours=total, gpu_hours_by_phase=phases, outputs=outputs,
                            status="completed", finished_at=utc_now().isoformat())  # fmt: skip
        except BaseException as e:
            total = (time.perf_counter() - t0) / 3600 * gpus
            manifest.update(status="failed", error=f"{type(e).__name__}: {e}", finished_at=utc_now().isoformat(),
                            gpu_hours=total)  # fmt: skip
            if row_added:
                _log_safely(manifest, runlog_path, run_id, status="failed", note=type(e).__name__,
                            **{"GPU-h": f"{total:.3f}"})  # fmt: skip
            raise
        # Bookkeeping after completion never turns a completed run into a failed one.
        try:
            git_end = git_state()
            changed = (git_end["commit"], git_end["dirty_files"]) != (git["commit"], git["dirty_files"])
        except Exception as e:  # noqa: BLE001 - recorded, not raised
            git_end, changed = {"error": f"{type(e).__name__}: {e}"}, None
        manifest.update(git_end=git_end, code_changed_during_run=changed)
        _log_safely(manifest, runlog_path, run_id, status="completed", **{"GPU-h": f"{total:.3f}"})
    finally:
        if previous is not None:
            signal.signal(signal.SIGTERM, previous)
    return out


def _execute(cfg, ds, fmt, out: Path, ckpt_dir: Path, model_factory, parent=None, evaluated=None) -> tuple:
    import torch

    from qet.eval.metrics import summarize
    from qet.eval.resume_prefix import evaluate_resume
    from qet.eval.runner import decoding_settings, evaluate_tasks
    from qet.infer.decoding import HFSampler
    from qet.manifest import sha256_file
    from qet.train.scoring import TokenCounts
    from qet.train.trainer import LMScorer, PackBuilder, TrainConfig, add_lora, train

    clock = time.perf_counter()
    seconds: dict[str, float] = {}
    torch.manual_seed(cfg.seed)
    vllm = cfg.eval.backend == "vllm"  # evaluation-only: the engine loads the backbone (and adapter) itself
    model = None if vllm else (model_factory or load_model)(cfg)
    if evaluated is not None and not vllm:  # checked against its training run's manifest by checkpoint_record
        from peft import PeftModel

        path = Path(evaluated["path"]) if Path(evaluated["path"]).is_absolute() else REPO / evaluated["path"]
        model = PeftModel.from_pretrained(model, str(path)).eval()
    seconds["load"] = time.perf_counter() - clock
    tokens: dict[str, Any] = {}
    checkpoints: dict[str, dict[str, str]] = {}
    checks: list[dict[str, Any]] = []  # format checks of a warm-up run (D-010)
    t, m = cfg.train, cfg.method
    clock = time.perf_counter()
    with open(out / "train_log.jsonl", "w", encoding="utf-8") as log:

        def logger(phase: str) -> Callable[[dict[str, Any]], None]:
            def write(row: dict[str, Any]) -> None:
                if torch.cuda.is_available():  # the run's peak so far, which checks the memory sizing (D-024e)
                    row = {**row, "max_memory_gib": torch.cuda.max_memory_allocated() / 2**30}
                log.write(json.dumps({"phase": phase, **row}) + "\n")
                log.flush()  # readable while the job runs

            return write

        if t is not None:
            if parent is not None:  # the shared warm-up's adapter, checked against its manifest by parent_record
                from peft import PeftModel

                path = Path(parent["path"]) if Path(parent["path"]).is_absolute() else REPO / parent["path"]
                model = PeftModel.from_pretrained(model, str(path), is_trainable=True).eval()
            else:
                model = add_lora(model, rank=t.lora_rank, alpha=t.lora_alpha, seed=cfg.seed)
            scorer = LMScorer(model, pad_id=fmt.eot_id, max_tokens=t.max_tokens, max_target_tokens=t.max_target_tokens)
            builder_args = {"run_seed": cfg.seed, "truncation": cfg.data.truncation, "k": t.k, "s_max": t.s_max}
            used, warm = 0, {"steps": 0, "scored": TokenCounts().to_json(), "gradient": TokenCounts().to_json()}
            before = {"scored": 0, "gradient": 0}  # the parent's tokens, which count toward the budget (D-010)
            if parent is not None:
                used, before = parent["tokens_used"], {"scored": parent["scored"], "gradient": parent["gradient"]}

            def save(label: str, row: dict[str, Any]) -> None:
                path = ckpt_dir / label
                model.save_pretrained(path)
                files = {p.name: sha256_file(p) for p in sorted(path.iterdir()) if p.is_file()}
                meta = {k: row.get(k) for k in ("step", "epoch", "tokens_used", "budget_fraction")}
                (path / "qet_checkpoint.json").write_text(json.dumps({**meta, "files": files}) + "\n", encoding="utf-8")
                checkpoints[_rel(path)] = files

            if t.warmup_steps:
                wcfg = TrainConfig(token_budget=t.token_budget, batch_graphs=t.batch_graphs, lr=t.lr, warmup=0.0,
                                   max_grad_norm=t.max_grad_norm, seed=cfg.seed, checkpoint_fractions=(),
                                   max_steps=t.warmup_steps)  # fmt: skip
                wopt = torch.optim.AdamW(scorer.parameters(), lr=t.lr)
                on_step = logger("warmup")
                if t.format_check:
                    check = _format_checker(cfg, ds, fmt, model, out, checks)
                    check(0)
                    log_step = on_step
                    on_step = lambda row: (log_step(row), check(row["step"]))  # noqa: E731
                warm = train(scorer, PackBuilder(fmt, ds.contracts, warmup_method(cfg), **builder_args),
                             ds.train_records, wcfg, wopt, on_step=on_step)  # fmt: skip
                used = warm["tokens_used"]
            if t.warmup_only:  # D-025d: the checkpoint that configuration runs name as their parent
                save("warmup", {"step": warm["steps"], "epoch": 0, "tokens_used": used,
                                "budget_fraction": used / t.token_budget})  # fmt: skip
                tokens.update(warmup={"steps": warm["steps"], "scored": warm["scored"], "gradient": warm["gradient"]},
                              scored=warm["scored"]["tokens"], gradient=warm["gradient"]["tokens"],
                              budget_used=used)  # fmt: skip
            else:
                tcfg = TrainConfig(token_budget=t.token_budget, batch_graphs=t.batch_graphs, lr=t.lr, warmup=t.warmup,
                                   max_grad_norm=t.max_grad_norm, seed=cfg.seed, plus=t.plus,
                                   checkpoint_fractions=t.checkpoint_fractions, tokens_at_start=used,
                                   max_minutes=t.max_minutes)  # fmt: skip
                opt = torch.optim.AdamW(scorer.parameters(), lr=t.lr)
                summary = train(scorer, PackBuilder(fmt, ds.contracts, m, **builder_args), ds.train_records, tcfg,
                                opt, on_step=logger("train"), on_checkpoint=save)  # fmt: skip
                tokens.update(warmup={"steps": warm["steps"], "scored": warm["scored"], "gradient": warm["gradient"]},
                              train={k: summary[k] for k in ("steps", "scored", "gradient", "diagnostic")},
                              parent=before if parent is not None else None,
                              scored=before["scored"] + warm["scored"]["tokens"] + summary["scored"]["tokens"],
                              gradient=before["gradient"] + warm["gradient"]["tokens"] + summary["gradient"]["tokens"],
                              budget_used=summary["tokens_used"])  # fmt: skip
    seconds["train"] = time.perf_counter() - clock
    if t is not None and t.warmup_only:  # no episodes: the format checks are the evaluation (D-010)
        below = next((c["step"] for c in checks if c["rate"] is not None and c["rate"] < 0.01), None)
        metrics = {"format_check": checks, "first_step_below_1_percent": below}
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        outputs = {"metrics": _rel(out / "metrics.json"), "train_log": _rel(out / "train_log.jsonl"),
                   "predictions": [], "checkpoints": checkpoints}  # fmt: skip
        return tokens, outputs, seconds

    clock = time.perf_counter()
    pred_dir = out / "predictions"
    pred_dir.mkdir()
    rows: list[dict[str, Any]] = []
    files: dict[str, Any] = {}

    def keep(row: dict[str, Any]) -> None:
        name = row["decoding"] if row["condition"] == "standard" else "resume_prefix"
        if name not in files:
            files[name] = open(pred_dir / f"{name}.jsonl", "w", encoding="utf-8")  # noqa: SIM115
        files[name].write(json.dumps(row, ensure_ascii=False) + "\n")
        rows.append(row)

    try:
        if vllm:
            from qet.infer.vllm_sampler import VLLMSampler, make_llm
            from qet.train.chat import backbone

            b = backbone(cfg.backbone)
            lora = None
            if evaluated is not None:
                lora = str(
                    Path(evaluated["path"]) if Path(evaluated["path"]).is_absolute() else REPO / evaluated["path"]
                )
            llm = make_llm(b.model_id, b.revision, lora=lora is not None, seed=cfg.seed)
            policy_for = lambda d: VLLMSampler(llm, fmt.eot_id, d, lora)  # noqa: E731
        else:
            policy_for = lambda d: HFSampler(model, fmt.eot_id, d)  # noqa: E731
        common = {"renderer": m.renderer, "truncation": cfg.data.truncation, "run_seed": cfg.seed, "on_episode": keep,
                  "width": cfg.eval.batch_episodes}  # fmt: skip
        if cfg.eval.standard:
            evaluate_tasks(ds.test_tasks, ds.make_env, policy_for, fmt, cfg.eval, **common)
        if cfg.eval.resume_prefix_tasks:
            settings = decoding_settings(cfg.eval)
            name, dcfg, _ = next(
                (d for d in settings if d[1].mode == "greedy"), settings[0]
            )  # greedy when there is one
            by_id = {task.task_id: task for task in ds.test_tasks}
            evaluate_resume(ds.test_records[: cfg.eval.resume_prefix_tasks], by_id, ds.make_env, policy_for(dcfg),
                            fmt, decoding=name, start_seed=cfg.eval.eval_seed,
                            start_mode=cfg.eval.resume_starts, history=cfg.eval.resume_history, **common)  # fmt: skip
    finally:
        for f in files.values():
            f.close()
    seconds["eval"] = time.perf_counter() - clock
    tokens["eval"] = {"episodes": len(rows),
                      "prompt": sum(s["prompt_tokens"] for r in rows for s in r["steps"]),
                      "output": sum(s["output_tokens"] for r in rows for s in r["steps"])}  # fmt: skip
    metrics = summarize(rows)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    outputs = {"metrics": _rel(out / "metrics.json"), "train_log": _rel(out / "train_log.jsonl"),
               "predictions": sorted(_rel(p) for p in pred_dir.iterdir()), "checkpoints": checkpoints}  # fmt: skip
    return tokens, outputs, seconds


def _format_checker(cfg, ds, fmt, model, out: Path, checks: list) -> Callable[[int], None]:
    """D-010: the format-error rate on `tasks` evaluation tasks (dev) spread evenly over the run's list (the dev
    manifest is ordered by grid cell), greedy, with the run's decoding limit, at the configured warm-up steps; each
    check is appended to `checks` and to format_check.jsonl."""
    from qet.eval.format_check import format_errors
    from qet.infer.decoding import DecodeConfig, HFSampler

    fc = cfg.train.format_check
    n, per, at = int(fc["tasks"]), int(fc.get("states_per_task", 2)), set(fc["at"])
    if n > len(ds.test_tasks):
        raise ValueError(f"format_check.tasks = {n}, but the run has {len(ds.test_tasks)} evaluation tasks")
    policy = HFSampler(model, fmt.eot_id, DecodeConfig("greedy", 1.0, 1.0, cfg.eval.max_new_tokens))
    picked = [i * len(ds.test_tasks) // n for i in range(n)]
    tasks, records = [ds.test_tasks[i] for i in picked], [ds.test_records[i] for i in picked]

    def check(step: int) -> None:
        if step not in at:
            return
        clock = time.perf_counter()
        res = format_errors(tasks, records, policy, fmt, ds.contracts,
                            renderer=cfg.method.renderer, truncation=cfg.data.truncation, states_per_task=per,
                            seed=cfg.eval.eval_seed, run_seed=cfg.seed)  # fmt: skip
        checks.append({"step": step, **res, "seconds": time.perf_counter() - clock,
                       "tasks": [t.task_id for t in tasks]})  # fmt: skip
        with open(out / "format_check.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(checks[-1]) + "\n")

    return check


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m qet.run", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    parser.add_argument("--config", required=True, action="append",
                        help="a config; several only with --dry-run, which then shares rendered prompts")  # fmt: skip
    parser.add_argument("--dry-run", action="store_true", help="print the plan and estimates; write nothing")
    parser.add_argument("--limit", type=int, help="at most N training graphs and N test tasks (smoke checks)")
    dirty_help = "development only: a dirty tree, recorded in the manifest; never a reported run"
    parser.add_argument("--allow-dirty", action="store_true", help=dirty_help)
    parser.add_argument("--runlog", type=Path, default=RUNLOG_PATH, help="the run log to append to (pinned jobs: the "
                        "main repository's docs/RUNLOG.md, D-025b)")  # fmt: skip
    args = parser.parse_args(argv)
    cfgs = [load_config(c) for c in args.config]
    if args.dry_run:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")  # read the cache only
        cache = DryRunCache()
        for i, cfg in enumerate(cfgs):
            print(("\n" if i else "") + format_plan(plan(cfg, limit=args.limit, cache=cache)), flush=True)
        return
    if len(cfgs) != 1:
        raise SystemExit("a run takes one config (several only with --dry-run)")
    print(run(cfgs[0], limit=args.limit, allow_dirty=args.allow_dirty, runlog_path=args.runlog))


if __name__ == "__main__":
    main()

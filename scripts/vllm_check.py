"""Validate and measure the vLLM evaluation backend (D-026b, D-030f) on dev tasks, in one GPU job:

    python scripts/vllm_check.py --compile-run results/raw/M1-compile-<stamp> [--tasks 64] [--throughput-tasks 224]
                                 [--adapter checkpoints/<warm-up run>/warmup] [--allow-dirty]

1. Agreement: greedy episodes on `--tasks` dev tasks (spread over the grid) with the Hugging Face loop and with vLLM,
   compared step by step (same raw output) and by outcome. With `--adapter`, both backends use that LoRA checkpoint.
2. Throughput: vLLM episodes on `--throughput-tasks` dev tasks with greedy decoding plus 8 samples at T = 1.0, as M1.7
   evaluates (EVAL_PLAN §7), timed for the evaluation budget.

Writes results/raw/M1-vllm-check-<stamp>/ (hf.jsonl, vllm.jsonl, throughput.jsonl, summary.json, manifest.json) and a
RUNLOG row. Runs in ~/venvs/qet-vllm (vLLM 0.11.0 over the base environment). Engineering numbers, not results.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> Path:
    import torch

    from qet import runlog
    from qet.config import DataConfig, EvalSettings, Split
    from qet.datasets import build_dataset
    from qet.eval.runner import evaluate_tasks
    from qet.infer.decoding import HFSampler
    from qet.infer.vllm_sampler import VLLMSampler, make_llm
    from qet.manifest import git_state, hardware, libraries, sha256_file, utc_now
    from qet.run import _rel, load_model
    from qet.train.chat import ChatFormat, backbone

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--compile-run", required=True)
    parser.add_argument("--backbone", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--tasks", type=int, default=64)
    parser.add_argument("--throughput-tasks", type=int, default=224)
    parser.add_argument("--adapter", help="a LoRA checkpoint folder both backends load")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--width", type=int, default=256, help="vLLM episodes per round")
    parser.add_argument("--out", type=Path, default=REPO / "results" / "raw")
    parser.add_argument("--runlog", type=Path, default=REPO / "docs" / "RUNLOG.md")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args(argv)
    git = git_state()
    if not args.allow_dirty and git["dirty"]:
        raise SystemExit(f"the check needs a clean tree (dirty: {git['dirty_files']})")
    data = DataConfig("funcbenchgen", "grid", {}, Split(0, 0), Split(0, 896),
                      source={"compile_run": args.compile_run, "train": "train", "eval": "dev"})  # fmt: skip
    ds = build_dataset(data)
    fmt = ChatFormat.load(args.backbone)
    spread = lambda n: [ds.test_tasks[i * len(ds.test_tasks) // n] for i in range(n)]  # noqa: E731
    agree_tasks, speed_tasks = spread(args.tasks), spread(args.throughput_tasks)
    adapter = None if args.adapter is None else str(Path(args.adapter).resolve())
    adapter_files = None
    if adapter:
        adapter_files = {p.name: sha256_file(p) for p in sorted(Path(adapter).iterdir()) if p.is_file()}

    created = utc_now()
    out = args.out / f"M1-vllm-check-{created:%Y%m%dT%H%M%SZ}"
    out.mkdir(parents=True)
    rel = _rel(out)
    row = {"run_id": out.name, "submitted": created.isoformat(), "milestone": "M1", "config": "scripts/vllm_check.py",
           "method": "vllm check", "backbone": args.backbone, "seed": "-", "status": "running", "GPU-h": "",
           "path": rel, "commit": git["commit"][:7] + ("+dirty" if git["dirty"] else ""),
           "note": f"tasks={args.tasks} throughput={args.throughput_tasks}"}  # fmt: skip
    runlog.append_row(args.runlog, row)
    t0, status, summary = time.perf_counter(), "failed", {}
    greedy = EvalSettings(({"mode": "greedy"},), args.max_new_tokens)
    full = EvalSettings(({"mode": "greedy"}, {"mode": "sample", "temperature": 1.0, "top_p": 1.0, "samples": 8}),
                        args.max_new_tokens)  # fmt: skip
    common = {"renderer": "quotient", "truncation": None, "run_seed": 0}

    def episodes(tasks, policy_for, settings, path, width=1):
        rows = []
        with open(path, "w", encoding="utf-8") as f:

            def keep(row):
                rows.append(row)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

            clock = time.perf_counter()
            evaluate_tasks(tasks, ds.make_env, policy_for, fmt, settings, on_episode=keep, width=width, **common)
        return rows, time.perf_counter() - clock

    def hf_phase():  # the Hugging Face model lives only here, so its memory is free for vLLM afterwards
        model = load_model(type("Cfg", (), {"backbone": args.backbone, "dtype": "bfloat16"})())
        if adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter).eval()
        return episodes(agree_tasks, lambda d: HFSampler(model, fmt.eot_id, d), greedy, out / "hf.jsonl")

    try:
        b = backbone(args.backbone)
        hf_rows, hf_s = hf_phase()
        gc.collect()  # the Hugging Face model is unreachable now; free its memory before vLLM reserves its share
        torch.cuda.empty_cache()
        llm = make_llm(b.model_id, b.revision, lora=adapter is not None, seed=0)
        vl_rows, vl_s = episodes(agree_tasks, lambda d: VLLMSampler(llm, fmt.eot_id, d, adapter), greedy,
                                 out / "vllm.jsonl", width=args.width)  # fmt: skip
        hf_steps = shared_steps = same_episodes = same_outcome = 0
        for h, v in zip(hf_rows, vl_rows, strict=True):
            a, c = [s["raw_output"] for s in h["steps"]], [s["raw_output"] for s in v["steps"]]
            prefix = next((i for i, (x, y) in enumerate(zip(a, c, strict=False)) if x != y), min(len(a), len(c)))
            hf_steps, shared_steps = hf_steps + len(a), shared_steps + prefix  # steps equal before the first difference
            same_episodes += a == c
            same_outcome += (h["end"], h["success"]) == (v["end"], v["success"])
        sp_rows, sp_s = episodes(speed_tasks, lambda d: VLLMSampler(llm, fmt.eot_id, d, adapter), full,
                                 out / "throughput.jsonl", width=args.width)  # fmt: skip
        steps = [s for r in sp_rows for s in r["steps"]]
        summary = {
            "agreement": {"episodes": len(hf_rows), "identical_episodes": same_episodes,
                          "same_end_and_success": same_outcome, "hf_steps": hf_steps,
                          "steps_before_first_difference": shared_steps, "hf_seconds": hf_s, "vllm_seconds": vl_s,
                          "hf_success": sum(r["success"] for r in hf_rows),
                          "vllm_success": sum(r["success"] for r in vl_rows)},
            "throughput": {"tasks": len(speed_tasks), "episodes": len(sp_rows), "steps": len(steps),
                           "prompt_tokens": sum(s["prompt_tokens"] for s in steps),
                           "output_tokens": sum(s["output_tokens"] for s in steps), "seconds": sp_s,
                           "width": args.width},
        }  # fmt: skip
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        status = "completed"
    except BaseException as e:
        status = f"failed: {type(e).__name__}: {e}"
        raise
    finally:
        hours = (time.perf_counter() - t0) / 3600 * (1 if torch.cuda.is_available() else 0)
        manifest = {"run_id": out.name, "created_at": created.isoformat(), "finished_at": utc_now().isoformat(),
                    "git": git, "script": "scripts/vllm_check.py", "args": {k: str(v) for k, v in vars(args).items()},
                    "backbone": {"id": args.backbone, "revision": backbone(args.backbone).revision},
                    "adapter": {"path": args.adapter, "files": adapter_files} if adapter else None,
                    "tokenizer": fmt.fingerprint(), "hardware": hardware(), "libraries": libraries(),
                    "gpu_hours": hours, "status": status,
                    "files": {p.name: sha256_file(p) for p in sorted(out.iterdir()) if p.is_file()}}  # fmt: skip
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        runlog.update_row(args.runlog, out.name, status="completed" if status == "completed" else "failed",
                          **{"GPU-h": f"{hours:.3f}"})  # fmt: skip
    print(out)
    return out


if __name__ == "__main__":
    sys.path.insert(0, str(REPO / "src"))
    main()

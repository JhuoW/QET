# Run log

One row per experiment run, appended by `qet.runlog` when a run is submitted and updated when it ends. Agents
may add a note (for example why a run was cancelled) but never type results here: results live in
`results/raw/<run_id>/` and tables come from `scripts/`.

| run_id | submitted | milestone | config | method | backbone | seed | commit | status | GPU-h | path | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1-smoke-sft-s0-20261017T074443Z | 2026-10-17T07:44:43+00:00 | E1 | configs/smoke/sft.yaml | 2 demo_order_sft | Qwen/Qwen2.5-0.5B-Instruct | 0 | c6f32b6 | completed | 0.156 | results/raw/E1-smoke-sft-s0-20261017T074443Z |  |
| E1-smoke-qet-s0-20261017T075405Z | 2026-10-17T07:54:05+00:00 | E1 | configs/smoke/qet.yaml | 6 qet | Qwen/Qwen2.5-0.5B-Instruct | 0 | c6f32b6 | completed | 0.141 | results/raw/E1-smoke-qet-s0-20261017T075405Z |  |
| M1-compile-20261017T161839Z | 2026-10-17T16:18:39+00:00 | M1 | scripts/compile_m1.py | compile | - | - | 2e428e3 | completed | 0 | results/raw/M1-compile-20261017T161839Z |  |
| M1-pilot-qet-s0-20261017T175029Z | 2026-10-17T17:50:29+00:00 | M1 | configs/m1/pilot-qet.yaml | 6 qet | Qwen/Qwen2.5-7B-Instruct | 0 | 3c8b23a | completed | 0.394 | results/raw/M1-pilot-qet-s0-20261017T175029Z |  |
| M1-pilot-stats-20261017T190141Z | 2026-10-17T19:01:41+00:00 | M1 | scripts/m1_pilot_stats.py | pilot statistics | Qwen/Qwen2.5-7B-Instruct | - | 195883c | completed | 0 | results/raw/M1-pilot-stats-20261017T190141Z | graphs=500 |
| M1-pilot-stats-20261017T172519Z | 2026-10-17T17:25:19+00:00 | M1 | scripts/m1_pilot_stats.py | pilot statistics | Qwen/Qwen2.5-7B-Instruct | - | e627e8e | completed | 0 | results/raw/M1-pilot-stats-20261017T172519Z | row added after the run (the script then wrote none); superseded by M1-pilot-stats-20261017T190141Z, byte-identical stats.json and graphs.jsonl |

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
| M1-compile-20261018T023058Z | 2026-10-18T02:30:58+00:00 | M1 | scripts/compile_m1.py | compile | - | - | aca1b0b | completed | 0 | results/raw/M1-compile-20261018T023058Z |  |
| M1-warmup-measure-qwen7b-s0-20261018T032752Z | 2026-10-18T03:27:52+00:00 | M1 | configs/m1/warmup-measure.yaml | 2 demo_order_sft | Qwen/Qwen2.5-7B-Instruct | 0 | f506d83 | completed | 0.529 | results/raw/M1-warmup-measure-qwen7b-s0-20261018T032752Z |  |
| M1-warmup-quotient-qwen7b-s1-20261018T040849Z | 2026-10-18T04:08:49+00:00 | M1 | configs/m1/warmup-quotient-qwen7b-s1.yaml | 2 demo_order_sft | Qwen/Qwen2.5-7B-Instruct | 1 | 22753ab | completed | 0.084 | results/raw/M1-warmup-quotient-qwen7b-s1-20261018T040849Z |  |
| M1-warmup-quotient-qwen7b-s0-20261018T040850Z | 2026-10-18T04:08:50+00:00 | M1 | configs/m1/warmup-quotient-qwen7b-s0.yaml | 2 demo_order_sft | Qwen/Qwen2.5-7B-Instruct | 0 | 22753ab | completed | 0.088 | results/raw/M1-warmup-quotient-qwen7b-s0-20261018T040850Z |  |
| M1-warmup-quotient-llama8b-s2-20261018T040850Z | 2026-10-18T04:08:50+00:00 | M1 | configs/m1/warmup-quotient-llama8b-s2.yaml | 2 demo_order_sft | meta-llama/Llama-3.1-8B-Instruct | 2 | 22753ab | failed | 0.003 | results/raw/M1-warmup-quotient-llama8b-s2-20261018T040850Z | AttributeError |
| M1-warmup-chrono-qwen7b-s0-20261018T040850Z | 2026-10-18T04:08:50+00:00 | M1 | configs/m1/warmup-chrono-qwen7b-s0.yaml | 1 chronological_sft | Qwen/Qwen2.5-7B-Instruct | 0 | 22753ab | completed | 0.085 | results/raw/M1-warmup-chrono-qwen7b-s0-20261018T040850Z |  |
| M1-warmup-chrono-qwen7b-s1-20261018T040850Z | 2026-10-18T04:08:50+00:00 | M1 | configs/m1/warmup-chrono-qwen7b-s1.yaml | 1 chronological_sft | Qwen/Qwen2.5-7B-Instruct | 1 | 22753ab | completed | 0.088 | results/raw/M1-warmup-chrono-qwen7b-s1-20261018T040850Z |  |
| M1-warmup-chrono-qwen7b-s2-20261018T040850Z | 2026-10-18T04:08:50+00:00 | M1 | configs/m1/warmup-chrono-qwen7b-s2.yaml | 1 chronological_sft | Qwen/Qwen2.5-7B-Instruct | 2 | 22753ab | completed | 0.090 | results/raw/M1-warmup-chrono-qwen7b-s2-20261018T040850Z |  |
| M1-warmup-native-qwen7b-s0-20261018T040852Z | 2026-10-18T04:08:52+00:00 | M1 | configs/m1/warmup-native-qwen7b-s0.yaml | a1 native_multiturn_sft | Qwen/Qwen2.5-7B-Instruct | 0 | 22753ab | completed | 0.084 | results/raw/M1-warmup-native-qwen7b-s0-20261018T040852Z |  |
| M1-warmup-quotient-qwen7b-s2-20261018T040853Z | 2026-10-18T04:08:53+00:00 | M1 | configs/m1/warmup-quotient-qwen7b-s2.yaml | 2 demo_order_sft | Qwen/Qwen2.5-7B-Instruct | 2 | 22753ab | completed | 0.085 | results/raw/M1-warmup-quotient-qwen7b-s2-20261018T040853Z |  |
| M1-warmup-quotient-llama8b-s0-20261018T040853Z | 2026-10-18T04:08:53+00:00 | M1 | configs/m1/warmup-quotient-llama8b-s0.yaml | 2 demo_order_sft | meta-llama/Llama-3.1-8B-Instruct | 0 | 22753ab | failed | 0.002 | results/raw/M1-warmup-quotient-llama8b-s0-20261018T040853Z | AttributeError |
| M1-warmup-quotient-llama8b-s1-20261018T040854Z | 2026-10-18T04:08:54+00:00 | M1 | configs/m1/warmup-quotient-llama8b-s1.yaml | 2 demo_order_sft | meta-llama/Llama-3.1-8B-Instruct | 1 | 22753ab | failed | 0.002 | results/raw/M1-warmup-quotient-llama8b-s1-20261018T040854Z | AttributeError |
| M1-vllm-check-20261018T040903Z | 2026-10-18T04:09:03+00:00 | M1 | scripts/vllm_check.py | vllm check | Qwen/Qwen2.5-7B-Instruct | - | 22753ab | failed | 0.212 | results/raw/M1-vllm-check-20261018T040903Z | tasks=64 throughput=224 |

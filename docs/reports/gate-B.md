# Gate B report (M1: FuncBenchGen)

From 51 completed evaluation runs (vLLM 0.11, D-026, D-032), recomputed from their saved predictions; the test widths come from the frozen compile run. vLLM's greedy decoding is not repeatable run to run (D-032d): each number here is one evaluation, read from its saved predictions.

Not run (D-032e: the evaluation was reduced to the pre-registered criteria before any result of configurations 1, 2, 4 or 5 existed, after the cost projection passed the D-032 guard; results of 3 and 6 existed, and no pre-registered criterion or comparison changed): configurations 1, 2, 4 and 5 were evaluated greedy only (5 also with Resume@prefix), without samples, budget curves or the renamed and held-out sets, and configuration 2 without ToolHop; the no-history condition, every Llama run, the ablations a1-a4, the base models and the D-033 extras were not evaluated. A dash in a table marks an evaluation not run.

Every comparison in section 1 is fresh and causal in design: the same pipeline, data, budget B and evaluation, differing only in the configuration. Criterion 3 compares per-task gaps (a difference of two success rates), which has no paired binary outcome, so no McNemar test is given for it.

## 1. Pre-registered criteria (EVAL_PLAN §9, verbatim)

## 9. Falsification criteria (pre-registered at Gate A, 2026-10-18; frozen)
Common terms: FuncBenchGen test graphs; every run evaluated at its checkpoint at 100% of B, so the compared runs have
spent the same compute (up to the last step's overshoot of B, D-023f) and no checkpoint is selected; success averaged
over the three seeds of each task; differences with paired-bootstrap 95% intervals (10,000 resamples, clustered by
task); "significant" means the interval excludes 0. McNemar per seed is reported alongside.
1. Greedy success on graphs of measured width ≥ 3: if the interval of (6 − 3) or of (6 − 4) contains 0 or lies below
   it, the objective claim fails.
2. Greedy success on all test graphs: if (2 − 1) is significant and positive but the interval of (6 − 2) contains 0
   or lies below it, the gain belongs to the interface, not the objective.
3. Resume@prefix greedy success on the test tasks that have off-path starts, with the off-path gap = on-path minus
   off-path success at the frozen starts: if the interval of (gap 3 − gap 6) or of (gap 5 − gap 6) contains 0 or lies
   below it, claim C4 fails.
4. If QET does not beat re-scored ToolLLaMA on StableToolBench, the published comparison fails; parity with
   DTA-Llama is acceptable (QET does not target parallel latency).
5. If the AppWorld audit finds most graphs collapse to chains, QET is reported as inapplicable there; compile
   rules are never tuned on test data.

| criterion | comparison | estimate [95% interval] | significant | McNemar p per seed |
| --- | --- | --- | --- | --- |
| 1 | 6 - 3 | -0.068 [-0.080, -0.055] | yes | s0: 4.11e-18, s1: 8.14e-08, s2: 1.08e-08 |
| 1 | 6 - 4 | +0.002 [-0.007, +0.011] | no | s0: 0.195, s1: 0.309, s2: 0.733 |
| 2 | 2 - 1 | +0.003 [-0.009, +0.014] | no | s0: 0.629, s1: 0.444, s2: 0.658 |
| 2 | 6 - 2 | -0.414 [-0.429, -0.398] | yes | s0: 1.94e-230, s1: 1.54e-258, s2: 3.28e-244 |
| 3 | gap 3 - gap 6 | +0.000 [-0.008, +0.008] | no | – |
| 3 | gap 5 - gap 6 | +0.004 [-0.002, +0.010] | no | – |

- Criterion 1: the objective claim fails (pool of 1397 tasks of width >= 3).
- Criterion 2: the condition is not met.
- Criterion 3: claim C4 fails.
- Criteria 4-5: M2 (StableToolBench, AppWorld); not evaluated at this gate.

## 2. Success on the frozen test set (checkpoint at 100% of B; base models untrained)

| backbone | configuration | seeds | greedy success [95% interval] | sampled success | pass^4 | tasks |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-7B-Instruct | 1 | 3 | 0.617 [0.601, 0.633] | – | – | 2240 |
| Qwen2.5-7B-Instruct | 2 | 3 | 0.620 [0.603, 0.636] | – | – | 2240 |
| Qwen2.5-7B-Instruct | 3 | 3 | 0.298 [0.282, 0.314] | 0.191 | 0.069 | 2240 |
| Qwen2.5-7B-Instruct | 4 | 3 | 0.204 [0.190, 0.218] | – | – | 2240 |
| Qwen2.5-7B-Instruct | 5 | 3 | 0.201 [0.187, 0.216] | – | – | 2240 |
| Qwen2.5-7B-Instruct | 6 | 3 | 0.206 [0.192, 0.220] | 0.133 | 0.051 | 2240 |

a4 runs (K = 2, 4, 8) share configuration a4 and seed 0, so their rows are listed in section 7.

## 3. Greedy success by measured width (Qwen, seeds averaged)

| configuration | width 1 | width 2 | width 3-4 | width 5+ |
| --- | --- | --- | --- | --- |
| 1 | 0.583 (n=482) | 0.797 (n=361) | 0.698 (n=643) | 0.483 (n=754) |
| 2 | 0.583 (n=482) | 0.806 (n=361) | 0.686 (n=643) | 0.497 (n=754) |
| 3 | 0.333 (n=482) | 0.421 (n=361) | 0.332 (n=643) | 0.187 (n=754) |
| 4 | 0.195 (n=482) | 0.289 (n=361) | 0.252 (n=643) | 0.126 (n=754) |
| 5 | 0.200 (n=482) | 0.280 (n=361) | 0.252 (n=643) | 0.122 (n=754) |
| 6 | 0.205 (n=482) | 0.283 (n=361) | 0.259 (n=643) | 0.124 (n=754) |

## 4. Resume@prefix (greedy, frozen starts at every depth; Qwen, seeds averaged)

| configuration | empty | on-path | off-path | off-path gap | tasks with off-path starts |
| --- | --- | --- | --- | --- | --- |
| 3 | 0.295 | 0.320 | 0.290 | +0.030 [+0.022, +0.039] | 1758 |
| 5 | 0.203 | 0.186 | 0.155 | +0.034 [+0.027, +0.041] | 1758 |
| 6 | 0.207 | 0.194 | 0.167 | +0.030 [+0.023, +0.037] | 1758 |

## 5. Greedy success along the budget (curves; Qwen and Llama, seeds averaged)

| backbone | configuration | 25% | 50% | 75% | 100% |
| --- | --- | --- | --- | --- | --- |
| Qwen2.5-7B-Instruct | 1 | – | – | – | 0.617 |
| Qwen2.5-7B-Instruct | 2 | – | – | – | 0.620 |
| Qwen2.5-7B-Instruct | 3 | 0.125 | 0.212 | 0.269 | 0.298 |
| Qwen2.5-7B-Instruct | 4 | – | – | – | 0.204 |
| Qwen2.5-7B-Instruct | 5 | – | – | – | 0.201 |
| Qwen2.5-7B-Instruct | 6 | 0.078 | 0.149 | 0.203 | 0.206 |

## 6. Renamed and held-out larger sets (greedy; Qwen, seeds averaged)

| configuration | renamed | held-out larger |
| --- | --- | --- |
| 3 | 0.306 | 0.031 |
| 6 | 0.205 | 0.000 |

## 7. Ablations, ToolHop and the extras

| run | split | seeds | greedy success | sampled success |
| --- | --- | --- | --- | --- |
| main configuration 6 | toolhop | 3 | 0.184 | 0.161 |

## 8. Training exposure at matched compute

Every run spends the same budget B of scored + gradient tokens (EVAL_PLAN §8), so a configuration that scores more states per graph trains on fewer distinct graphs. Read from the training run of each evaluated checkpoint: tokens used against B (the last step overshoots it, D-023f) from its manifest, graphs from its train log.

| backbone | run | seeds | tokens used / B | largest overshoot | distinct graphs trained on | training graphs | passes over the training graphs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-7B-Instruct | main configuration 1 | 3 | 1.0003-1.0006 | 0.06% | 4,712-4,720 | 5,000 | 0.94-0.94 |
| Qwen2.5-7B-Instruct | main configuration 2 | 3 | 1.0001-1.0006 | 0.06% | 4,792-4,800 | 5,000 | 0.96-0.96 |
| Qwen2.5-7B-Instruct | main configuration 3 | 3 | 1.0006-1.0047 | 0.47% | 908-956 | 5,000 | 0.18-0.19 |
| Qwen2.5-7B-Instruct | main configuration 4 | 3 | 1.0007-1.0062 | 0.62% | 440-472 | 5,000 | 0.09-0.09 |
| Qwen2.5-7B-Instruct | main configuration 5 | 3 | 1.0007-1.0062 | 0.62% | 440-472 | 5,000 | 0.09-0.09 |
| Qwen2.5-7B-Instruct | main configuration 6 | 3 | 1.0007-1.0062 | 0.62% | 440-472 | 5,000 | 0.09-0.09 |

## 9. Decision for the human

Whether M2 runs (MILESTONES GATE B). **STOP.** The agent starts no M2 work before the human decides.

## Sources (sha256)

- `results/raw/M1-compile-20261018T023058Z/manifest.json` 78d2f69f81a3937806adf48340cbe13c13f90f0fdf1705ec8b2996212b93f095
- `results/raw/M1-compile-20261018T023058Z/test/records.jsonl` 613328b0deb0842cd73250aaddfe6ad4c3708de3036373d2977a419aef4409d6
- `results/raw/M1-eval-c1-qwen7b-crit-s0-20261019T234939Z/manifest.json` 58459c355eb0df4758565257b6942933df66326e4ad0f5e193da92b5ee458296
- `results/raw/M1-eval-c1-qwen7b-crit-s0-20261019T234939Z/predictions/greedy.jsonl` bee906142215c981c74246e9afdbd1a1657e097862044f49b53ccc3cb1867f40
- `results/raw/M1-eval-c1-qwen7b-crit-s1-20261020T000221Z/manifest.json` 97f688cd7ee969e1b93b50552e87e4b04308e1216fe210d1b73fb5c8ea80708e
- `results/raw/M1-eval-c1-qwen7b-crit-s1-20261020T000221Z/predictions/greedy.jsonl` cc3d07e4d29679965f69207e77d8f2c7e2009ba0b0b74470b265747b4c9218fe
- `results/raw/M1-eval-c1-qwen7b-crit-s2-20261020T001437Z/manifest.json` 76f61e1941085f1a64a7f71c08385698b8d8ce7bf718f1f61a5a6ac0f627eb3d
- `results/raw/M1-eval-c1-qwen7b-crit-s2-20261020T001437Z/predictions/greedy.jsonl` d4585a6988bfc9be794787322e00213b06abe7d34f2fd2f50412acafce904bd8
- `results/raw/M1-eval-c2-qwen7b-crit-s0-20261020T002535Z/manifest.json` 5c26b11b204b5251b4093683052b9c0861a8882ec999b89c4613f8e56174d8a6
- `results/raw/M1-eval-c2-qwen7b-crit-s0-20261020T002535Z/predictions/greedy.jsonl` ada8a77bf8308169551460c575cd14f869741c3de6e08735bdedf46a3ff2fa7d
- `results/raw/M1-eval-c2-qwen7b-crit-s1-20261020T003938Z/manifest.json` 336e472d3ba7e0a5b0edab1ffe57ecee6446fd0ab64509df66691f7b5a6c6115
- `results/raw/M1-eval-c2-qwen7b-crit-s1-20261020T003938Z/predictions/greedy.jsonl` 990ad9d43d3b6e4a09a94df10d61864bce4942dbfe8fff01de0cc2df10292729
- `results/raw/M1-eval-c2-qwen7b-crit-s2-20261020T005328Z/manifest.json` 822b17b0510ad6af495b31888306269e16c64c46bf6340e6a704ee61746995ea
- `results/raw/M1-eval-c2-qwen7b-crit-s2-20261020T005328Z/predictions/greedy.jsonl` 4fc03f0cea25f1e5a0d31d955d4e1b45f96151cb3152d3709cc28eeb6004500e
- `results/raw/M1-eval-c3-qwen7b-frac0.25-s0-20261019T113716Z/manifest.json` 8f84c29c4bf15dbe5a4805e62323747270072e663df6ef32f4505269dd8c60a1
- `results/raw/M1-eval-c3-qwen7b-frac0.25-s0-20261019T113716Z/predictions/greedy.jsonl` 0aa18887ed382f28381e80a7b5ce1da6253854fcb1d8ac43c8cd8c5806786347
- `results/raw/M1-eval-c3-qwen7b-frac0.25-s1-20261019T174634Z/manifest.json` 434aff79e8bb7efe452f592d1ff843a7aaf30d48a9fd937b6bd7f97055301454
- `results/raw/M1-eval-c3-qwen7b-frac0.25-s1-20261019T174634Z/predictions/greedy.jsonl` a263669fe5b2ebce7d775a4077ba753e4602641358767e80f0b1201ad320ca86
- `results/raw/M1-eval-c3-qwen7b-frac0.25-s2-20261019T154751Z/manifest.json` 71ca50a1476fb6b97173b0854266783e57779948fb346dd6d1efbd61c962f297
- `results/raw/M1-eval-c3-qwen7b-frac0.25-s2-20261019T154751Z/predictions/greedy.jsonl` 04f2f0ff0f99b77dcdea49615a5bf219adde3b5446a7ef25c3d1e150b6b715ae
- `results/raw/M1-eval-c3-qwen7b-frac0.5-s0-20261019T115145Z/manifest.json` 80a1e0ff8bb0c6afc61099d17283b1e207dba4efebb9901c2d40574086722656
- `results/raw/M1-eval-c3-qwen7b-frac0.5-s0-20261019T115145Z/predictions/greedy.jsonl` e6e65d7038fc6e3c400f938daf27cd879b8408035ec2d32d9aa179e1d827d9df
- `results/raw/M1-eval-c3-qwen7b-frac0.5-s1-20261019T175905Z/manifest.json` 84fd806695b8ca5abe314de1c96af2655268d2258b9d5491985374b00e4a1052
- `results/raw/M1-eval-c3-qwen7b-frac0.5-s1-20261019T175905Z/predictions/greedy.jsonl` c612ebd86ab6fd398f196fe2e57de4616dd9276c86c602ea4098594a692247a3
- `results/raw/M1-eval-c3-qwen7b-frac0.5-s2-20261019T160102Z/manifest.json` 474ff620fed0e9cfda716e7e53aa8add16303f45ee74285194f1ec575531846a
- `results/raw/M1-eval-c3-qwen7b-frac0.5-s2-20261019T160102Z/predictions/greedy.jsonl` 0fcef6d95510d16775b56d24abdbf15a15c38844b92b2f59dd5f585fc140872d
- `results/raw/M1-eval-c3-qwen7b-frac0.75-s0-20261019T120655Z/manifest.json` 9d0451885825cf0ed955c57b31ef16cec833c77851ce7ff626a28f180778e6e0
- `results/raw/M1-eval-c3-qwen7b-frac0.75-s0-20261019T120655Z/predictions/greedy.jsonl` 22bd967486f6c205e67d46905f1571d3a5df47cbfa26b04e04dbd0e60342dd18
- `results/raw/M1-eval-c3-qwen7b-frac0.75-s1-20261019T181258Z/manifest.json` 21558a21386a99beb3d79251ea49b2ba182054ea3462980eee1b94f090ade460
- `results/raw/M1-eval-c3-qwen7b-frac0.75-s1-20261019T181258Z/predictions/greedy.jsonl` f0b01de510a398dfcce67dafd17ef3901040dca29a49c7fba9315c24e6b0a076
- `results/raw/M1-eval-c3-qwen7b-frac0.75-s2-20261019T161417Z/manifest.json` ba0507918e26064d2c03a0a780ea89638e6de50a76a5488cb2893adc7ed0353a
- `results/raw/M1-eval-c3-qwen7b-frac0.75-s2-20261019T161417Z/predictions/greedy.jsonl` 7a3a33e36dc482136e1f25374b71bc5f19ca36fa1d3c7c4d0e2b413396084177
- `results/raw/M1-eval-c3-qwen7b-full-s0-20261019T081145Z/manifest.json` 901537db7b83f7bfa4b42da448a2223174eef404c114305d40b3e89638af0017
- `results/raw/M1-eval-c3-qwen7b-full-s0-20261019T081145Z/predictions/greedy.jsonl` f84da005750ac4d6159f7f28ee35348b74b6b16476000b451236fe231f18b313
- `results/raw/M1-eval-c3-qwen7b-full-s0-20261019T081145Z/predictions/resume_prefix.jsonl` ef131013f7762099b84e81c97b89cc1eaae080a4f89750a390e98605bd1a4661
- `results/raw/M1-eval-c3-qwen7b-full-s0-20261019T081145Z/predictions/sample-t1-p1.jsonl` f24788cac174fa5a2151bc82a96619c5f404f673804d8c23849d1c2f057065d6
- `results/raw/M1-eval-c3-qwen7b-full-s1-20261019T142012Z/manifest.json` 15e2e79d713e1094c467abab3f32527368abd852f5baa156db87eb909187a256
- `results/raw/M1-eval-c3-qwen7b-full-s1-20261019T142012Z/predictions/greedy.jsonl` 1a4c3e64c6a9eddacff8ff198730ce70509b003afc286fb084609905f96e95bd
- `results/raw/M1-eval-c3-qwen7b-full-s1-20261019T142012Z/predictions/resume_prefix.jsonl` bff47b2d2a3029e4eea95fd321a5f7613106561af6fb173c22c7285bcd5afcfe
- `results/raw/M1-eval-c3-qwen7b-full-s1-20261019T142012Z/predictions/sample-t1-p1.jsonl` b6905092f0f72fcd9bc9aa13b59f8162e581784c8e7a10f3861c574ebf40819e
- `results/raw/M1-eval-c3-qwen7b-full-s2-20261019T124109Z/manifest.json` f31f1f2bd957e25f79c0c02f7316ca7a6416f6f756e4d99d2db20ba5768efefe
- `results/raw/M1-eval-c3-qwen7b-full-s2-20261019T124109Z/predictions/greedy.jsonl` 227996c7d41a6cf005f3a309ecadad9d6ccb0e163bc2c39b5e0ab40534115289
- `results/raw/M1-eval-c3-qwen7b-full-s2-20261019T124109Z/predictions/resume_prefix.jsonl` fbaa08b8577b23d701eed6538fac7f59f595a2f3b04cb9b198b19d47e6e3e857
- `results/raw/M1-eval-c3-qwen7b-full-s2-20261019T124109Z/predictions/sample-t1-p1.jsonl` 75b15095606694d2a7a61e3d6ba3d20a4382c9d06b7de5ac0c04476d26a03e9d
- `results/raw/M1-eval-c3-qwen7b-heldout-s0-20261019T123432Z/manifest.json` 6587f9a60e82fbe52bf882d001a718327d7927e33d6028faa8d684b5149f0007
- `results/raw/M1-eval-c3-qwen7b-heldout-s0-20261019T123432Z/predictions/greedy.jsonl` 2844878a2f2bbf81956145362e72a85371e3d4be55f612fefea7a7b09e8511bb
- `results/raw/M1-eval-c3-qwen7b-heldout-s1-20261019T184212Z/manifest.json` b89fecf5fc490efd12d237788f7fcb13bc8d98b2374d82c07c1afb7b33e6cbba
- `results/raw/M1-eval-c3-qwen7b-heldout-s1-20261019T184212Z/predictions/greedy.jsonl` db8063b411c51cc66ed9cd1e05da01fd9bc68f415483c246b0ea4439540fecfb
- `results/raw/M1-eval-c3-qwen7b-heldout-s2-20261019T164100Z/manifest.json` 72dfaefb4d2945f5d3d6ef5eeb7133b1723ff2eeaec83fe989939b57eb07b0bd
- `results/raw/M1-eval-c3-qwen7b-heldout-s2-20261019T164100Z/predictions/greedy.jsonl` 08e9b74ced4a2c7259c267780c5c86d783aa622d83507b0529b0f070b43d281f
- `results/raw/M1-eval-c3-qwen7b-renamed-s0-20261019T122027Z/manifest.json` 152e5601d377ac995475c51a2d0be7184cf0b48d40d3e6db7b3f8db2e3a4434d
- `results/raw/M1-eval-c3-qwen7b-renamed-s0-20261019T122027Z/predictions/greedy.jsonl` c1a5db426b6c544cdd67b5a151a7a5ee07de1af6447683704a9dba1c9aa205ee
- `results/raw/M1-eval-c3-qwen7b-renamed-s1-20261019T182745Z/manifest.json` a62ce3a6b8ac559f7fb07fd19cafb0df3b20d2b8b7076662dbaba126f28bd872
- `results/raw/M1-eval-c3-qwen7b-renamed-s1-20261019T182745Z/predictions/greedy.jsonl` 363f08ed686a2b69bdb2daf56a3287f3ebdbc640a65a69b68e18dbda3b1c757b
- `results/raw/M1-eval-c3-qwen7b-renamed-s2-20261019T162732Z/manifest.json` 012fca856f0dd16dd0461788687a121fb089b8bf30a5c98377aa36b144164dd6
- `results/raw/M1-eval-c3-qwen7b-renamed-s2-20261019T162732Z/predictions/greedy.jsonl` 6fb05140b617f14cba01f213100db161c184487a6cc183bce761dac12ae6c5b6
- `results/raw/M1-eval-c4-qwen7b-crit-s0-20261019T230522Z/manifest.json` 1373a684bf893d1c14d2a8d74671c411c7ea58fafc1307dfe828ff1fafb0e38e
- `results/raw/M1-eval-c4-qwen7b-crit-s0-20261019T230522Z/predictions/greedy.jsonl` 80c9b0dc53118a35b796e149f1f308d2d12a33b8d56d9d11461311192a46f700
- `results/raw/M1-eval-c4-qwen7b-crit-s1-20261019T232022Z/manifest.json` d87dce751619e7630815b572d842650839d9d7f30e09c6b9a6b9b61a9f7a39ad
- `results/raw/M1-eval-c4-qwen7b-crit-s1-20261019T232022Z/predictions/greedy.jsonl` 945ed6a7e135c827efcaae9b5044492571817983998a52d6744d479d95389102
- `results/raw/M1-eval-c4-qwen7b-crit-s2-20261019T233412Z/manifest.json` 31972b85d957d115f90a9d397e2bf96a9acad131b0c6640d1e9b4f2b1435e31c
- `results/raw/M1-eval-c4-qwen7b-crit-s2-20261019T233412Z/predictions/greedy.jsonl` e9cd948020e57b309acd4de898c1c50f446075f3de659e7c893e4739447e9058
- `results/raw/M1-eval-c5-qwen7b-crit-s0-20261019T230534Z/manifest.json` 8593f9464c0296e6f2685407206fa2d9cd338df1d51eee1d6b93570cee96c42f
- `results/raw/M1-eval-c5-qwen7b-crit-s0-20261019T230534Z/predictions/greedy.jsonl` ddd49e52c443a841465185782d324e2fe48bd09abc9c699a82c830cb78e46ac9
- `results/raw/M1-eval-c5-qwen7b-crit-s0-20261019T230534Z/predictions/resume_prefix.jsonl` 51b706483e75e8c8812a8a2efeb3c75399a67e9f109a74510150fc6c0f9a156c
- `results/raw/M1-eval-c5-qwen7b-crit-s1-20261020T012837Z/manifest.json` f526374f931cd3db4d20d18b95e283d30f1205c801389ba58ad2db34f8e4f4c3
- `results/raw/M1-eval-c5-qwen7b-crit-s1-20261020T012837Z/predictions/greedy.jsonl` fd604fb7f1c7bc878b0a23f6d282dc526b1e756eac7808f37d45211fdea0bd39
- `results/raw/M1-eval-c5-qwen7b-crit-s1-20261020T012837Z/predictions/resume_prefix.jsonl` 9f1727909f7f67d04d65fbbd4f1a87c4edb875f43ccaccd863dad4f25309eb64
- `results/raw/M1-eval-c5-qwen7b-crit-s2-20261020T010735Z/manifest.json` a7a6591a23647b965f027df86bfdf4cc39ce1de5b81288fdbb15691e591217fc
- `results/raw/M1-eval-c5-qwen7b-crit-s2-20261020T010735Z/predictions/greedy.jsonl` 586be1a11d2ea7aee47b474542537abee3b5548b6f7dcb18af5dc651988cb33c
- `results/raw/M1-eval-c5-qwen7b-crit-s2-20261020T010735Z/predictions/resume_prefix.jsonl` 60d53a621bf77b95baa81e56b57748cc92d99ae52311c45d7dac995733cdd17a
- `results/raw/M1-eval-c6-qwen7b-frac0.25-s0-20261019T070714Z/manifest.json` 80791e429b6af5e97f9f20222ae41373433b2949c3ebb5b0a673914ce45b3238
- `results/raw/M1-eval-c6-qwen7b-frac0.25-s0-20261019T070714Z/predictions/greedy.jsonl` 28455d428885094c4cfa6fd4aa34eb5a966187d9e189a777205de6c2af1d4969
- `results/raw/M1-eval-c6-qwen7b-frac0.25-s1-20261019T064115Z/manifest.json` aaa1cd8237540376e344ed08a666ae3e5345d8e52986d4eb0bbf3045abe69a0e
- `results/raw/M1-eval-c6-qwen7b-frac0.25-s1-20261019T064115Z/predictions/greedy.jsonl` 05888d272b2d24f0bbb0c641d1cfb35a60713f8b55527801ec3a61e16a816d2d
- `results/raw/M1-eval-c6-qwen7b-frac0.25-s2-20261019T124427Z/manifest.json` 135e5a09ef09708284cd4d72d2135ecd30636d02220c0c3a783cd5a0340591e0
- `results/raw/M1-eval-c6-qwen7b-frac0.25-s2-20261019T124427Z/predictions/greedy.jsonl` e04bfda72baa05ad10ea7c3ffd99d4357222bf6eec27e898a3538c81e2045294
- `results/raw/M1-eval-c6-qwen7b-frac0.5-s0-20261019T072322Z/manifest.json` df7d7212bc868af4a1d045124f3fe6c0c416d9500a37f76f290b69fc774a40e4
- `results/raw/M1-eval-c6-qwen7b-frac0.5-s0-20261019T072322Z/predictions/greedy.jsonl` bdbd1a1cf5e60d3f090edb3dbbc40c6e147bd31e82245215bfbe529ce37866bc
- `results/raw/M1-eval-c6-qwen7b-frac0.5-s1-20261019T065748Z/manifest.json` 7a5490deeacfa6014e21817e81f92a503f0d4f75324201afbb6c2c90f157f463
- `results/raw/M1-eval-c6-qwen7b-frac0.5-s1-20261019T065748Z/predictions/greedy.jsonl` db50b75e426c941b793a87210e631221a48020c4c6d3059784d81e7832e4ebf4
- `results/raw/M1-eval-c6-qwen7b-frac0.5-s2-20261019T130103Z/manifest.json` 4ba7130458823fe1ad45a96291217631869ccbdc10615099d2b18f5215e19b27
- `results/raw/M1-eval-c6-qwen7b-frac0.5-s2-20261019T130103Z/predictions/greedy.jsonl` 24db29eee80aabe9a85fbbb541d984640ded9510c8e5d966fac0b1a89730f6cb
- `results/raw/M1-eval-c6-qwen7b-frac0.75-s0-20261019T073825Z/manifest.json` 834092cdf53039f5532087fafcab31981401021c8d241bb65091df6c4a8396fb
- `results/raw/M1-eval-c6-qwen7b-frac0.75-s0-20261019T073825Z/predictions/greedy.jsonl` 4dc7cd8690c6a0283d5d4eb31b38ee0cac6b9c55c5ebfea6fad44f053c14f79e
- `results/raw/M1-eval-c6-qwen7b-frac0.75-s1-20261019T071356Z/manifest.json` d681b3abd6b665c8c153353fd963076fdb2470aa942732541062047728173ea2
- `results/raw/M1-eval-c6-qwen7b-frac0.75-s1-20261019T071356Z/predictions/greedy.jsonl` e415fc3a7864b209cf5f3addb1541ce546380fecd9039b436c854786c82b1a80
- `results/raw/M1-eval-c6-qwen7b-frac0.75-s2-20261019T131625Z/manifest.json` 314e864f35514fa472f95031ae230b0d0a32b68251790869fd75b06cfa9c9145
- `results/raw/M1-eval-c6-qwen7b-frac0.75-s2-20261019T131625Z/predictions/greedy.jsonl` bafb76e45834c18c1b490eccd50eae42dc46d961b7d7717f61cec7ae888dd684
- `results/raw/M1-eval-c6-qwen7b-full-s0-20261019T033011Z/manifest.json` bccdf0b5cd5e8f1f8f0b7e068bdc3883b1e2e47549942227ca1cf0cd49122d23
- `results/raw/M1-eval-c6-qwen7b-full-s0-20261019T033011Z/predictions/greedy.jsonl` 651cccc8d1cdb853e287f4e198097aa330dc4de04610c306fc36b59fe765ae23
- `results/raw/M1-eval-c6-qwen7b-full-s0-20261019T033011Z/predictions/resume_prefix.jsonl` a8fa8b801989c047ee82e5af58ee2534a8f72b3522ab350780c71ea0a935dace
- `results/raw/M1-eval-c6-qwen7b-full-s0-20261019T033011Z/predictions/sample-t1-p1.jsonl` 90e87d17ea33808b3d49f289e8154f2575e8259b57e8b5409ebc5240d32fcbda
- `results/raw/M1-eval-c6-qwen7b-full-s1-20261019T033011Z/manifest.json` 259cc110e367c54c5be8b15703201a57c0cf51d2e744f86cb6f07b52d7eed274
- `results/raw/M1-eval-c6-qwen7b-full-s1-20261019T033011Z/predictions/greedy.jsonl` 54f3f93a6f2fa58759c260e6aa1eeb9c2256c50a3100b67d5bdd55bc59ea644f
- `results/raw/M1-eval-c6-qwen7b-full-s1-20261019T033011Z/predictions/resume_prefix.jsonl` cbdfaaccbbf52bc06e9f3e2575402a7f2566c4b6622b0f4bd54617e3d544f2ba
- `results/raw/M1-eval-c6-qwen7b-full-s1-20261019T033011Z/predictions/sample-t1-p1.jsonl` 1c1cef89224c4848263cb942964e352a831221c2702a1d901d80c886571d4356
- `results/raw/M1-eval-c6-qwen7b-full-s2-20261019T083821Z/manifest.json` 771fa5d6128a18f422a2e5bdc3ac1665a7517c0c3a740895f4cb632df6d88354
- `results/raw/M1-eval-c6-qwen7b-full-s2-20261019T083821Z/predictions/greedy.jsonl` 1c433dc342f74c2d5978ce025ef41761c207d075b1f57c32202ea5afe6eb1480
- `results/raw/M1-eval-c6-qwen7b-full-s2-20261019T083821Z/predictions/resume_prefix.jsonl` 659495010dde3cc3ba048167a95fea8f941c2ae42c6d73476a19212db809eac6
- `results/raw/M1-eval-c6-qwen7b-full-s2-20261019T083821Z/predictions/sample-t1-p1.jsonl` 02bebee2971096f24eee7655b6a628580a07048aae2ee8d9b49333b4d593212d
- `results/raw/M1-eval-c6-qwen7b-heldout-s0-20261019T080911Z/manifest.json` e118ae3d334b3333e110a9c810b76ea9dd79962f61c69faf692d2d8471e438ed
- `results/raw/M1-eval-c6-qwen7b-heldout-s0-20261019T080911Z/predictions/greedy.jsonl` d5c0a1f81fa09d3ee4f943a5e24c73970d17a3fdf0028a424b65b47f8f9b76fd
- `results/raw/M1-eval-c6-qwen7b-heldout-s1-20261019T074048Z/manifest.json` 38044bea0d9c2f1bf31d78f69bf340a00f12b3dfaaaf8b6cbc76444e2379bf67
- `results/raw/M1-eval-c6-qwen7b-heldout-s1-20261019T074048Z/predictions/greedy.jsonl` 1f1806b4521569ef5d1dd66220988386cc29132b970d11f2944d43083eb6b9c8
- `results/raw/M1-eval-c6-qwen7b-heldout-s2-20261019T134728Z/manifest.json` 4eab07380fc69cb451438e9d36a9ec4987ebc044c63f59706b9ed8adf78f0acd
- `results/raw/M1-eval-c6-qwen7b-heldout-s2-20261019T134728Z/predictions/greedy.jsonl` 1cc6461f264a6ae8136281ab4d7dc862476b511d3e5b916c0525ff8dafe41de9
- `results/raw/M1-eval-c6-qwen7b-renamed-s0-20261019T075405Z/manifest.json` 1df811dd4a91834229f340c4712aabf6662e8885ee2bb298e880a62806f8b401
- `results/raw/M1-eval-c6-qwen7b-renamed-s0-20261019T075405Z/predictions/greedy.jsonl` 2885c8b0a9a5a68ea425f3f62fb7a3af40e88ba1e126b3d7bb9d457c09427fed
- `results/raw/M1-eval-c6-qwen7b-renamed-s1-20261019T072711Z/manifest.json` 3f2834a50ed5fd00dfbfe506a560cf6c2714d51aebbf9f4fb4979e5bd551bbcc
- `results/raw/M1-eval-c6-qwen7b-renamed-s1-20261019T072711Z/predictions/greedy.jsonl` fd0336eb5a0d7559f7304a146df934a1b03743286a7df83865861bf982ffec2c
- `results/raw/M1-eval-c6-qwen7b-renamed-s2-20261019T133215Z/manifest.json` 0131832ef0a2332280485debf4bbd725b81efa1460276181d1fa7d25c7c3199b
- `results/raw/M1-eval-c6-qwen7b-renamed-s2-20261019T133215Z/predictions/greedy.jsonl` 3eb88ea3fbf98ff7da1e66fd4de8b56eb19207cd12c8749acee193c5124a2b73
- `results/raw/M1-eval-c6-qwen7b-toolhop-s0-20261019T081513Z/manifest.json` c49da8088bb998f2a87d7374e812009ac824075ffc77a95c5de3e8d12c010fa2
- `results/raw/M1-eval-c6-qwen7b-toolhop-s0-20261019T081513Z/predictions/greedy.jsonl` 174213c7370b14619c0c10a3a7fd3493ef1791cdaab57a7063c75506b4d172ed
- `results/raw/M1-eval-c6-qwen7b-toolhop-s0-20261019T081513Z/predictions/sample-t1-p1.jsonl` 3110e573418007a8e30a684967ba62ec48d014bf741e4402ca236b63e01ba402
- `results/raw/M1-eval-c6-qwen7b-toolhop-s1-20261019T074623Z/manifest.json` ff2680b2a9db62eb60c6c3d9dc2fdf8434005b43a8beaea3d057597034b3a166
- `results/raw/M1-eval-c6-qwen7b-toolhop-s1-20261019T074623Z/predictions/greedy.jsonl` c9000eb738595f97cd2ccf9a4ca18ea8251330b4a783fe26457bfc45ddfa2e94
- `results/raw/M1-eval-c6-qwen7b-toolhop-s1-20261019T074623Z/predictions/sample-t1-p1.jsonl` 0a0dbac42be4cc8ed949fb4f6c4ab8ebc4b937e84cb115350484ab9f2081bdd9
- `results/raw/M1-eval-c6-qwen7b-toolhop-s2-20261019T135434Z/manifest.json` 7621d2130aca33e00b35b0ea76330d794358ce3224aa2823d3b4e8598a783084
- `results/raw/M1-eval-c6-qwen7b-toolhop-s2-20261019T135434Z/predictions/greedy.jsonl` 3788ce5a12882fa5a74b1c34cfc28cf18764c5ad0969adcd376aceaf9304189d
- `results/raw/M1-eval-c6-qwen7b-toolhop-s2-20261019T135434Z/predictions/sample-t1-p1.jsonl` b299d150c88f78b0906203753f7d2e856c5cb4dd4998b61994e02d51980e69e1
- `docs/EVAL_PLAN.md` 4d4ada994eddb9879fcb686e226fe4ba67924c836642e3f48e43af90754d3bdd
- `results/raw/M1-c1-qwen7b-s0-20261018T060950Z/manifest.json` 42871d72436d4c1216ff428d8bfdd67df9ef6dda100864c181f79d8fce8ebc28
- `results/raw/M1-c1-qwen7b-s0-20261018T060950Z/train_log.jsonl` 060b0b0e8eb56072ef58198c5073769b649e2ed517a077bce3a5f7ac2ff0aecf
- `results/raw/M1-c1-qwen7b-s0-20261018T060950Z/data/data_manifest.json` 8becdfde7aa79322c7e7bd083f61e7bd12c4dc840d502a63bc986085c19328da
- `results/raw/M1-c1-qwen7b-s1-20261018T060950Z/manifest.json` 3bd06db9fe57c62bc6188a25e394ea3b7ac582b96f2e09a5c8a6eb140fcd2580
- `results/raw/M1-c1-qwen7b-s1-20261018T060950Z/train_log.jsonl` 23337be80ad4ec05cdb72e70e1fe7209dd26b62395ab465b5b6e164423125d50
- `results/raw/M1-c1-qwen7b-s1-20261018T060950Z/data/data_manifest.json` 8becdfde7aa79322c7e7bd083f61e7bd12c4dc840d502a63bc986085c19328da
- `results/raw/M1-c1-qwen7b-s2-20261018T060950Z/manifest.json` 671e0ecbd2122936781e1d6121f8436d9ae568bdd20fe79de3632e1e1a63a29c
- `results/raw/M1-c1-qwen7b-s2-20261018T060950Z/train_log.jsonl` cff2b4c2dd0890523f41722d3c68677aa5f64c164e8b02efa7943ce33c91cdc1
- `results/raw/M1-c1-qwen7b-s2-20261018T060950Z/data/data_manifest.json` 8becdfde7aa79322c7e7bd083f61e7bd12c4dc840d502a63bc986085c19328da
- `results/raw/M1-c2-qwen7b-s0-20261018T045428Z/manifest.json` 2d19e2a90a048fba218c22ff833d0ca679f07068d0fe6af8b20da2a2d1a1d511
- `results/raw/M1-c2-qwen7b-s0-20261018T045428Z/train_log.jsonl` cfa07683306f0c9b10331dd8cffd40c951981f59ca56626671f2c404fd01ad1b
- `results/raw/M1-c2-qwen7b-s0-20261018T045428Z/data/data_manifest.json` 8becdfde7aa79322c7e7bd083f61e7bd12c4dc840d502a63bc986085c19328da
- `results/raw/M1-c2-qwen7b-s1-20261018T060950Z/manifest.json` a0a8f92e4c3897ddd3a07d8fcd58985e4242169ebdeac6c5fb29403196b1766e
- `results/raw/M1-c2-qwen7b-s1-20261018T060950Z/train_log.jsonl` 9d2cabdff82ecc0013d7f8c51815ff95aead781bbfcf52f9f1920f1e61538b21
- `results/raw/M1-c2-qwen7b-s1-20261018T060950Z/data/data_manifest.json` 8becdfde7aa79322c7e7bd083f61e7bd12c4dc840d502a63bc986085c19328da
- `results/raw/M1-c2-qwen7b-s2-20261018T060950Z/manifest.json` e2215af2104ee3129ca3b3d0e8b298b7db27efc209e0fcebf8ae798e4a067048
- `results/raw/M1-c2-qwen7b-s2-20261018T060950Z/train_log.jsonl` 5b4fd9c4128d4485b64e9347cc3223072d12103c7e08695300aaa2e2d0cb78f9
- `results/raw/M1-c2-qwen7b-s2-20261018T060950Z/data/data_manifest.json` 8becdfde7aa79322c7e7bd083f61e7bd12c4dc840d502a63bc986085c19328da
- `results/raw/M1-c3-qwen7b-s0-20261018T060952Z/manifest.json` f4fe9ee2009fe9851fb4e977b0e5c98f3cab1bb9e75b6540138ac762f8c6170a
- `results/raw/M1-c3-qwen7b-s0-20261018T060952Z/train_log.jsonl` 145d5c68373a6cf3f0a6d045d8b66c6f1e0dc983520c9ff3e674ffb9242545e3
- `results/raw/M1-c3-qwen7b-s0-20261018T060952Z/data/data_manifest.json` 8becdfde7aa79322c7e7bd083f61e7bd12c4dc840d502a63bc986085c19328da
- `results/raw/M1-c3-qwen7b-s1-20261018T060952Z/manifest.json` 7e40ba3dd5c3e77c3e87ce7498a8734c0b9232642b5780983d13f075c5c86d29
- `results/raw/M1-c3-qwen7b-s1-20261018T060952Z/train_log.jsonl` 06b074ab9cd19d0635ffe366a65e1333c1a2501011dff821e59822ccc7ed749f
- `results/raw/M1-c3-qwen7b-s1-20261018T060952Z/data/data_manifest.json` 8becdfde7aa79322c7e7bd083f61e7bd12c4dc840d502a63bc986085c19328da
- `results/raw/M1-c3-qwen7b-s2-20261018T060952Z/manifest.json` 1d779aa3a3707f1cc5adc7c4b368c8d0c959ea5bd1fae02cb05a3f3db45d8d8c
- `results/raw/M1-c3-qwen7b-s2-20261018T060952Z/train_log.jsonl` 1fefb236245cd70d54a8f26c543fa3e8a63d41484605c89fb9fb88a37f24834a
- `results/raw/M1-c3-qwen7b-s2-20261018T060952Z/data/data_manifest.json` 8becdfde7aa79322c7e7bd083f61e7bd12c4dc840d502a63bc986085c19328da
- `results/raw/M1-c4-qwen7b-s0-20261018T060956Z/manifest.json` bc622e552b5550ce6c1ae4aae9cd7df66c74b0abcfa1813ec4b09ef57fc41709
- `results/raw/M1-c4-qwen7b-s0-20261018T060956Z/train_log.jsonl` a0b67a4c098a37045e77169a9b94cba24127e8c1fdb384c9b388bd8c8cba6a66
- `results/raw/M1-c4-qwen7b-s0-20261018T060956Z/data/data_manifest.json` 8becdfde7aa79322c7e7bd083f61e7bd12c4dc840d502a63bc986085c19328da
- `results/raw/M1-c4-qwen7b-s1-20261018T060956Z/manifest.json` 67d6d53ef6e7a8ef78852f158f46c5a5edb2056d2839373a447744312e387039
- `results/raw/M1-c4-qwen7b-s1-20261018T060956Z/train_log.jsonl` fbe0f0680ae1a2bd0392ffca533b705beffa9a660d3443895b8106a769a692fd
- `results/raw/M1-c4-qwen7b-s1-20261018T060956Z/data/data_manifest.json` 8becdfde7aa79322c7e7bd083f61e7bd12c4dc840d502a63bc986085c19328da
- `results/raw/M1-c4-qwen7b-s2-20261018T060956Z/manifest.json` 6a626bd31ed469387afd10b275a9871fbfb1316ca54b05beced0d3778e6387b5
- `results/raw/M1-c4-qwen7b-s2-20261018T060956Z/train_log.jsonl` bc2a6754ca2478f1fa313762ece3fbdef359fb943f748521279f2bb9d3a33ca6
- `results/raw/M1-c4-qwen7b-s2-20261018T060956Z/data/data_manifest.json` 8becdfde7aa79322c7e7bd083f61e7bd12c4dc840d502a63bc986085c19328da
- `results/raw/M1-c5-qwen7b-s0-20261018T060956Z/manifest.json` 478b8166652b47a6f018ba0c029fab96131747db21eba49d5a62f0122649fbc7
- `results/raw/M1-c5-qwen7b-s0-20261018T060956Z/train_log.jsonl` 07b7bb54e5d6f6e84eed8a3eed59e891575ac1af516bcd783fe594b6592645f0
- `results/raw/M1-c5-qwen7b-s0-20261018T060956Z/data/data_manifest.json` 8becdfde7aa79322c7e7bd083f61e7bd12c4dc840d502a63bc986085c19328da
- `results/raw/M1-c5-qwen7b-s1-20261018T060956Z/manifest.json` b98ede597f1f20c6240dec61e5c2805d17fb523ba038eaf9e3a762793a77b09d
- `results/raw/M1-c5-qwen7b-s1-20261018T060956Z/train_log.jsonl` 4efb13e91b38c61c28d8f66f54d7a3e906f1d7fd0b4605a0a38bcccf852a9349
- `results/raw/M1-c5-qwen7b-s1-20261018T060956Z/data/data_manifest.json` 8becdfde7aa79322c7e7bd083f61e7bd12c4dc840d502a63bc986085c19328da
- `results/raw/M1-c5-qwen7b-s2-20261018T060956Z/manifest.json` 274a939dfbe6dc2998d55a1f3aa06bafcb865c44a882de107ba4b09e92c3b59e
- `results/raw/M1-c5-qwen7b-s2-20261018T060956Z/train_log.jsonl` 3ce110b8521ac0813ebe825223ddb8da12e1d7b9deae82579985c509dcbad6be
- `results/raw/M1-c5-qwen7b-s2-20261018T060956Z/data/data_manifest.json` 8becdfde7aa79322c7e7bd083f61e7bd12c4dc840d502a63bc986085c19328da
- `results/raw/M1-c6-qwen7b-s0-20261018T045428Z/manifest.json` 79d15a168656d44534cf669ba810ab97a04ca86decb6aa1b2711d2ac1abdadb3
- `results/raw/M1-c6-qwen7b-s0-20261018T045428Z/train_log.jsonl` 415a88f09514f5e92f38591e3fa10fc978e67fab2f76e04d408ecbc1e51f639a
- `results/raw/M1-c6-qwen7b-s0-20261018T045428Z/data/data_manifest.json` 8becdfde7aa79322c7e7bd083f61e7bd12c4dc840d502a63bc986085c19328da
- `results/raw/M1-c6-qwen7b-s1-20261018T060956Z/manifest.json` bf5007d0d9b6a68e8baa9a885378c158daf6c6c4760d3818d03eaa072cc4ee09
- `results/raw/M1-c6-qwen7b-s1-20261018T060956Z/train_log.jsonl` bedc6c15a7fa1f4e387bc40499274b728074a91927d05515410e1f711ed9ff7b
- `results/raw/M1-c6-qwen7b-s1-20261018T060956Z/data/data_manifest.json` 8becdfde7aa79322c7e7bd083f61e7bd12c4dc840d502a63bc986085c19328da
- `results/raw/M1-c6-qwen7b-s2-20261018T060956Z/manifest.json` 8c991be4d3eb5e39d970ae8115c979777330f5d353d52463c2efc8502332956a
- `results/raw/M1-c6-qwen7b-s2-20261018T060956Z/train_log.jsonl` 974348ea193fb90a52f5941fa4af1b9c6e5676bbc9d06bb481e3d20d8aff7b0c
- `results/raw/M1-c6-qwen7b-s2-20261018T060956Z/data/data_manifest.json` 8becdfde7aa79322c7e7bd083f61e7bd12c4dc840d502a63bc986085c19328da

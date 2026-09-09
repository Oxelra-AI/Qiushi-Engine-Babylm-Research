# Near-length FineWeb rewrites and repetition

## Result

Two exact 100M-exposure models were trained on the frozen medium risk-hard clean-Qwen row-holdout overlay, using the inherited COMPACT_EXPERIENCE clean-Qwen recipe (DeBERTa-v2 8x480, baseline16k, fixed seq256, fixed WWM, AdamW lr 0.001, warmup 0.06, seed 43 / init 43022 / train RNG 43023, checkpoint every 1M words).

Training tasks:

- `cleanqwen_fineweb_repeat_near_core`: finished 2026-08-29T13:25:16Z, 100M words, 2,533 steps, loss_first 9.8153, loss_last 2.4065, 100 checkpoints.
- `cleanqwen_fineweb_near_view_core`: finished 2026-08-29T13:27:13Z, 100M words, 2,533 steps, loss_first 9.8145, loss_last 2.4082, 100 checkpoints.

The initial evaluation without AoA failed because of a Python dependency error (`ModuleNotFoundError: transformers`) and an output-configuration error. After correcting the evaluation environment, the evaluator produced the following results:

`experiments/archive/frontier_consolidation/data/density_noaoa_eval_retry/density_noaoa_eval_summary.json`

Task-family scores:

| target | BLiMP | Supp | EWoK | Entity fast | Entity full | COMPS | GPIQA | Reading | equal7 fast | equal7 fullEnt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| near_repeat | 67.190 | 62.400 | 49.450 | 23.250 | 23.560 | 51.780 | 34.150 | 7.745 | 42.281 | 42.325 |
| near_view | 66.820 | 63.200 | 49.180 | 27.970 | 27.130 | 51.170 | 34.135 | 8.520 | 42.999 | 42.879 |

Mechanism contrast `near_view - near_repeat`:

- equal7 fast +0.719; equal7 using full Entity +0.554
- Entity fast +4.72; Entity full +3.57
- Reading +0.775; Supplement +0.80
- GlobalPIQA mean -0.015, effectively flat with parallel +0.97 and nonparallel -1.00
- BLiMP -0.37, EWoK -0.27, COMPS -0.61

## Interpretation

This is the first frontier_consolidation model-behavior evidence that the FineWeb same-source generated second view has transferable value over source repetition at the inherited clean-Qwen base. The training losses were essentially tied, so the result is not a simple language-modeling-loss effect. The strongest gain is Entity Tracking, matching earlier same-window second-view evidence that generated views can help object/state/entity representations when source and view remain adjacent. The gain does not solve the full SOTA problem by itself: EWoK, COMPS, and GlobalPIQA remain below the visible leader surface, and this screen omits SuperGLUE and AoA. It nevertheless justifies continuing the compact and reinvested variants because the source-aligned view mechanism is alive under real training.

The near-view arm also provides a useful ceiling/control for compression. If compact core views beat compact source repetition and approach or exceed near_view while enabling added-source reinvestment, the density principle becomes stronger: generated views would be useful not only as near paraphrases but as a way to trade redundant words for more distinct source relations. If compact views fail while near views help, the next mechanism work should focus on preserving relation force/modality under compression rather than expanding the current compact corpus.

## Work in progress when this note was written

- training `cleanqwen_fineweb_repeat_compact_core_neutral` on GPU1.
- training `cleanqwen_fineweb_compact_view_core_neutral` on GPU0.

After both compact core arms finish, evaluate them with `experiments/archive/frontier_consolidation/scripts/fast_eval_density_arms.py` and compare `compact_view_core - compact_repeat_core`. If compact view core is positive on broad task families, launch/evaluate `cleanqwen_fineweb_compact_view_reinvest` to test added-source diversity; the dry-run and expected hash for that arm already succeeded.

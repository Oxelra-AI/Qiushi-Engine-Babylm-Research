# changed block neardup scan AoA source repair and decision thresholds

## Why this repair was necessary

The active best endpoint `compact_view_reinvest` seed43022 has a narrow SOTA-facing margin: Overall 42.0867857, only +0.2868 above the visible 41.8 leader. The arithmetic shows this lead rests on AoA staying non-significant (leaderboard-unit AoA = 0.0).

During changed block neardup scan source reading, I found a row-count mismatch in the local AoA path:

- `experiments/archive/compact_experience/scripts/aoa_local_ckpts_for_model.py` hardcodes `load_eval(word_path, 20, False)`.
- Current repo source `experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_pipeline/collate_preds.py` has `AOA_SIZE = 8005` and checks that each strict-small checkpoint has exactly 8005 AoA predictions.
- Counting `cdi_childes.json` directly gives:
  - `min_context=0`: 504 words / **8005 contexts**
  - `min_context=20`: 328 words / **6560 contexts**

Therefore the older min_context=20 helper reproduces the previous local convention but cannot settle current official row-count compatibility. The current source-compatible setting is `min_context=0`, expected_rows_per_step=8005.

## Repairs made in changed block neardup scan

- Cancelled the first seed43122 AoA evaluation before it consumed GPU because it would have used the old min_context=20 helper.
- Cancelled the old AoA localization evaluation because it was hardcoded to min_context=20. It had already produced partial older-convention mechanism evidence:
  - near_repeat: AoA 0.0, row_count_values [6560]
  - near_view: AoA 0.0, row_count_values [6560]
  - compact_repeat_core: AoA −0.188992, row_count_values [6560]
  These are useful as local/mechanism evidence, but not sufficient for current official viability.
- Wrote `experiments/archive/frontier_consolidation/scripts/aoa_local_ckpts_minctx.py`, a repaired local-checkpoint AoA helper with explicit `--min_context` and `--expected_rows_per_step`.
- Wrote `experiments/archive/frontier_consolidation/scripts/run_official_rowcount_aoa_reinvest_seeds.py`, a two-target launcher for:
  - `reinvest_seed43022`: `experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022/hf_model`
  - `reinvest_seed43122`: `experiments/archive/representation_and_objectives/training/runs/repl_compact_view_reinvest_seed43122/hf_model`
- CPU preflight confirmed both ladders are complete, with all chck_1M…chck_100M checkpoints present, and the repaired helper exists.
- Started the repaired evaluation of official-row-count AoA (`min_context=0`, 8005 rows/checkpoint) on both reinvest seeds. This is the decisive pending AoA comparison.

## Decision thresholds

For the seed43022 endpoint with all non-AoA columns fixed as in the full evaluation:

- Non-AoA sum = Overall(42.086785719138156) × 9 − AoA(0.0) = **378.7810714722434**.
- To remain above 41.8: total must be ≥ 376.2, so leaderboard-unit AoA must be ≥ **−2.5810714722434**.
  - Since leaderboard AoA = raw_curve_fitness × 100, raw AoA must be ≥ **−0.025810714722434** if significant.
- To remain above clean-Qwen 41.34429066479573: total must be ≥ 372.0986159831616, so leaderboard-unit AoA must be ≥ **−6.6824554890818** (raw ≥ −0.066824554890818).

Thus for seed43022 under the repaired official-row-count setting:

- p > 0.1 → AoA maps to 0.0 and the 42.0868 result remains viable under current row count.
- p ≤ 0.1 with raw correlation more negative than −0.02581 → the endpoint no longer clears the visible 41.8 leader.
- p ≤ 0.1 with raw correlation more negative than −0.06682 → it also falls below clean-Qwen.

For seed43122:

- AoA must be measured before spending full official-surface GPU time. A fast/no-AoA screen cannot settle this.
- If seed43122 min_context=0 AoA is p > 0.1, or significant but only weakly negative enough that a plausible full surface remains above 41.8, complete the remaining official surface to determine the second-seed Overall.
- If seed43122 min_context=0 AoA is significantly negative at a magnitude that would erase the margin, do not launch expensive full second-seed evaluation; treat the endpoint as seed-fragile and return to mechanism repair.

## What not to do

- Do not use the fast seed43122 screen as the AoA stability answer; it has no official AoA.
- Do not spend full second-seed SuperGLUE/full-surface time before reading the official-row-count AoA results.
- Do not change or retrain the frozen seed43022 endpoint while this row-count AoA evidence is unresolved.

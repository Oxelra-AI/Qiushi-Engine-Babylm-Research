# full overall eval — full official-style Overall interpretation and immediate next experiment

## Evidence inspected

- Summary: `experiments/archive/compact_experience/data/full_overall_eval/full_overall_summary.json`.
- Per-target evidence:
  - `experiments/archive/compact_experience/data/full_overall_eval/per_target/mix25_16k_seed43.json`
  - `experiments/archive/compact_experience/data/full_overall_eval/per_target/phase2s_mix25.json`
- Note table: `research/notes/compact_experience/full_overall_eval.md`.

All zero-shot, Reading, and SuperGLUE tasks for the two selected candidates returned code 0 and have recorded prediction paths. Launcher stderr files were empty.

## Full official-style scores

Important AoA status: both evaluated COMPACT_EXPERIENCE candidates only saved `chck_10M,20M,...,100M`. They lack `chck_1M..chck_9M`, so official AoA cannot be computed for these artifacts without retraining or resaving. The reported Overall below uses AoA=0.0 only for immediate leaderboard-style arithmetic and is **not submit-ready official AoA**.

| target | Overall | BLiMP | Supp | EWoK | Entity | COMPS | GPIQA | SuperGLUE | Reading | AoA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mix25_16k_seed43 | 41.4803 | 67.71 | 62.28 | 52.26 | 22.70 | 51.98 | 38.12 | 69.8678 | 8.405 | 0.0 |
| phase2s_mix25 | 40.5637 | 66.63 | 58.29 | 53.40 | 24.78 | 52.71 | 34.635 | 68.5535 | 6.075 | 0.0 |
| inherited INITIAL_MODEL_STUDIES best chck_80M | 40.7028 | 66.54 | 61.00 | 50.44 | 22.20 | 53.00 | 37.59 | 68.2551 | 7.30 | 0.0 |
| public leader reference | 41.80 | 67.20 | 56.01 | 56.07 | 28.45 | 53.57 | 39.67 | 69.79 | 5.42 | 0.0 |

## Interpretation

The 16k mix25 endpoint is now the best locally measured full official-style coordinate: Overall 41.4803, +0.7775 over the inherited 40.7028 coordinate. This validates the low-dose aligned mixture as a real official-score improvement, not merely a fast-screen artifact.

It is still not SOTA: it is -0.3197 behind the visible 41.8 leader. The remaining deficits versus the leader are concentrated in Entity (-5.75), EWoK (-3.81), COMPS (-1.59), and GlobalPIQA (-1.55). It already exceeds the leader on Supplement (+6.27), SuperGLUE (+0.078), Reading (+2.985), and BLiMP (+0.51). The bottleneck is therefore not general finetuning ability or broad syntax alone; it is the leader-like world/commonsense/entity columns without sacrificing our strong Supplement/Reading/SuperGLUE.

The Phase-2 12x384/LAMB/40k route did not help Overall in this implementation. It improved Entity and EWoK relative to some controls but lost too much Supplement, GlobalPIQA, Reading, and SuperGLUE. Its full Overall 40.5637 is below both mix25_16k and the inherited coordinate. Current evidence argues against continuing that exact Phase-2 recipe as the main SOTA route without a major mechanism repair.

## Immediate next action launched

Since INITIAL_MODEL_STUDIES's best endpoint was `chck_80M` rather than `chck_100M`, the current `chck_100M` mix25 result may not be the best single endpoint. I wrote and launched a full-eval zero-shot/Reading checkpoint trajectory sweep:

- Script: `experiments/archive/compact_experience/scripts/eval_checkpoint_trajectory_fullzeroshot.py`
- Launcher: `experiments/archive/compact_experience/scripts/launch_checkpoint_trajectory_fullzeroshot.sh`
- Outputs: `experiments/archive/compact_experience/data/checkpoint_trajectory`

It evaluates `chck_10M..chck_100M` for both `mix25_16k_seed43` and `phase2s_mix25` on full official zero-shot columns plus Reading. The proposed selection rule, once those measurements are complete, is to select the best existing checkpoint by full-eval equal-7, then run SuperGLUE only on any checkpoint whose zero-shot/Reading gain can plausibly exceed the current 41.4803 coordinate. If an endpoint is chosen for any submit-like artifact, a retrain/resave path with `chck_1M..chck_9M` is still required for official AoA.

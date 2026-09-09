# reading direction and official40k route judgment — Reading direction and official40k route judgment

## Source-grounded Reading direction

Official reading runner output is a positive score computed as percent relative R² gain from adding model surprisal to baseline regressors:

- `evaluation_pipeline/reading/run.py` writes `EYE TRACKING SCORE` as the mean of eye-tracking relative R² gains across RT variables.
- It writes `SELF-PACED READING SCORE` as the relative R² gain for self-paced reading time.

The leaderboard uses these values directly. The visible Strict-Small leader row in `data/leaderboard_probe/leaderboard_config_parsed.json` has:

- Reading = 5.42
- Eye Tracking = 6.83
- Self-paced Reading Time = 4.02
- AoA = 0.0
- Human-like Average = 2.71 = (5.42 + 0.0) / 2

Therefore Reading is a direct higher-is-better column in the current leaderboard aggregation. The official40k Reading value 0.89 is not an improvement over baseline16k DeBERTa's 7.62; it is a large loss.

## official40k vs baseline16k DeBERTa on available columns

Evidence:

- `data/official40k_vs_baseline16k_available_comparison.json`
- raw Reading reports:
  - baseline16k: `training/runs/babylm_fullcycle_debertav2_8x480_wwm_seed42_100M_b256/eval_results_available/hf_model/chck_100M/zero_shot/mlm/reading/report.txt`
  - official40k: `training/runs/babylm_fullcycle_debertav2_8x480_official40k_wwm_seed42_100M_b128_acc2/eval_results_available/hf_model/chck_100M/zero_shot/mlm/reading/report.txt`

| column | baseline16k | official40k | 40k − 16k |
|---|---:|---:|---:|
| BLiMP | 66.76 | 66.65 | -0.11 |
| Supplement | 59.88 | 59.27 | -0.61 |
| Entity | 22.62 | 22.16 | -0.46 |
| COMPS | 52.19 | 52.32 | +0.13 |
| GlobalPIQA mean | 35.635 | 36.605 | +0.970 |
| Reading | 7.62 | 0.89 | -6.73 |

Known-column delta sum over these six leaderboard columns is **-6.81**, or **-1.135 points per column**. The modest GlobalPIQA gain and tiny COMPS gain do not compensate for the Reading loss plus small BLiMP/Supplement/Entity losses.

## Consequence for further official40k evaluation

If official40k has the same SuperGLUE and full EWoK as baseline16k DeBERTa, its non-official sensitivity score would drop by roughly 0.76–0.85 Overall points depending on whether EWoK is omitted or substituted by fast EWoK. To break even with baseline16k over the seven known-plus-SuperGLUE columns while EWoK stays equal, official40k SuperGLUE would need to improve by about **+6.81** points over baseline16k SuperGLUE 68.02, i.e. reach roughly **74.83**. That is possible in principle but large relative to the remaining leader gap and should not be assumed.

If official40k SuperGLUE equals baseline16k and AoA remains 0, official40k would require full EWoK about **70.29** to reach the visible leader Overall 41.80. This is far above the visible leader's EWoK 56.07 and far above our fast-interim EWoK 46.18, so official40k is not currently the best candidate under ordinary expectations.

## Route judgment

The official40k experiment is scientifically valuable but currently mixed-to-negative for the Overall target:

- useful positives: lower full-cycle truncation (0.182 vs 0.292), lower tokens/word, preserved BLiMP near 67, Supplement still above the visible leader, GlobalPIQA mean +0.97.
- harmful effects: Reading collapses from 7.62 to 0.89 under the official runner; Entity worsens from 22.62 to 22.16; Supplement and BLiMP also dip.

Before spending more expensive evaluation time on official40k, the next useful work is to decide whether SuperGLUE/AoA are worth running only as a limited completion of an already mixed result, or whether execution should pivot back to the protected baseline16k DeBERTa backbone and attack Entity/GlobalPIQA through a targeted relation/entity mechanism. If official40k is pursued further, it should be because full EWoK or SuperGLUE plausibly changes the judgment, not because Reading was mistakenly treated as lower-is-better.

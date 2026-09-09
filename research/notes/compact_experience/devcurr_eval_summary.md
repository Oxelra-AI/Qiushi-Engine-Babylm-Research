# aoa safety audit and route developmental first-pass clean-Qwen evaluation

Summary JSON: `experiments/archive/compact_experience/data/devcurr_eval/devcurr_eval_summary.json`

AoA below is leaderboard units (`100 * raw correlation`). Same-seed contrasts isolate the first-pass order effect relative to the already evaluated clean-Qwen arms.

| target | Overall | BLiMP | Supp | EWoK | Entity | COMPS | GPIQA | SuperGLUE | Reading | AoA lb | AoA raw | ready |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen_devcurr_firstpass_seed43022 | 40.6475 | 66.03 | 62.83 | 50.21 | 23.72 | 51.44 | 34.23 | 69.7325 | 7.64 | 0.0 | 0.0 | True |
| qwen_devcurr_firstpass_seed43122 | 40.5656 | 65.69 | 59.48 | 49.99 | 24.86 | 52.17 | 35.63 | 69.4606 | 7.8 | 0.0 | 0.0 | True |
| qwen_clean_aligned_seed43022 | 41.3443 | 66.84 | 62.84 | 50.19 | 25.76 | 51.78 | 36.62 | 70.3086 | 7.76 | 0.0 | 0.0 | True |
| qwen_clean_aligned_seed43122 | 40.6501 | 66.02 | 61.51 | 50.43 | 25.26 | 52.06 | 34.62 | 68.7455 | 7.21 | 0.0 | 0.0 | True |

## Same-seed contrasts

- `devcurr_seed43022_minus_clean` Overall Δ=-0.6968; task Δs: BLiMP=-0.81, Supplement=-0.01, EWoK=0.02, Entity=-2.04, COMPS=-0.34, GlobalPIQA=-2.395, SuperGLUE=-0.576, Reading=-0.12, AoA=0.0; contributions: NLP7=-0.6835, Reading=-0.0133, AoA=0.0
- `devcurr_seed43122_minus_clean` Overall Δ=-0.0844; task Δs: BLiMP=-0.33, Supplement=-2.03, EWoK=-0.44, Entity=-0.4, COMPS=0.11, GlobalPIQA=1.015, SuperGLUE=0.715, Reading=0.6, AoA=0.0; contributions: NLP7=-0.1511, Reading=0.0667, AoA=0.0

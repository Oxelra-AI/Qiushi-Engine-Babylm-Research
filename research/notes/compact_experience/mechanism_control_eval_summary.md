# clean qwen control interpretation and next mechanism mechanism-control evaluation

Summary JSON: `experiments/archive/compact_experience/data/mechanism_eval_summary.json`

## Targets
| target | Overall | BLiMP | Supp | EWoK | Entity | COMPS | GPIQA | SuperGLUE | Reading | AoA leaderboard | AoA raw | ready |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| official_lengthmatched | 38.9404 | 67.29 | 59.04 | 50.8 | 23.31 | 53.13 | 36.68 | 67.9681 | 7.95 | -15.7045 | -0.157045 | True |
| qwen_clean_aligned | 41.3443 | 66.84 | 62.84 | 50.19 | 25.76 | 51.78 | 36.62 | 70.3086 | 7.76 | 0.0 | 0.0 | True |
| qwen_shuffled_control | 39.224 | 68.29 | 59.42 | 50.33 | 22.87 | 52.29 | 36.65 | 67.4352 | 8.52 | -12.7891 | -0.127891 | True |
| official_original_dup | 37.8266 | 65.52 | 60.72 | 48.0 | 18.61 | 52.15 | 31.65 | 68.8245 | 8.28 | -13.3155 | -0.133155 | True |
| selected_original_dup_all | 40.9915 | 64.74 | 61.2 | 48.37 | 24.35 | 51.48 | 41.08 | 69.5939 | 8.11 | 0.0 | 0.0 | True |
| qwen_separated_pair | 39.5643 | 67.07 | 58.38 | 48.54 | 20.9 | 51.84 | 37.16 | 63.5587 | 8.62 | 0.0 | 0.0 | True |

## Contrasts

- `aligned_minus_selected_original_dup_all` Overall Δ=0.3528; task Δs: BLiMP=2.1, Supplement=1.64, EWoK=1.82, Entity=1.41, COMPS=0.3, GlobalPIQA=-4.46, SuperGLUE=0.715, Reading=-0.35, AoA=0.0
- `aligned_minus_separated_pair` Overall Δ=1.78; task Δs: BLiMP=-0.23, Supplement=4.46, EWoK=1.65, Entity=4.86, COMPS=-0.06, GlobalPIQA=-0.545, SuperGLUE=6.75, Reading=-0.865, AoA=0.0
- `separated_minus_shuffled` Overall Δ=0.3403; task Δs: BLiMP=-1.22, Supplement=-1.04, EWoK=-1.79, Entity=-1.97, COMPS=-0.45, GlobalPIQA=0.515, SuperGLUE=-3.877, Reading=0.105, AoA=12.789
- `selected_original_dup_all_minus_official` Overall Δ=2.0511; task Δs: BLiMP=-2.55, Supplement=2.16, EWoK=-2.43, Entity=1.04, COMPS=-1.65, GlobalPIQA=4.4, SuperGLUE=1.626, Reading=0.16, AoA=15.705
- `separated_pair_minus_official` Overall Δ=0.6239; task Δs: BLiMP=-0.22, Supplement=-0.66, EWoK=-2.26, Entity=-2.41, COMPS=-1.29, GlobalPIQA=0.485, SuperGLUE=-4.409, Reading=0.675, AoA=15.705
- `selected_original_dup_all_minus_old_originaldup` Overall Δ=3.165; task Δs: BLiMP=-0.78, Supplement=0.48, EWoK=0.37, Entity=5.74, COMPS=-0.67, GlobalPIQA=9.43, SuperGLUE=0.769, Reading=-0.17, AoA=13.315

## Current interpretation

```json
{
  "complete_new_targets": [
    "selected_original_dup_all",
    "qwen_separated_pair"
  ],
  "mechanism_read": "aligned treatment retains a positive component beyond selected-original duplication and beyond separated coexistence, supporting same-window generated second-view correspondence as an active component under corrected AoA leaderboard units",
  "aligned_beats_selected_original_dup_all": true,
  "aligned_beats_separated_pair": true
}
```

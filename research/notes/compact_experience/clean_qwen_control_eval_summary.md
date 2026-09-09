# clean qwen control eval summary clean-Qwen control and replication evaluation

Summary JSON: `experiments/archive/compact_experience/data/control_eval_summary.json`

## Targets
| target | Overall | BLiMP | Supp | EWoK | Entity | COMPS | GPIQA | SuperGLUE | Reading | AoA leaderboard | AoA raw | ready |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| official_lengthmatched | 38.9404 | 67.29 | 59.04 | 50.8 | 23.31 | 53.13 | 36.68 | 67.9681 | 7.95 | -15.7045 | -0.157045 | True |
| qwen_clean_aligned | 41.3443 | 66.84 | 62.84 | 50.19 | 25.76 | 51.78 | 36.62 | 70.3086 | 7.76 | 0.0 | 0.0 | True |
| qwen_shuffled_control | 39.224 | 68.29 | 59.42 | 50.33 | 22.87 | 52.29 | 36.65 | 67.4352 | 8.52 | -12.7891 | -0.127891 | True |
| official_sourcematched | 39.6838 | 66.2 | 60.46 | 48.64 | 22.63 | 51.97 | 34.16 | 64.9438 | 8.14 | 0.0 | 0.0 | True |
| official_original_dup | 37.8266 | 65.52 | 60.72 | 48.0 | 18.61 | 52.15 | 31.65 | 68.8245 | 8.28 | -13.3155 | -0.133155 | True |
| official_lengthmatched_seed43122 | 38.9424 | 67.28 | 60.6 | 47.79 | 22.83 | 52.31 | 37.08 | 66.3688 | 7.95 | -11.7221 | -0.117221 | True |
| qwen_clean_aligned_seed43122 | 40.6501 | 66.02 | 61.51 | 50.43 | 25.26 | 52.06 | 34.62 | 68.7455 | 7.21 | 0.0 | 0.0 | True |

## Contrasts

- `seed43022_qwen_minus_lengthmatched` Overall Δ=2.4039; task Δs: BLiMP=-0.45, Supplement=3.8, EWoK=-0.61, Entity=2.45, COMPS=-1.35, GlobalPIQA=-0.06, SuperGLUE=2.34, Reading=-0.19, AoA=15.705
- `seed43022_qwen_minus_shuffled` Overall Δ=2.1203; task Δs: BLiMP=-1.45, Supplement=3.42, EWoK=-0.14, Entity=2.89, COMPS=-0.51, GlobalPIQA=-0.03, SuperGLUE=2.873, Reading=-0.76, AoA=12.789
- `seed43022_qwen_minus_sourcematched` Overall Δ=1.6605; task Δs: BLiMP=0.64, Supplement=2.38, EWoK=1.55, Entity=3.13, COMPS=-0.19, GlobalPIQA=2.455, SuperGLUE=5.365, Reading=-0.385, AoA=0.0
- `seed43022_qwen_minus_originaldup` Overall Δ=3.5177; task Δs: BLiMP=1.32, Supplement=2.12, EWoK=2.19, Entity=7.15, COMPS=-0.37, GlobalPIQA=4.97, SuperGLUE=1.484, Reading=-0.52, AoA=13.315
- `seed43122_qwen_minus_lengthmatched` Overall Δ=1.7076; task Δs: BLiMP=-1.26, Supplement=0.91, EWoK=2.64, Entity=2.43, COMPS=-0.25, GlobalPIQA=-2.46, SuperGLUE=2.377, Reading=-0.74, AoA=11.722
- `qwen_seed43122_minus_seed43022` Overall Δ=-0.6942; task Δs: BLiMP=-0.82, Supplement=-1.33, EWoK=0.24, Entity=-0.5, COMPS=0.28, GlobalPIQA=-2.0, SuperGLUE=-1.563, Reading=-0.555, AoA=0.0
- `official_seed43122_minus_seed43022` Overall Δ=0.002; task Δs: BLiMP=-0.01, Supplement=1.56, EWoK=-3.01, Entity=-0.48, COMPS=-0.82, GlobalPIQA=0.4, SuperGLUE=-1.599, Reading=-0.005, AoA=3.982

## Current interpretation

```json
{
  "first_seed_positive_over_0p25": true,
  "second_seed_positive_over_0p25": true,
  "pair_correspondence_survives_shuffled_control": true,
  "beyond_source_matching": true,
  "beyond_original_duplication": true,
  "mechanism_statement": "replicated positive effect with controls favors a semantic-pair/cross-view training signal component, now under corrected AoA leaderboard units; official-rule confirmation and leaderboard-level score remain separate requirements"
}
```

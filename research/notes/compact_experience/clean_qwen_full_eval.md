# clean qwen compliance and validity clean-Qwen full evaluation

Summary JSON: `experiments/archive/compact_experience/data/full_eval/full_eval_summary.json`

| target | Overall | BLiMP | Supp | EWoK | Entity | COMPS | GPIQA | SuperGLUE | Reading | AoA leaderboard | AoA raw | submit-ready AoA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| official_lengthmatched | 38.9404 | 67.29 | 59.04 | 50.8 | 23.31 | 53.13 | 36.68 | 67.9681 | 7.95 | -15.7045 | -0.157045 | True |
| official_lengthmatched_seed43122 | 38.9424 | 67.28 | 60.6 | 47.79 | 22.83 | 52.31 | 37.08 | 66.3688 | 7.95 | -11.7221 | -0.117221 | True |
| official_original_dup | 37.8266 | 65.52 | 60.72 | 48.0 | 18.61 | 52.15 | 31.65 | 68.8245 | 8.28 | -13.3155 | -0.133155 | True |
| official_sourcematched | 39.6838 | 66.2 | 60.46 | 48.64 | 22.63 | 51.97 | 34.16 | 64.9438 | 8.14 | 0.0 | 0.0 | True |
| qwen_clean_aligned | 41.3443 | 66.84 | 62.84 | 50.19 | 25.76 | 51.78 | 36.62 | 70.3086 | 7.76 | 0.0 | 0.0 | True |
| qwen_clean_aligned_seed43122 | 40.6501 | 66.02 | 61.51 | 50.43 | 25.26 | 52.06 | 34.62 | 68.7455 | 7.21 | 0.0 | 0.0 | True |
| qwen_shuffled_control | 39.224 | 68.29 | 59.42 | 50.33 | 22.87 | 52.29 | 36.65 | 67.4352 | 8.52 | -12.7891 | -0.127891 | True |

## qwen_clean_aligned minus official_lengthmatched

```json
{
  "BLiMP": -0.45,
  "Supplement": 3.8,
  "EWoK": -0.61,
  "Entity": 2.45,
  "COMPS": -1.35,
  "GlobalPIQA": -0.06,
  "SuperGLUE": 2.340479,
  "Reading": -0.19,
  "AoA": 15.704501,
  "Overall": 2.403887,
  "NLP_average": 0.874354,
  "Human_like_average": 7.757251
}
```

Decision: `continue_to_second_seed_and_shuffled_control`

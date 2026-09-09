# aoa unit correction AoA unit correction

Audit JSON: `experiments/archive/compact_experience/data/aoa_unit_correction/aoa_unit_correction_audit.json`

Official leaderboard arithmetic uses AoA in leaderboard units (`100 * raw correlation`). The previous local ledgers used raw AoA directly in Overall; this note records the corrected state.

## Regression

```json
{
  "self_test": {
    "leaderboard_row_calc": 40.62222222222223,
    "expected": 40.62,
    "raw_0p229_scaled": 22.900000000000002
  },
  "top20_checks": [
    {
      "Model_plain": "BabySteps_MurphysLaw-10M-mixed",
      "reported_overall": 40.86,
      "calculated_from_leaderboard_units": 40.85666666666666,
      "AoA_leaderboard_units": -15.1,
      "abs_error": 0.003333333333337407
    },
    {
      "Model_plain": "deberta-base-75k-sam_ext-s1",
      "reported_overall": 40.62,
      "calculated_from_leaderboard_units": 40.62222222222223,
      "AoA_leaderboard_units": 22.9,
      "abs_error": 0.002222222222229675
    },
    {
      "Model_plain": "instanton-baseline",
      "reported_overall": 40.31,
      "calculated_from_leaderboard_units": 40.315555555555555,
      "AoA_leaderboard_units": 20.2,
      "abs_error": 0.005555555555552871
    }
  ]
}
```

## Patched rows

| file | AoA before | AoA after | Overall before | Overall after | ΔOverall |
|---|---:|---:|---:|---:|---:|
| `experiments/archive/compact_experience/data/full_overall_eval/per_target/mix25_16k_seed43.json` | 0.0 | 0.0 | 41.4803 | 41.4803 | 0.0 |
| `experiments/archive/compact_experience/data/full_overall_eval/per_target/phase2s_mix25.json` | 0.0 | 0.0 | 40.5637 | 40.5637 | 0.0 |
| `experiments/archive/compact_experience/data/full_eval/per_target/official_lengthmatched.json` | -0.157 | -15.7045 | 40.6679 | 38.9404 | -1.7275 |
| `experiments/archive/compact_experience/data/full_eval/per_target/official_lengthmatched_seed43122.json` | -0.1172 | -11.7221 | 40.2318 | 38.9424 | -1.2894 |
| `experiments/archive/compact_experience/data/full_eval/per_target/official_original_dup.json` | -0.1332 | -13.3155 | 39.2913 | 37.8266 | -1.4647 |
| `experiments/archive/compact_experience/data/full_eval/per_target/official_sourcematched.json` | 0.0 | 0.0 | 39.6838 | 39.6838 | 0.0 |
| `experiments/archive/compact_experience/data/full_eval/per_target/qwen_clean_aligned.json` | 0.0 | 0.0 | 41.3443 | 41.3443 | 0.0 |
| `experiments/archive/compact_experience/data/full_eval/per_target/qwen_clean_aligned_seed43122.json` | 0.0 | 0.0 | 40.6501 | 40.6501 | 0.0 |
| `experiments/archive/compact_experience/data/full_eval/per_target/qwen_shuffled_control.json` | -0.1279 | -12.7891 | 40.6308 | 39.224 | -1.4068 |
| `experiments/archive/compact_experience/data/mechanism_eval/per_target/qwen_separated_pair.json` | 0.0 | 0.0 | 39.5643 | 39.5643 | 0.0 |
| `experiments/archive/compact_experience/data/mechanism_eval/per_target/selected_original_dup_all.json` | 0.0 | 0.0 | 40.9915 | 40.9915 | 0.0 |

## AoA leverage for current clean-Qwen endpoint

```json
{
  "current_overall": 41.34429066479573,
  "what_if_raw_aoa_0p229_leaderboard_22p9": 43.88873510924017,
  "delta": 2.5444444444444443
}
```

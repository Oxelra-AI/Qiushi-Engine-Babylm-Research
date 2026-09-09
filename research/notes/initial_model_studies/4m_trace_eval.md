# 4m trace eval — 4M paired BSM trace evaluation

JSON: `experiments/archive/initial_model_studies/data/4m_trace_eval.json`

**Checkpoint warning:** exact 1M/2M/3M checkpoint directories were overwritten by the trainer naming bug. Interpret `chck_1M/2M/3M` as late-in-bin available snapshots, not exact exposure checkpoints. Exact 250k/500k/750k/4M are valid.

## 4M official fast scores

| arm | BLiMP | Supplement | Entity | EWoK | GPIQA mean | Reading |
|---|---:|---:|---:|---:|---:|---:|
| official_standard | 55.38 | 46.40 | 17.77 | 49.91 | 35.725 | 6.79 |
| official_token_matched_reference | 54.25 | 46.40 | 17.99 | 50.45 | 37.195 | 6.71 |
| bsm_paired_coherent | 54.30 | 46.40 | 17.84 | 54.00 | 32.265 | 6.64 |
| bsm_paired_swapped | 53.97 | 46.40 | 17.88 | 50.91 | 34.240 | 6.67 |

## 4M deltas

```json
{
  "coherent_minus_swapped": {
    "BLiMP": 0.3299999999999983,
    "Supplement": 0.0,
    "Entity": -0.03999999999999915,
    "EWoK": 3.0900000000000034,
    "GlobalPIQA_parallel": -1.950000000000001,
    "GlobalPIQA_nonparallel": -2.0,
    "GlobalPIQA_mean": -1.9750000000000014,
    "Reading": -0.024999999999999467
  },
  "coherent_minus_reference": {
    "BLiMP": 0.04999999999999716,
    "Supplement": 0.0,
    "Entity": -0.14999999999999858,
    "EWoK": 3.549999999999997,
    "GlobalPIQA_parallel": -4.860000000000001,
    "GlobalPIQA_nonparallel": -5.0,
    "GlobalPIQA_mean": -4.93,
    "Reading": -0.0649999999999995
  },
  "swapped_minus_reference": {
    "BLiMP": -0.28000000000000114,
    "Supplement": 0.0,
    "Entity": -0.10999999999999943,
    "EWoK": 0.45999999999999375,
    "GlobalPIQA_parallel": -2.91,
    "GlobalPIQA_nonparallel": -3.0,
    "GlobalPIQA_mean": -2.9549999999999983,
    "Reading": -0.040000000000000036
  },
  "reference_minus_standard": {
    "BLiMP": -1.1300000000000026,
    "Supplement": 0.0,
    "Entity": 0.21999999999999886,
    "EWoK": 0.5400000000000063,
    "GlobalPIQA_parallel": 1.9400000000000013,
    "GlobalPIQA_nonparallel": 1.0,
    "GlobalPIQA_mean": 1.4699999999999989,
    "Reading": -0.08000000000000007
  }
}
```

## Continuous margin summary (train/natural)

| ckpt | arm | train margin | train both-correct | natural margin | natural both-correct |
|---|---|---:|---:|---:|---:|
| chck_250k | official_token_matched_reference | -0.0000 | 0.000 | -0.0000 | 0.000 |
| chck_250k | bsm_paired_coherent | -0.0000 | 0.000 | 0.0004 | 0.000 |
| chck_250k | bsm_paired_swapped | -0.0000 | 0.000 | 0.0085 | 0.025 |
| chck_500k | official_token_matched_reference | 0.0000 | 0.000 | 0.0000 | 0.000 |
| chck_500k | bsm_paired_coherent | -0.0000 | 0.000 | 0.0380 | 0.025 |
| chck_500k | bsm_paired_swapped | -0.0000 | 0.000 | -0.0000 | 0.000 |
| chck_750k | official_token_matched_reference | 0.0000 | 0.000 | -0.0000 | 0.000 |
| chck_750k | bsm_paired_coherent | 0.0000 | 0.000 | -0.0000 | 0.000 |
| chck_750k | bsm_paired_swapped | -0.0000 | 0.000 | 0.0000 | 0.000 |
| chck_1M | official_token_matched_reference | 0.0000 | 0.000 | -0.0000 | 0.000 |
| chck_1M | bsm_paired_coherent | 0.0000 | 0.000 | 0.0000 | 0.000 |
| chck_1M | bsm_paired_swapped | -0.0000 | 0.000 | 0.0000 | 0.000 |
| chck_2M | official_token_matched_reference | -0.0000 | 0.000 | -0.0000 | 0.000 |
| chck_2M | bsm_paired_coherent | 0.0000 | 0.000 | 0.0000 | 0.000 |
| chck_2M | bsm_paired_swapped | -0.0000 | 0.000 | 0.0000 | 0.000 |
| chck_3M | official_token_matched_reference | 0.0000 | 0.000 | 0.0000 | 0.000 |
| chck_3M | bsm_paired_coherent | -0.0000 | 0.000 | 0.0000 | 0.000 |
| chck_3M | bsm_paired_swapped | 0.0000 | 0.000 | -0.0000 | 0.000 |
| chck_4M | official_token_matched_reference | 0.0000 | 0.000 | 0.0000 | 0.000 |
| chck_4M | bsm_paired_coherent | 0.0000 | 0.000 | 0.0000 | 0.000 |
| chck_4M | bsm_paired_swapped | -0.0000 | 0.000 | 0.0000 | 0.000 |

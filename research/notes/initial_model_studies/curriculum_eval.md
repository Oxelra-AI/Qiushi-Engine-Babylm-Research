# curriculum eval — Replicated same-content curriculum-order evaluation

JSON: `experiments/archive/initial_model_studies/data/curriculum_eval.json`

All arms are the same 4M content, trained with `--no_shuffle`; only file order differs.

## 4M endpoint scores

| arm | BLiMP | Supplement | EWoK | Entity | COMPS | GlobalPIQA mean | Reading |
|---|---:|---:|---:|---:|---:|---:|---:|
| random_a | 54.370 | 50.400 | 47.730 | 18.650 | 50.110 | 36.195 | 6.850 |
| random_b | 53.610 | 50.800 | 50.640 | 18.300 | 49.920 | 32.750 | 6.765 |
| random_mean | 53.990 | 50.600 | 49.185 | 18.475 | 50.015 | 34.472 | 6.807 |
| curriculum | 54.490 | 48.800 | 50.000 | 18.680 | 49.890 | 31.265 | 7.045 |

## Deltas and weighted proxy

| contrast | BLiMP | Supp | EWoK | Entity | COMPS | GPIQA | Reading | proxy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| curriculum_minus_random_a | 0.120 | -1.600 | 2.270 | 0.030 | -0.220 | -4.930 | 0.195 | -0.440 |
| curriculum_minus_random_b | 0.880 | -2.000 | -0.640 | 0.380 | -0.030 | -1.485 | 0.280 | -0.275 |
| curriculum_minus_random_mean | 0.500 | -1.800 | 0.815 | 0.205 | -0.125 | -3.207 | 0.238 | -0.357 |
| random_b_minus_random_a | -0.760 | 0.400 | 2.910 | -0.350 | -0.190 | -3.445 | -0.085 | -0.164 |

Proxy = (3/28) * delta(BLiMP+Supplement+EWoK+Entity+COMPS+GlobalPIQA_mean) + (1/8) * delta(Reading). SuperGLUE/AoA are not evaluated in this fast screen.

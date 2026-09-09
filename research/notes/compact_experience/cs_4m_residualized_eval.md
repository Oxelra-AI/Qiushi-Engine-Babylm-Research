# cs 4m residualized eval — Residualized C/S 4M fast evaluation

Summary JSON: `experiments/archive/compact_experience/data/cs_4m_eval/cs_4m_residualized_eval_summary.json`

This is a mechanism-screen evaluation, not official Overall: SuperGLUE and AoA are absent, fast subsets are used for most zero-shot columns, and COMPS is evaluated from the full COMPS set because no fast COMPS directory exists in this repo snapshot.

## Scores

| arm | BLiMP | Supplement | EWoK | Entity | COMPS | GPIQA mean | Reading |
|---|---:|---:|---:|---:|---:|---:|---:|
| r_strat_a | 53.950 | 49.600 | 50.450 | 17.950 | 49.880 | 34.240 | 6.860 |
| r_strat_b | 52.790 | 49.200 | 49.090 | 17.890 | 50.280 | 36.180 | 6.925 |
| random_mean | 53.370 | 49.400 | 49.770 | 17.920 | 50.080 | 35.210 | 6.893 |
| c_unique_control | 54.270 | 49.600 | 49.730 | 17.580 | 50.480 | 33.240 | 6.745 |
| c_unique_high | 54.220 | 49.200 | 48.820 | 16.410 | 50.090 | 32.280 | 6.720 |
| s_unique_control | 52.820 | 45.200 | 48.820 | 17.220 | 49.750 | 36.210 | 7.125 |
| s_unique_high | 53.630 | 47.200 | 48.270 | 17.500 | 50.090 | 33.265 | 6.990 |

## Deltas

| contrast | BLiMP | Supp | EWoK | Entity | COMPS | GPIQA | Reading | weighted fast proxy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| c_high_minus_c_control | -0.050 | -0.400 | -0.910 | -1.170 | -0.390 | -0.960 | -0.025 | -0.419 |
| s_high_minus_s_control | 0.810 | 2.000 | -0.550 | 0.280 | 0.340 | -2.945 | -0.135 | -0.024 |
| c_high_minus_random_mean | 0.850 | -0.200 | -0.950 | -1.510 | 0.010 | -2.930 | -0.172 | -0.528 |
| s_high_minus_random_mean | 0.260 | -2.200 | -1.500 | -0.420 | 0.010 | -1.945 | 0.098 | -0.609 |
| c_control_minus_random_mean | 0.900 | 0.200 | -0.040 | -0.340 | 0.400 | -1.970 | -0.148 | -0.110 |
| s_control_minus_random_mean | -0.550 | -4.200 | -0.950 | -0.700 | -0.330 | 1.000 | 0.232 | -0.585 |
| r_b_minus_r_a | -1.160 | -0.400 | -1.360 | -0.060 | 0.400 | 1.940 | 0.065 | -0.060 |

Weighted fast proxy = (3/28) * delta(BLiMP+Supplement+EWoK+Entity+COMPS+GlobalPIQA_mean) + (1/8) * delta(Reading). SuperGLUE and AoA are not included.

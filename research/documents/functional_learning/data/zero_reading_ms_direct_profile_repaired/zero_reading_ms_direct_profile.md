# clean replication and ms direct pending direct zero-shot/Reading profile: clean preservation vs exact `(M,S)`

This parses actual official zero-shot/Reading payloads. It includes coherent86, dense `(M,M)` seeds, exact acquisition-only `(M,S)` seed62064, and clean preservation seeds 62064/62065.

## Seven-component zero/Reading sums

- coherent86: 309.270000
- dense_seed62064_MM: 310.810000
- dense_seed62065_MM: 310.710000
- ms_acquisition_seed62064_MS: 310.935000
- clean_pres_seed62064_MSplusKL: 311.170000
- clean_pres_seed62065_MSplusKL: 311.065000
- clean64 minus exact `(M,S)` zero/Reading sum: 0.235000 = 0.026111 Overall units before SuperGLUE/AoA
- clean65 minus exact `(M,S)` zero/Reading sum: 0.130000 = 0.014444 Overall units before SuperGLUE/AoA

## Component deltas

| component | coherent86 | exact `(M,S)` | clean64 | clean65 | clean64-`(M,S)` | clean65-`(M,S)` | clean65-clean64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| BLiMP | 68.5100 | 68.1200 | 68.2600 | 68.2200 | 0.1400 | 0.1000 | -0.0400 |
| Supplement | 63.6400 | 63.0800 | 63.2800 | 63.2900 | 0.2000 | 0.2100 | 0.0100 |
| EWoK | 50.0200 | 49.9500 | 49.8200 | 49.7200 | -0.1300 | -0.2300 | -0.1000 |
| Entity | 28.3200 | 29.3900 | 29.4000 | 29.4500 | 0.0100 | 0.0600 | 0.0500 |
| COMPS | 52.0500 | 52.1500 | 52.1600 | 52.1400 | 0.0100 | -0.0100 | -0.0200 |
| GlobalPIQA | 38.5650 | 40.0500 | 40.0500 | 40.0500 | 0.0000 | 0.0000 | 0.0000 |
| Reading | 8.1650 | 8.1950 | 8.2000 | 8.1950 | 0.0050 | 0.0000 | -0.0050 |

## Item-level direct comparisons for zero-shot columns

| column | clean64-vs-`(M,S)` correctness agree | clean64 correct/`(M,S)` wrong | `(M,S)` correct/clean64 wrong | net clean64-`(M,S)` pp | clean65-vs-`(M,S)` correctness agree | clean65 correct/`(M,S)` wrong | `(M,S)` correct/clean65 wrong | net clean65-`(M,S)` pp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BLiMP | 0.9907 | 321 | 233 | 0.1470 | 0.9903 | 321 | 258 | 0.1052 |
| Supplement | 0.9929 | 19 | 18 | 0.0192 | 0.9931 | 18 | 18 | 0.0000 |
| EWoK | 0.9776 | 78 | 93 | -0.1969 | 0.9782 | 74 | 92 | -0.2363 |
| Entity | 0.9854 | 50 | 49 | 0.0147 | 0.9854 | 52 | 47 | 0.0737 |
| COMPS | 0.9773 | 1046 | 1023 | 0.0253 | 0.9765 | 1072 | 1066 | 0.0066 |
| GlobalPIQA_parallel | 1.0000 | 0 | 0 | 0.0000 | 1.0000 | 0 | 0 | 0.0000 |
| GlobalPIQA_nonparallel | 1.0000 | 0 | 0 | 0.0000 | 1.0000 | 0 | 0 | 0.0000 |

## Parent / `(M,S)` / clean decomposition

| column | clean seed | recover `(M,S)` parent loss | lose parent item `(M,S)` kept | keep `(M,S)` gain | drop `(M,S)` gain | new gain beyond `(M,S)` | shared loss | net clean-`(M,S)` pp |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| BLiMP | 62064 | 308 | 16 | 565 | 217 | 13 | 711 | 0.1470 |
| BLiMP | 62065 | 309 | 25 | 549 | 233 | 12 | 710 | 0.1052 |
| Supplement | 62064 | 18 | 1 | 37 | 17 | 1 | 36 | 0.0192 |
| Supplement | 62065 | 17 | 0 | 36 | 18 | 1 | 37 | 0.0000 |
| EWoK | 62064 | 68 | 10 | 153 | 83 | 10 | 154 | -0.1969 |
| EWoK | 62065 | 66 | 13 | 157 | 79 | 8 | 156 | -0.2363 |
| Entity | 62064 | 38 | 8 | 174 | 41 | 12 | 135 | 0.0147 |
| Entity | 62065 | 41 | 6 | 174 | 41 | 11 | 132 | 0.0737 |
| COMPS | 62064 | 982 | 76 | 2544 | 947 | 64 | 2387 | 0.0253 |
| COMPS | 62065 | 1005 | 76 | 2501 | 990 | 67 | 2364 | 0.0066 |
| GlobalPIQA_parallel | 62064 | 0 | 0 | 2 | 0 | 0 | 1 | 0.0000 |
| GlobalPIQA_parallel | 62065 | 0 | 0 | 2 | 0 | 0 | 1 | 0.0000 |
| GlobalPIQA_nonparallel | 62064 | 0 | 0 | 2 | 0 | 0 | 0 | 0.0000 |
| GlobalPIQA_nonparallel | 62065 | 0 | 0 | 2 | 0 | 0 | 0 | 0.0000 |

## Reading comparisons

- clean_pres_seed62064_MSplusKL_vs_ms_acquisition_seed62064_MS: {"n": 1726, "pred_mean_abs_diff": 0.07536332613527545, "pred_mean_signed_diff_a_minus_b": -0.048535984750297675, "pred_pearson": 0.9998382563300703, "prev_pred_mean_abs_diff": 0.06141692705604427, "prev_pred_mean_signed_diff_a_minus_b": -0.031274374400149775, "prev_pred_pearson": 0.9997350586197238, "score_a": 8.2, "score_b": 8.195, "score_delta_a_minus_b": 0.004999999999999005}
- clean_pres_seed62065_MSplusKL_vs_ms_acquisition_seed62064_MS: {"n": 1726, "pred_mean_abs_diff": 0.0764168072852784, "pred_mean_signed_diff_a_minus_b": -0.049315590424124214, "pred_pearson": 0.9998306476554961, "prev_pred_mean_abs_diff": 0.06218570024315063, "prev_pred_mean_signed_diff_a_minus_b": -0.03176769622022627, "prev_pred_pearson": 0.9997247343645003, "score_a": 8.195, "score_b": 8.195, "score_delta_a_minus_b": 0.0}
- clean_pres_seed62064_MSplusKL_vs_clean_pres_seed62065_MSplusKL: {"n": 1726, "pred_mean_abs_diff": 0.01187593425854518, "pred_mean_signed_diff_a_minus_b": 0.0007796056738265481, "pred_pearson": 0.9999949901599272, "prev_pred_mean_abs_diff": 0.011319091025221302, "prev_pred_mean_signed_diff_a_minus_b": 0.0004933218200764976, "prev_pred_pearson": 0.9999907847007538, "score_a": 8.2, "score_b": 8.195, "score_delta_a_minus_b": 0.004999999999999005}

## Scientific reading

On the seven zero-shot/Reading components, exact `(M,S)` is already close to clean. Clean seed62064 gains +0.235 component-sum over `(M,S)`, or +0.026111 Overall units before SuperGLUE, mainly through BLiMP and Supplement, while losing EWoK and leaving GlobalPIQA unchanged. Clean seed62065 gains only +0.130 component-sum over `(M,S)`, or +0.014444 Overall units before SuperGLUE. Combining these with the completed SuperGLUE profile explains the final same-coordinate table: preservation's direct Overall advantage over exact `(M,S)` is real but small for seed62064, and the replicated clean-over-parent result is stronger than the direct clean-over-`(M,S)` increment. The scientific interpretation should remain an acquisition-retention improvement with modest preservation-specific score value, not a broad monotonic upgrade.

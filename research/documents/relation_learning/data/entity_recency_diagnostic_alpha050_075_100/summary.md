# earlier analysis Entity last-operation recency diagnostic

`rec_avail_pick` is the percent of rows where the prediction equals the final state of a box affected by the last operation, among rows where that state appears as an official option. `rec_not_gold_pick` restricts to cases where such a recency option is present but is not the gold answer.

## Summary

| label | group | n | acc | rec_avail_n | rec_avail_pick | rec_not_gold_n | rec_not_gold_pick |
|---|---|---:|---:|---:|---:|---:|---:|
| binding_ep25_alpha0p50 | ALL | 6780 | 26.73 | 3197 | 35.78 | 1528 | 35.60 |
| binding_ep25_alpha0p50 | lastop_recency_gold | 1850 | 32.43 | 1850 | 34.97 | 181 | 25.97 |
| binding_ep25_alpha0p50 | lastop_recency_not_gold | 1347 | 20.27 | 1347 | 36.90 | 1347 | 36.90 |
| binding_ep25_alpha0p50 | lastop_recency_not_gold_available | 1528 | 21.20 | 1528 | 38.94 | 1528 | 35.60 |
| binding_ep25_alpha0p50 | rel_eq0 | 1541 | 30.37 | 321 | 28.97 | 321 | 28.97 |
| binding_ep25_alpha0p50 | rel_eq0_irrelevant_ops_1to3 | 641 | 35.57 | 166 | 18.07 | 166 | 18.07 |
| binding_ep25_alpha0p50 | rel_eq0_irrelevant_ops_4to6 | 323 | 28.48 | 76 | 32.89 | 76 | 32.89 |
| binding_ep25_alpha0p50 | rel_eq0_irrelevant_ops_ge7 | 273 | 15.38 | 79 | 48.10 | 79 | 48.10 |
| binding_ep25_alpha0p50 | rel_eq0_irrelevant_ops_gt0 | 1237 | 29.26 | 321 | 28.97 | 321 | 28.97 |
| binding_ep25_alpha0p50 | rel_ge1 | 5239 | 25.65 | 2876 | 36.54 | 1207 | 37.37 |
| binding_ep25_alpha0p50 | rel_ge1_postrel_ops0 | 1850 | 32.43 | 1850 | 34.97 | 181 | 25.97 |
| binding_ep25_alpha0p50 | rel_ge1_postrel_ops_gt0 | 3389 | 21.95 | 1026 | 39.38 | 1026 | 39.38 |
| binding_ep25_alpha0p50 | rel_ge3_postrel_ops0 | 1114 | 36.71 | 1114 | 39.59 | 115 | 27.83 |
| binding_ep25_alpha0p50 | rel_ge3_postrel_ops_gt0 | 1536 | 26.30 | 439 | 39.86 | 439 | 39.86 |
| binding_ep25_alpha0p50 | stale_available_not_gold | 1221 | 29.57 | 681 | 37.74 | 275 | 31.64 |
| binding_ep25_alpha0p75 | ALL | 6780 | 25.55 | 3197 | 34.97 | 1528 | 33.90 |
| binding_ep25_alpha0p75 | lastop_recency_gold | 1850 | 32.43 | 1850 | 35.30 | 181 | 29.28 |
| binding_ep25_alpha0p75 | lastop_recency_not_gold | 1347 | 19.67 | 1347 | 34.52 | 1347 | 34.52 |
| binding_ep25_alpha0p75 | lastop_recency_not_gold_available | 1528 | 21.01 | 1528 | 37.57 | 1528 | 33.90 |
| binding_ep25_alpha0p75 | rel_eq0 | 1541 | 26.22 | 321 | 29.60 | 321 | 29.60 |
| binding_ep25_alpha0p75 | rel_eq0_irrelevant_ops_1to3 | 641 | 31.83 | 166 | 18.07 | 166 | 18.07 |
| binding_ep25_alpha0p75 | rel_eq0_irrelevant_ops_4to6 | 323 | 21.67 | 76 | 38.16 | 76 | 38.16 |
| binding_ep25_alpha0p75 | rel_eq0_irrelevant_ops_ge7 | 273 | 14.29 | 79 | 45.57 | 79 | 45.57 |
| binding_ep25_alpha0p75 | rel_eq0_irrelevant_ops_gt0 | 1237 | 25.30 | 321 | 29.60 | 321 | 29.60 |
| binding_ep25_alpha0p75 | rel_ge1 | 5239 | 25.35 | 2876 | 35.57 | 1207 | 35.05 |
| binding_ep25_alpha0p75 | rel_ge1_postrel_ops0 | 1850 | 32.43 | 1850 | 35.30 | 181 | 29.28 |
| binding_ep25_alpha0p75 | rel_ge1_postrel_ops_gt0 | 3389 | 21.48 | 1026 | 36.06 | 1026 | 36.06 |
| binding_ep25_alpha0p75 | rel_ge3_postrel_ops0 | 1114 | 36.18 | 1114 | 39.23 | 115 | 29.57 |
| binding_ep25_alpha0p75 | rel_ge3_postrel_ops_gt0 | 1536 | 25.85 | 439 | 37.13 | 439 | 37.13 |
| binding_ep25_alpha0p75 | stale_available_not_gold | 1221 | 29.16 | 681 | 36.42 | 275 | 29.82 |
| binding_ep25_alpha1p00 | ALL | 6780 | 22.48 | 3197 | 31.15 | 1528 | 30.30 |
| binding_ep25_alpha1p00 | lastop_recency_gold | 1850 | 28.81 | 1850 | 31.30 | 181 | 25.41 |
| binding_ep25_alpha1p00 | lastop_recency_not_gold | 1347 | 18.93 | 1347 | 30.96 | 1347 | 30.96 |
| binding_ep25_alpha1p00 | lastop_recency_not_gold_available | 1528 | 20.42 | 1528 | 34.03 | 1528 | 30.30 |
| binding_ep25_alpha1p00 | rel_eq0 | 1541 | 21.22 | 321 | 24.30 | 321 | 24.30 |
| binding_ep25_alpha1p00 | rel_eq0_irrelevant_ops_1to3 | 641 | 26.05 | 166 | 15.06 | 166 | 15.06 |
| binding_ep25_alpha1p00 | rel_eq0_irrelevant_ops_4to6 | 323 | 14.55 | 76 | 27.63 | 76 | 27.63 |
| binding_ep25_alpha1p00 | rel_eq0_irrelevant_ops_ge7 | 273 | 15.02 | 79 | 40.51 | 79 | 40.51 |
| binding_ep25_alpha1p00 | rel_eq0_irrelevant_ops_gt0 | 1237 | 20.61 | 321 | 24.30 | 321 | 24.30 |
| binding_ep25_alpha1p00 | rel_ge1 | 5239 | 22.85 | 2876 | 31.92 | 1207 | 31.90 |
| binding_ep25_alpha1p00 | rel_ge1_postrel_ops0 | 1850 | 28.81 | 1850 | 31.30 | 181 | 25.41 |
| binding_ep25_alpha1p00 | rel_ge1_postrel_ops_gt0 | 3389 | 19.59 | 1026 | 33.04 | 1026 | 33.04 |
| binding_ep25_alpha1p00 | rel_ge3_postrel_ops0 | 1114 | 31.69 | 1114 | 34.29 | 115 | 25.22 |
| binding_ep25_alpha1p00 | rel_ge3_postrel_ops_gt0 | 1536 | 23.31 | 439 | 32.80 | 439 | 32.80 |
| binding_ep25_alpha1p00 | stale_available_not_gold | 1221 | 25.47 | 681 | 32.45 | 275 | 28.73 |
| chck82 | ALL | 6780 | 27.89 | 3197 | 32.66 | 1528 | 31.54 |
| chck82 | lastop_recency_gold | 1850 | 30.38 | 1850 | 32.59 | 181 | 22.65 |
| chck82 | lastop_recency_not_gold | 1347 | 24.72 | 1347 | 32.74 | 1347 | 32.74 |
| chck82 | lastop_recency_not_gold_available | 1528 | 24.54 | 1528 | 34.29 | 1528 | 31.54 |
| chck82 | rel_eq0 | 1541 | 37.44 | 321 | 23.05 | 321 | 23.05 |
| chck82 | rel_eq0_irrelevant_ops_1to3 | 641 | 39.00 | 166 | 16.27 | 166 | 16.27 |
| chck82 | rel_eq0_irrelevant_ops_4to6 | 323 | 43.03 | 76 | 27.63 | 76 | 27.63 |
| chck82 | rel_eq0_irrelevant_ops_ge7 | 273 | 29.67 | 79 | 32.91 | 79 | 32.91 |
| chck82 | rel_eq0_irrelevant_ops_gt0 | 1237 | 38.00 | 321 | 23.05 | 321 | 23.05 |
| chck82 | rel_ge1 | 5239 | 25.08 | 2876 | 33.73 | 1207 | 33.80 |
| chck82 | rel_ge1_postrel_ops0 | 1850 | 30.38 | 1850 | 32.59 | 181 | 22.65 |
| chck82 | rel_ge1_postrel_ops_gt0 | 3389 | 22.19 | 1026 | 35.77 | 1026 | 35.77 |
| chck82 | rel_ge3_postrel_ops0 | 1114 | 34.56 | 1114 | 37.43 | 115 | 27.83 |
| chck82 | rel_ge3_postrel_ops_gt0 | 1536 | 26.95 | 439 | 35.31 | 439 | 35.31 |
| chck82 | stale_available_not_gold | 1221 | 27.44 | 681 | 34.95 | 275 | 32.00 |
| coherent86 | ALL | 6780 | 27.83 | 3197 | 32.87 | 1528 | 31.94 |
| coherent86 | lastop_recency_gold | 1850 | 30.43 | 1850 | 32.81 | 181 | 24.31 |
| coherent86 | lastop_recency_not_gold | 1347 | 24.72 | 1347 | 32.96 | 1347 | 32.96 |
| coherent86 | lastop_recency_not_gold_available | 1528 | 24.61 | 1528 | 34.75 | 1528 | 31.94 |
| coherent86 | rel_eq0 | 1541 | 37.18 | 321 | 23.99 | 321 | 23.99 |
| coherent86 | rel_eq0_irrelevant_ops_1to3 | 641 | 38.69 | 166 | 16.27 | 166 | 16.27 |
| coherent86 | rel_eq0_irrelevant_ops_4to6 | 323 | 42.11 | 76 | 28.95 | 76 | 28.95 |
| coherent86 | rel_eq0_irrelevant_ops_ge7 | 273 | 29.67 | 79 | 35.44 | 79 | 35.44 |
| coherent86 | rel_eq0_irrelevant_ops_gt0 | 1237 | 37.59 | 321 | 23.99 | 321 | 23.99 |
| coherent86 | rel_ge1 | 5239 | 25.08 | 2876 | 33.87 | 1207 | 34.05 |
| coherent86 | rel_ge1_postrel_ops0 | 1850 | 30.43 | 1850 | 32.81 | 181 | 24.31 |
| coherent86 | rel_ge1_postrel_ops_gt0 | 3389 | 22.16 | 1026 | 35.77 | 1026 | 35.77 |
| coherent86 | rel_ge3_postrel_ops0 | 1114 | 34.29 | 1114 | 37.43 | 115 | 30.43 |
| coherent86 | rel_ge3_postrel_ops_gt0 | 1536 | 26.89 | 439 | 35.08 | 439 | 35.08 |
| coherent86 | stale_available_not_gold | 1221 | 27.35 | 681 | 34.80 | 275 | 31.64 |

## Deltas vs baseline

Baseline label: `chck82`

| label | group | n | d_acc | d_rec_avail_pick | d_rec_not_gold_pick |
|---|---|---:|---:|---:|---:|
| binding_ep25_alpha0p50 | ALL | 6780 | -1.17 | 3.13 | 4.06 |
| binding_ep25_alpha0p50 | lastop_recency_gold | 1850 | 2.05 | 2.38 | 3.31 |
| binding_ep25_alpha0p50 | lastop_recency_not_gold | 1347 | -4.45 | 4.16 | 4.16 |
| binding_ep25_alpha0p50 | lastop_recency_not_gold_available | 1528 | -3.34 | 4.65 | 4.06 |
| binding_ep25_alpha0p50 | rel_eq0 | 1541 | -7.07 | 5.92 | 5.92 |
| binding_ep25_alpha0p50 | rel_eq0_irrelevant_ops_1to3 | 641 | -3.43 | 1.81 | 1.81 |
| binding_ep25_alpha0p50 | rel_eq0_irrelevant_ops_4to6 | 323 | -14.55 | 5.26 | 5.26 |
| binding_ep25_alpha0p50 | rel_eq0_irrelevant_ops_ge7 | 273 | -14.29 | 15.19 | 15.19 |
| binding_ep25_alpha0p50 | rel_eq0_irrelevant_ops_gt0 | 1237 | -8.73 | 5.92 | 5.92 |
| binding_ep25_alpha0p50 | rel_ge1 | 5239 | 0.57 | 2.82 | 3.56 |
| binding_ep25_alpha0p50 | rel_ge1_postrel_ops0 | 1850 | 2.05 | 2.38 | 3.31 |
| binding_ep25_alpha0p50 | rel_ge1_postrel_ops_gt0 | 3389 | -0.24 | 3.61 | 3.61 |
| binding_ep25_alpha0p50 | rel_ge3_postrel_ops0 | 1114 | 2.15 | 2.15 | 0.00 |
| binding_ep25_alpha0p50 | rel_ge3_postrel_ops_gt0 | 1536 | -0.65 | 4.56 | 4.56 |
| binding_ep25_alpha0p50 | stale_available_not_gold | 1221 | 2.13 | 2.79 | -0.36 |
| binding_ep25_alpha0p75 | ALL | 6780 | -2.35 | 2.31 | 2.36 |
| binding_ep25_alpha0p75 | lastop_recency_gold | 1850 | 2.05 | 2.70 | 6.63 |
| binding_ep25_alpha0p75 | lastop_recency_not_gold | 1347 | -5.05 | 1.78 | 1.78 |
| binding_ep25_alpha0p75 | lastop_recency_not_gold_available | 1528 | -3.53 | 3.27 | 2.36 |
| binding_ep25_alpha0p75 | rel_eq0 | 1541 | -11.23 | 6.54 | 6.54 |
| binding_ep25_alpha0p75 | rel_eq0_irrelevant_ops_1to3 | 641 | -7.18 | 1.81 | 1.81 |
| binding_ep25_alpha0p75 | rel_eq0_irrelevant_ops_4to6 | 323 | -21.36 | 10.53 | 10.53 |
| binding_ep25_alpha0p75 | rel_eq0_irrelevant_ops_ge7 | 273 | -15.38 | 12.66 | 12.66 |
| binding_ep25_alpha0p75 | rel_eq0_irrelevant_ops_gt0 | 1237 | -12.69 | 6.54 | 6.54 |
| binding_ep25_alpha0p75 | rel_ge1 | 5239 | 0.27 | 1.84 | 1.24 |
| binding_ep25_alpha0p75 | rel_ge1_postrel_ops0 | 1850 | 2.05 | 2.70 | 6.63 |
| binding_ep25_alpha0p75 | rel_ge1_postrel_ops_gt0 | 3389 | -0.71 | 0.29 | 0.29 |
| binding_ep25_alpha0p75 | rel_ge3_postrel_ops0 | 1114 | 1.62 | 1.80 | 1.74 |
| binding_ep25_alpha0p75 | rel_ge3_postrel_ops_gt0 | 1536 | -1.11 | 1.82 | 1.82 |
| binding_ep25_alpha0p75 | stale_available_not_gold | 1221 | 1.72 | 1.47 | -2.18 |
| binding_ep25_alpha1p00 | ALL | 6780 | -5.41 | -1.50 | -1.24 |
| binding_ep25_alpha1p00 | lastop_recency_gold | 1850 | -1.57 | -1.30 | 2.76 |
| binding_ep25_alpha1p00 | lastop_recency_not_gold | 1347 | -5.79 | -1.78 | -1.78 |
| binding_ep25_alpha1p00 | lastop_recency_not_gold_available | 1528 | -4.12 | -0.26 | -1.24 |
| binding_ep25_alpha1p00 | rel_eq0 | 1541 | -16.22 | 1.25 | 1.25 |
| binding_ep25_alpha1p00 | rel_eq0_irrelevant_ops_1to3 | 641 | -12.95 | -1.20 | -1.20 |
| binding_ep25_alpha1p00 | rel_eq0_irrelevant_ops_4to6 | 323 | -28.48 | 0.00 | 0.00 |
| binding_ep25_alpha1p00 | rel_eq0_irrelevant_ops_ge7 | 273 | -14.65 | 7.59 | 7.59 |
| binding_ep25_alpha1p00 | rel_eq0_irrelevant_ops_gt0 | 1237 | -17.38 | 1.25 | 1.25 |
| binding_ep25_alpha1p00 | rel_ge1 | 5239 | -2.23 | -1.81 | -1.91 |
| binding_ep25_alpha1p00 | rel_ge1_postrel_ops0 | 1850 | -1.57 | -1.30 | 2.76 |
| binding_ep25_alpha1p00 | rel_ge1_postrel_ops_gt0 | 3389 | -2.60 | -2.73 | -2.73 |
| binding_ep25_alpha1p00 | rel_ge3_postrel_ops0 | 1114 | -2.87 | -3.14 | -2.61 |
| binding_ep25_alpha1p00 | rel_ge3_postrel_ops_gt0 | 1536 | -3.65 | -2.51 | -2.51 |
| binding_ep25_alpha1p00 | stale_available_not_gold | 1221 | -1.97 | -2.50 | -3.27 |
| coherent86 | ALL | 6780 | -0.06 | 0.22 | 0.39 |
| coherent86 | lastop_recency_gold | 1850 | 0.05 | 0.22 | 1.66 |
| coherent86 | lastop_recency_not_gold | 1347 | 0.00 | 0.22 | 0.22 |
| coherent86 | lastop_recency_not_gold_available | 1528 | 0.07 | 0.46 | 0.39 |
| coherent86 | rel_eq0 | 1541 | -0.26 | 0.93 | 0.93 |
| coherent86 | rel_eq0_irrelevant_ops_1to3 | 641 | -0.31 | 0.00 | 0.00 |
| coherent86 | rel_eq0_irrelevant_ops_4to6 | 323 | -0.93 | 1.32 | 1.32 |
| coherent86 | rel_eq0_irrelevant_ops_ge7 | 273 | 0.00 | 2.53 | 2.53 |
| coherent86 | rel_eq0_irrelevant_ops_gt0 | 1237 | -0.40 | 0.93 | 0.93 |
| coherent86 | rel_ge1 | 5239 | 0.00 | 0.14 | 0.25 |
| coherent86 | rel_ge1_postrel_ops0 | 1850 | 0.05 | 0.22 | 1.66 |
| coherent86 | rel_ge1_postrel_ops_gt0 | 3389 | -0.03 | 0.00 | 0.00 |
| coherent86 | rel_ge3_postrel_ops0 | 1114 | -0.27 | 0.00 | 2.61 |
| coherent86 | rel_ge3_postrel_ops_gt0 | 1536 | -0.07 | -0.23 | -0.23 |
| coherent86 | stale_available_not_gold | 1221 | -0.08 | -0.15 | -0.36 |

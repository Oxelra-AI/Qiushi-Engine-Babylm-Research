# earlier analysis Entity last-operation recency diagnostic

`rec_avail_pick` is the percent of rows where the prediction equals the final state of a box affected by the last operation, among rows where that state appears as an official option. `rec_not_gold_pick` restricts to cases where such a recency option is present but is not the gold answer.

## Summary

| label | group | n | acc | rec_avail_n | rec_avail_pick | rec_not_gold_n | rec_not_gold_pick |
|---|---|---:|---:|---:|---:|---:|---:|
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
| tight_private_scale0p2 | ALL | 6780 | 24.38 | 3197 | 19.05 | 1528 | 18.59 |
| tight_private_scale0p2 | lastop_recency_gold | 1850 | 17.57 | 1850 | 19.57 | 181 | 20.44 |
| tight_private_scale0p2 | lastop_recency_not_gold | 1347 | 24.50 | 1347 | 18.34 | 1347 | 18.34 |
| tight_private_scale0p2 | lastop_recency_not_gold_available | 1528 | 23.43 | 1528 | 20.42 | 1528 | 18.59 |
| tight_private_scale0p2 | rel_eq0 | 1541 | 45.75 | 321 | 12.77 | 321 | 12.77 |
| tight_private_scale0p2 | rel_eq0_irrelevant_ops_1to3 | 641 | 47.11 | 166 | 10.24 | 166 | 10.24 |
| tight_private_scale0p2 | rel_eq0_irrelevant_ops_4to6 | 323 | 45.82 | 76 | 9.21 | 76 | 9.21 |
| tight_private_scale0p2 | rel_eq0_irrelevant_ops_ge7 | 273 | 37.36 | 79 | 21.52 | 79 | 21.52 |
| tight_private_scale0p2 | rel_eq0_irrelevant_ops_gt0 | 1237 | 44.62 | 321 | 12.77 | 321 | 12.77 |
| tight_private_scale0p2 | rel_ge1 | 5239 | 18.10 | 2876 | 19.75 | 1207 | 20.13 |
| tight_private_scale0p2 | rel_ge1_postrel_ops0 | 1850 | 17.57 | 1850 | 19.57 | 181 | 20.44 |
| tight_private_scale0p2 | rel_ge1_postrel_ops_gt0 | 3389 | 18.38 | 1026 | 20.08 | 1026 | 20.08 |
| tight_private_scale0p2 | rel_ge3_postrel_ops0 | 1114 | 18.67 | 1114 | 21.18 | 115 | 24.35 |
| tight_private_scale0p2 | rel_ge3_postrel_ops_gt0 | 1536 | 18.49 | 439 | 23.01 | 439 | 23.01 |
| tight_private_scale0p2 | stale_available_not_gold | 1221 | 15.48 | 681 | 18.80 | 275 | 20.36 |

## Deltas vs baseline

Baseline label: `coherent86`

| label | group | n | d_acc | d_rec_avail_pick | d_rec_not_gold_pick |
|---|---|---:|---:|---:|---:|
| chck82 | ALL | 6780 | 0.06 | -0.22 | -0.39 |
| chck82 | lastop_recency_gold | 1850 | -0.05 | -0.22 | -1.66 |
| chck82 | lastop_recency_not_gold | 1347 | 0.00 | -0.22 | -0.22 |
| chck82 | lastop_recency_not_gold_available | 1528 | -0.07 | -0.46 | -0.39 |
| chck82 | rel_eq0 | 1541 | 0.26 | -0.93 | -0.93 |
| chck82 | rel_eq0_irrelevant_ops_1to3 | 641 | 0.31 | 0.00 | 0.00 |
| chck82 | rel_eq0_irrelevant_ops_4to6 | 323 | 0.93 | -1.32 | -1.32 |
| chck82 | rel_eq0_irrelevant_ops_ge7 | 273 | 0.00 | -2.53 | -2.53 |
| chck82 | rel_eq0_irrelevant_ops_gt0 | 1237 | 0.40 | -0.93 | -0.93 |
| chck82 | rel_ge1 | 5239 | 0.00 | -0.14 | -0.25 |
| chck82 | rel_ge1_postrel_ops0 | 1850 | -0.05 | -0.22 | -1.66 |
| chck82 | rel_ge1_postrel_ops_gt0 | 3389 | 0.03 | 0.00 | 0.00 |
| chck82 | rel_ge3_postrel_ops0 | 1114 | 0.27 | 0.00 | -2.61 |
| chck82 | rel_ge3_postrel_ops_gt0 | 1536 | 0.07 | 0.23 | 0.23 |
| chck82 | stale_available_not_gold | 1221 | 0.08 | 0.15 | 0.36 |
| tight_private_scale0p2 | ALL | 6780 | -3.45 | -13.83 | -13.35 |
| tight_private_scale0p2 | lastop_recency_gold | 1850 | -12.86 | -13.24 | -3.87 |
| tight_private_scale0p2 | lastop_recency_not_gold | 1347 | -0.22 | -14.63 | -14.63 |
| tight_private_scale0p2 | lastop_recency_not_gold_available | 1528 | -1.18 | -14.33 | -13.35 |
| tight_private_scale0p2 | rel_eq0 | 1541 | 8.57 | -11.21 | -11.21 |
| tight_private_scale0p2 | rel_eq0_irrelevant_ops_1to3 | 641 | 8.42 | -6.02 | -6.02 |
| tight_private_scale0p2 | rel_eq0_irrelevant_ops_4to6 | 323 | 3.72 | -19.74 | -19.74 |
| tight_private_scale0p2 | rel_eq0_irrelevant_ops_ge7 | 273 | 7.69 | -13.92 | -13.92 |
| tight_private_scale0p2 | rel_eq0_irrelevant_ops_gt0 | 1237 | 7.03 | -11.21 | -11.21 |
| tight_private_scale0p2 | rel_ge1 | 5239 | -6.99 | -14.12 | -13.92 |
| tight_private_scale0p2 | rel_ge1_postrel_ops0 | 1850 | -12.86 | -13.24 | -3.87 |
| tight_private_scale0p2 | rel_ge1_postrel_ops_gt0 | 3389 | -3.78 | -15.69 | -15.69 |
| tight_private_scale0p2 | rel_ge3_postrel_ops0 | 1114 | -15.62 | -16.25 | -6.09 |
| tight_private_scale0p2 | rel_ge3_postrel_ops_gt0 | 1536 | -8.40 | -12.07 | -12.07 |
| tight_private_scale0p2 | stale_available_not_gold | 1221 | -11.88 | -16.01 | -11.27 |

# matched support source contrast result matched-support source contrast — seed 42

Train/probe support repair: one matched and one unmatched final slot per relation-arm sequence; matched target slot is uniformly slot 6/7; source slot is uniform 0..5 and not sorted. Probes use the same support. `U_nomatch` is now in train support for relation arms. `U_wrongmatch` is an extra diagnostic with the same entity but a wrong source attribute.

## Final epoch table

| condition | copy_gain_no | content_gain_no | copy_T | copy_U_no | content_T | content_U_no | content_U_wrong | matched_train | unmatched_train | loss |
|---|---|---|---|---|---|---|---|---|---|---|
| unique | -0.000 | +0.001 | 4.270 | 4.270 | 4.051 | 4.052 | 4.052 |  | 4.161 | 1.877 |
| support_control | +0.000 | -0.000 | 4.144 | 4.144 | 4.200 | 4.200 | 4.200 | 4.146 | 4.165 | 1.893 |
| repeat_full | +2.400 | +0.011 | 1.837 | 4.237 | 5.871 | 5.882 | 5.866 | 1.923 | 4.483 | 1.841 |
| repeat_masked | -0.010 | +0.004 | 4.360 | 4.350 | 3.920 | 3.924 | 3.920 | 4.424 | 4.165 | 1.894 |
| varied_full | +0.181 | +2.406 | 5.270 | 5.452 | 2.070 | 4.476 | 4.550 | 1.919 | 4.489 | 1.839 |
| varied_masked | -0.023 | +0.007 | 5.161 | 5.138 | 3.803 | 3.810 | 3.803 | 3.761 | 4.359 | 1.872 |
| wrong_full | +0.016 | -0.030 | 3.672 | 3.688 | 6.170 | 6.140 | 6.169 | 3.688 | 4.209 | 1.880 |

## Source-specific contrasts

Positive `excess_true_cost` means A is worse on true-source use beyond its unrelated-source change: `(T_A-T_B)-(U_A-U_B) = -(gain_A-gain_B)`.

| A | B | relation | U | T_delta | U_delta | excess_true_cost | gain_delta |
|---|---|---|---|---:|---:|---:|---:|
| repeat_full | repeat_masked | content | nomatch | +1.951 | +1.957 | -0.007 | +0.007 |
| varied_full | varied_masked | content | nomatch | -1.732 | +0.666 | -2.398 | +2.398 |
| repeat_full | unique | content | nomatch | +1.820 | +1.830 | -0.010 | +0.010 |
| repeat_full | support_control | content | nomatch | +1.671 | +1.681 | -0.011 | +0.011 |
| varied_full | unique | content | nomatch | -1.981 | +0.424 | -2.405 | +2.405 |
| varied_full | support_control | content | nomatch | -2.130 | +0.276 | -2.406 | +2.406 |
| repeat_full | varied_full | content | nomatch | +3.801 | +1.406 | +2.395 | -2.395 |
| repeat_full | repeat_masked | content | wrongmatch | +1.951 | +1.946 | +0.004 | -0.004 |
| varied_full | varied_masked | content | wrongmatch | -1.732 | +0.747 | -2.479 | +2.479 |
| repeat_full | unique | content | wrongmatch | +1.820 | +1.815 | +0.005 | -0.005 |
| repeat_full | support_control | content | wrongmatch | +1.671 | +1.666 | +0.005 | -0.005 |
| varied_full | unique | content | wrongmatch | -1.981 | +0.498 | -2.479 | +2.479 |
| varied_full | support_control | content | wrongmatch | -2.130 | +0.349 | -2.479 | +2.479 |
| repeat_full | varied_full | content | wrongmatch | +3.801 | +1.316 | +2.484 | -2.484 |
| repeat_full | repeat_masked | copy | nomatch | -2.523 | -0.113 | -2.411 | +2.411 |
| varied_full | varied_masked | copy | nomatch | +0.110 | +0.313 | -0.204 | +0.204 |
| repeat_full | unique | copy | nomatch | -2.433 | -0.033 | -2.400 | +2.400 |
| repeat_full | support_control | copy | nomatch | -2.307 | +0.093 | -2.400 | +2.400 |
| varied_full | unique | copy | nomatch | +1.000 | +1.182 | -0.181 | +0.181 |
| varied_full | support_control | copy | nomatch | +1.127 | +1.307 | -0.181 | +0.181 |
| repeat_full | varied_full | copy | nomatch | -3.433 | -1.214 | -2.219 | +2.219 |
| repeat_full | repeat_masked | copy | wrongmatch | -2.523 | -0.029 | -2.494 | +2.494 |
| varied_full | varied_masked | copy | wrongmatch | +0.110 | +0.126 | -0.016 | +0.016 |
| repeat_full | unique | copy | wrongmatch | -2.433 | +0.061 | -2.494 | +2.494 |
| repeat_full | support_control | copy | wrongmatch | -2.307 | +0.188 | -2.495 | +2.495 |
| varied_full | unique | copy | wrongmatch | +1.000 | +1.010 | -0.010 | +0.010 |
| varied_full | support_control | copy | wrongmatch | +1.127 | +1.137 | -0.011 | +0.011 |
| repeat_full | varied_full | copy | wrongmatch | -3.433 | -0.949 | -2.484 | +2.484 |

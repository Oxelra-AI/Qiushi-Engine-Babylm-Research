# matched support source contrast result matched-support source contrast — seed 100

Train/probe support repair: one matched and one unmatched final slot per relation-arm sequence; matched target slot is uniformly slot 6/7; source slot is uniform 0..5 and not sorted. Probes use the same support. `U_nomatch` is now in train support for relation arms. `U_wrongmatch` is an extra diagnostic with the same entity but a wrong source attribute.

## Final epoch table

| condition | copy_gain_no | content_gain_no | copy_T | copy_U_no | content_T | content_U_no | content_U_wrong | matched_train | unmatched_train | loss |
|---|---|---|---|---|---|---|---|---|---|---|
| unique | -0.001 | -0.002 | 4.156 | 4.156 | 4.173 | 4.171 | 4.173 |  | 4.157 | 1.876 |
| support_control | +0.000 | +0.000 | 4.153 | 4.153 | 4.178 | 4.179 | 4.178 | 4.152 | 4.151 | 1.892 |
| repeat_full | +0.267 | +0.052 | 3.435 | 3.702 | 6.850 | 6.902 | 6.846 | 3.433 | 4.282 | 1.878 |
| repeat_masked | +0.004 | +0.045 | 3.983 | 3.987 | 4.514 | 4.559 | 4.512 | 4.019 | 4.144 | 1.873 |
| varied_full | +0.024 | +1.067 | 4.425 | 4.449 | 3.107 | 4.174 | 4.205 | 2.522 | 4.380 | 1.857 |
| varied_masked | +0.011 | -0.006 | 4.306 | 4.317 | 4.084 | 4.078 | 4.085 | 4.066 | 4.181 | 1.873 |
| wrong_full | +0.022 | +0.010 | 3.703 | 3.725 | 6.201 | 6.210 | 6.204 | 3.684 | 4.235 | 1.883 |

## Source-specific contrasts

Positive `excess_true_cost` means A is worse on true-source use beyond its unrelated-source change: `(T_A-T_B)-(U_A-U_B) = -(gain_A-gain_B)`.

| A | B | relation | U | T_delta | U_delta | excess_true_cost | gain_delta |
|---|---|---|---|---:|---:|---:|---:|
| repeat_full | repeat_masked | content | nomatch | +2.336 | +2.342 | -0.006 | +0.006 |
| varied_full | varied_masked | content | nomatch | -0.977 | +0.096 | -1.073 | +1.073 |
| repeat_full | unique | content | nomatch | +2.677 | +2.730 | -0.054 | +0.054 |
| repeat_full | support_control | content | nomatch | +2.672 | +2.723 | -0.051 | +0.051 |
| varied_full | unique | content | nomatch | -1.067 | +0.002 | -1.069 | +1.069 |
| varied_full | support_control | content | nomatch | -1.072 | -0.005 | -1.066 | +1.066 |
| repeat_full | varied_full | content | nomatch | +3.743 | +2.728 | +1.015 | -1.015 |
| repeat_full | repeat_masked | content | wrongmatch | +2.336 | +2.333 | +0.003 | -0.003 |
| varied_full | varied_masked | content | wrongmatch | -0.977 | +0.120 | -1.097 | +1.097 |
| repeat_full | unique | content | wrongmatch | +2.677 | +2.672 | +0.004 | -0.004 |
| repeat_full | support_control | content | wrongmatch | +2.672 | +2.667 | +0.005 | -0.005 |
| varied_full | unique | content | wrongmatch | -1.067 | +0.032 | -1.098 | +1.098 |
| varied_full | support_control | content | wrongmatch | -1.072 | +0.026 | -1.098 | +1.098 |
| repeat_full | varied_full | content | wrongmatch | +3.743 | +2.641 | +1.103 | -1.103 |
| repeat_full | repeat_masked | copy | nomatch | -0.547 | -0.285 | -0.263 | +0.263 |
| varied_full | varied_masked | copy | nomatch | +0.119 | +0.132 | -0.013 | +0.013 |
| repeat_full | unique | copy | nomatch | -0.721 | -0.454 | -0.267 | +0.267 |
| repeat_full | support_control | copy | nomatch | -0.718 | -0.451 | -0.266 | +0.266 |
| varied_full | unique | copy | nomatch | +0.268 | +0.293 | -0.025 | +0.025 |
| varied_full | support_control | copy | nomatch | +0.272 | +0.296 | -0.024 | +0.024 |
| repeat_full | varied_full | copy | nomatch | -0.990 | -0.747 | -0.242 | +0.242 |
| repeat_full | repeat_masked | copy | wrongmatch | -0.547 | -0.258 | -0.290 | +0.290 |
| varied_full | varied_masked | copy | wrongmatch | +0.119 | +0.123 | -0.004 | +0.004 |
| repeat_full | unique | copy | wrongmatch | -0.721 | -0.430 | -0.291 | +0.291 |
| repeat_full | support_control | copy | wrongmatch | -0.718 | -0.427 | -0.291 | +0.291 |
| varied_full | unique | copy | wrongmatch | +0.268 | +0.272 | -0.003 | +0.003 |
| varied_full | support_control | copy | wrongmatch | +0.272 | +0.275 | -0.003 | +0.003 |
| repeat_full | varied_full | copy | wrongmatch | -0.990 | -0.702 | -0.288 | +0.288 |

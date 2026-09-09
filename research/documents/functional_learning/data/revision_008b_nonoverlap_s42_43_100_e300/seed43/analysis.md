# matched support source contrast result matched-support source contrast — seed 43

Train/probe support repair: one matched and one unmatched final slot per relation-arm sequence; matched target slot is uniformly slot 6/7; source slot is uniform 0..5 and not sorted. Probes use the same support. `U_nomatch` is now in train support for relation arms. `U_wrongmatch` is an extra diagnostic with the same entity but a wrong source attribute.

## Final epoch table

| condition | copy_gain_no | content_gain_no | copy_T | copy_U_no | content_T | content_U_no | content_U_wrong | matched_train | unmatched_train | loss |
|---|---|---|---|---|---|---|---|---|---|---|
| unique | +0.000 | +0.001 | 4.129 | 4.129 | 4.183 | 4.184 | 4.183 |  | 4.142 | 1.875 |
| support_control | -0.000 | +0.001 | 4.134 | 4.134 | 4.195 | 4.196 | 4.195 | 4.161 | 4.151 | 1.893 |
| repeat_full | +2.941 | +0.102 | 1.552 | 4.493 | 5.333 | 5.435 | 5.330 | 1.865 | 4.367 | 1.839 |
| repeat_masked | -0.002 | +0.036 | 4.064 | 4.062 | 5.570 | 5.606 | 5.568 | 4.080 | 4.147 | 1.873 |
| varied_full | +0.083 | +2.613 | 5.216 | 5.298 | 1.811 | 4.423 | 4.582 | 1.885 | 4.449 | 1.840 |
| varied_masked | -0.009 | +0.009 | 4.266 | 4.256 | 4.140 | 4.148 | 4.142 | 4.072 | 4.153 | 1.873 |
| wrong_full | +0.016 | +0.080 | 3.674 | 3.690 | 5.960 | 6.040 | 5.958 | 3.687 | 4.209 | 1.881 |

## Source-specific contrasts

Positive `excess_true_cost` means A is worse on true-source use beyond its unrelated-source change: `(T_A-T_B)-(U_A-U_B) = -(gain_A-gain_B)`.

| A | B | relation | U | T_delta | U_delta | excess_true_cost | gain_delta |
|---|---|---|---|---:|---:|---:|---:|
| repeat_full | repeat_masked | content | nomatch | -0.237 | -0.172 | -0.066 | +0.066 |
| varied_full | varied_masked | content | nomatch | -2.329 | +0.275 | -2.604 | +2.604 |
| repeat_full | unique | content | nomatch | +1.150 | +1.251 | -0.101 | +0.101 |
| repeat_full | support_control | content | nomatch | +1.138 | +1.239 | -0.101 | +0.101 |
| varied_full | unique | content | nomatch | -2.373 | +0.239 | -2.612 | +2.612 |
| varied_full | support_control | content | nomatch | -2.384 | +0.228 | -2.612 | +2.612 |
| repeat_full | varied_full | content | nomatch | +3.522 | +1.011 | +2.511 | -2.511 |
| repeat_full | repeat_masked | content | wrongmatch | -0.237 | -0.238 | +0.001 | -0.001 |
| varied_full | varied_masked | content | wrongmatch | -2.329 | +0.440 | -2.769 | +2.769 |
| repeat_full | unique | content | wrongmatch | +1.150 | +1.147 | +0.003 | -0.003 |
| repeat_full | support_control | content | wrongmatch | +1.138 | +1.135 | +0.003 | -0.003 |
| varied_full | unique | content | wrongmatch | -2.373 | +0.400 | -2.772 | +2.772 |
| varied_full | support_control | content | wrongmatch | -2.384 | +0.388 | -2.772 | +2.772 |
| repeat_full | varied_full | content | wrongmatch | +3.522 | +0.747 | +2.775 | -2.775 |
| repeat_full | repeat_masked | copy | nomatch | -2.512 | +0.431 | -2.943 | +2.943 |
| varied_full | varied_masked | copy | nomatch | +0.950 | +1.042 | -0.092 | +0.092 |
| repeat_full | unique | copy | nomatch | -2.577 | +0.364 | -2.941 | +2.941 |
| repeat_full | support_control | copy | nomatch | -2.581 | +0.360 | -2.941 | +2.941 |
| varied_full | unique | copy | nomatch | +1.087 | +1.169 | -0.082 | +0.082 |
| varied_full | support_control | copy | nomatch | +1.082 | +1.165 | -0.083 | +0.083 |
| repeat_full | varied_full | copy | nomatch | -3.664 | -0.805 | -2.858 | +2.858 |
| repeat_full | repeat_masked | copy | wrongmatch | -2.512 | +0.530 | -3.042 | +3.042 |
| varied_full | varied_masked | copy | wrongmatch | +0.950 | +0.952 | -0.002 | +0.002 |
| repeat_full | unique | copy | wrongmatch | -2.577 | +0.466 | -3.043 | +3.043 |
| repeat_full | support_control | copy | wrongmatch | -2.581 | +0.462 | -3.043 | +3.043 |
| varied_full | unique | copy | wrongmatch | +1.087 | +1.082 | +0.005 | -0.005 |
| varied_full | support_control | copy | wrongmatch | +1.082 | +1.078 | +0.004 | -0.004 |
| repeat_full | varied_full | copy | wrongmatch | -3.664 | -0.616 | -3.047 | +3.047 |

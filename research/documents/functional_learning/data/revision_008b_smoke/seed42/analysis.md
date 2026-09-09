# matched support source contrast result matched-support source contrast — seed 42

Train/probe support repair: one matched and one unmatched final slot per relation-arm sequence; matched target slot is uniformly slot 6/7; source slot is uniform 0..5 and not sorted. Probes use the same support. `U_nomatch` is now in train support for relation arms. `U_wrongmatch` is an extra diagnostic with the same entity but a wrong source attribute.

## Final epoch table

| condition | copy_gain_no | content_gain_no | copy_T | copy_U_no | content_T | content_U_no | content_U_wrong | matched_train | unmatched_train | loss |
|---|---|---|---|---|---|---|---|---|---|---|
| unique | +0.002 | +0.004 | 4.568 | 4.569 | 4.774 | 4.778 | 4.775 |  | 4.723 | 4.055 |
| support_control | +0.002 | +0.004 | 4.552 | 4.554 | 4.765 | 4.769 | 4.767 | 4.726 | 4.877 | 4.043 |
| repeat_full | +0.002 | +0.004 | 4.517 | 4.519 | 4.795 | 4.800 | 4.797 | 4.543 | 4.703 | 4.016 |
| repeat_masked | +0.002 | +0.004 | 4.517 | 4.519 | 4.796 | 4.800 | 4.797 | 4.543 | 4.703 | 4.017 |
| varied_full | +0.002 | +0.004 | 4.577 | 4.578 | 4.756 | 4.760 | 4.757 | 4.820 | 4.738 | 4.066 |
| varied_masked | +0.002 | +0.004 | 4.576 | 4.578 | 4.756 | 4.760 | 4.758 | 4.820 | 4.738 | 4.067 |
| wrong_full | +0.002 | +0.004 | 4.519 | 4.521 | 4.805 | 4.809 | 4.807 | 4.651 | 4.767 | 4.040 |

## Source-specific contrasts

Positive `excess_true_cost` means A is worse on true-source use beyond its unrelated-source change: `(T_A-T_B)-(U_A-U_B) = -(gain_A-gain_B)`.

| A | B | relation | U | T_delta | U_delta | excess_true_cost | gain_delta |
|---|---|---|---|---:|---:|---:|---:|
| repeat_full | repeat_masked | content | nomatch | -0.000 | -0.000 | +0.000 | -0.000 |
| varied_full | varied_masked | content | nomatch | -0.000 | -0.000 | -0.000 | +0.000 |
| repeat_full | unique | content | nomatch | +0.022 | +0.022 | -0.000 | +0.000 |
| repeat_full | support_control | content | nomatch | +0.030 | +0.031 | -0.001 | +0.001 |
| varied_full | unique | content | nomatch | -0.018 | -0.018 | +0.000 | -0.000 |
| varied_full | support_control | content | nomatch | -0.010 | -0.009 | -0.000 | +0.000 |
| repeat_full | varied_full | content | nomatch | +0.040 | +0.040 | -0.000 | +0.000 |
| repeat_full | repeat_masked | content | wrongmatch | -0.000 | -0.000 | +0.000 | +0.000 |
| varied_full | varied_masked | content | wrongmatch | -0.000 | -0.000 | -0.000 | +0.000 |
| repeat_full | unique | content | wrongmatch | +0.022 | +0.022 | -0.000 | +0.000 |
| repeat_full | support_control | content | wrongmatch | +0.030 | +0.030 | -0.000 | +0.000 |
| varied_full | unique | content | wrongmatch | -0.018 | -0.018 | -0.000 | +0.000 |
| varied_full | support_control | content | wrongmatch | -0.010 | -0.010 | -0.000 | +0.000 |
| repeat_full | varied_full | content | wrongmatch | +0.040 | +0.040 | +0.000 | -0.000 |
| repeat_full | repeat_masked | copy | nomatch | -0.000 | -0.000 | -0.000 | +0.000 |
| varied_full | varied_masked | copy | nomatch | +0.001 | +0.000 | +0.000 | -0.000 |
| repeat_full | unique | copy | nomatch | -0.051 | -0.051 | -0.000 | +0.000 |
| repeat_full | support_control | copy | nomatch | -0.035 | -0.035 | -0.000 | +0.000 |
| varied_full | unique | copy | nomatch | +0.009 | +0.009 | +0.000 | -0.000 |
| varied_full | support_control | copy | nomatch | +0.025 | +0.025 | +0.000 | -0.000 |
| repeat_full | varied_full | copy | nomatch | -0.060 | -0.060 | -0.001 | +0.001 |
| repeat_full | repeat_masked | copy | wrongmatch | -0.000 | -0.000 | -0.000 | +0.000 |
| varied_full | varied_masked | copy | wrongmatch | +0.001 | +0.001 | -0.000 | +0.000 |
| repeat_full | unique | copy | wrongmatch | -0.051 | -0.051 | -0.000 | +0.000 |
| repeat_full | support_control | copy | wrongmatch | -0.035 | -0.035 | -0.000 | +0.000 |
| varied_full | unique | copy | wrongmatch | +0.009 | +0.009 | -0.000 | +0.000 |
| varied_full | support_control | copy | wrongmatch | +0.025 | +0.025 | -0.000 | +0.000 |
| repeat_full | varied_full | copy | wrongmatch | -0.060 | -0.060 | -0.000 | +0.000 |

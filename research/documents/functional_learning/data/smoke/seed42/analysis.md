# matched support source contrast result matched-support source contrast — seed 42

Train/probe support repair: one matched and one unmatched final slot per relation-arm sequence; matched target slot is uniformly slot 6/7; source slot is uniform 0..5 and not sorted. Probes use the same support. `U_nomatch` is now in train support for relation arms. `U_wrongmatch` is an extra diagnostic with the same entity but a wrong source attribute.

## Final epoch table

| condition | copy_gain_no | content_gain_no | copy_T | copy_U_no | content_T | content_U_no | content_U_wrong | matched_train | unmatched_train | loss |
|---|---|---|---|---|---|---|---|---|---|---|
| unique | -0.009 | -0.010 | 4.205 | 4.196 | 4.280 | 4.270 | 4.279 |  | 4.474 | 3.759 |
| support_control | -0.009 | -0.010 | 4.200 | 4.191 | 4.275 | 4.265 | 4.274 | 4.471 | 4.281 | 3.755 |
| repeat_full | -0.008 | -0.009 | 4.217 | 4.209 | 4.295 | 4.286 | 4.294 | 4.295 | 4.310 | 3.747 |
| repeat_masked | -0.009 | -0.009 | 4.217 | 4.209 | 4.295 | 4.286 | 4.294 | 4.295 | 4.310 | 3.747 |
| varied_full | -0.008 | -0.009 | 4.216 | 4.208 | 4.290 | 4.280 | 4.289 | 4.350 | 4.486 | 3.757 |
| varied_masked | -0.008 | -0.010 | 4.217 | 4.209 | 4.290 | 4.280 | 4.289 | 4.350 | 4.486 | 3.757 |
| wrong_full | -0.008 | -0.009 | 4.210 | 4.202 | 4.269 | 4.260 | 4.268 | 4.368 | 4.446 | 3.761 |

## Source-specific contrasts

Positive `excess_true_cost` means A is worse on true-source use beyond its unrelated-source change: `(T_A-T_B)-(U_A-U_B) = -(gain_A-gain_B)`.

| A | B | relation | U | T_delta | U_delta | excess_true_cost | gain_delta |
|---|---|---|---|---:|---:|---:|---:|
| repeat_full | repeat_masked | content | nomatch | +0.000 | +0.000 | -0.000 | +0.000 |
| varied_full | varied_masked | content | nomatch | -0.000 | -0.000 | -0.000 | +0.000 |
| repeat_full | unique | content | nomatch | +0.015 | +0.016 | -0.001 | +0.001 |
| repeat_full | support_control | content | nomatch | +0.020 | +0.021 | -0.001 | +0.001 |
| varied_full | unique | content | nomatch | +0.010 | +0.010 | -0.000 | +0.000 |
| varied_full | support_control | content | nomatch | +0.015 | +0.015 | -0.000 | +0.000 |
| repeat_full | varied_full | content | nomatch | +0.005 | +0.006 | -0.001 | +0.001 |
| repeat_full | repeat_masked | content | wrongmatch | +0.000 | +0.000 | -0.000 | +0.000 |
| varied_full | varied_masked | content | wrongmatch | -0.000 | -0.000 | -0.000 | +0.000 |
| repeat_full | unique | content | wrongmatch | +0.015 | +0.015 | +0.000 | -0.000 |
| repeat_full | support_control | content | wrongmatch | +0.020 | +0.020 | +0.000 | -0.000 |
| varied_full | unique | content | wrongmatch | +0.010 | +0.010 | -0.000 | +0.000 |
| varied_full | support_control | content | wrongmatch | +0.015 | +0.015 | -0.000 | +0.000 |
| repeat_full | varied_full | content | wrongmatch | +0.005 | +0.005 | +0.001 | -0.001 |
| repeat_full | repeat_masked | copy | nomatch | -0.000 | +0.000 | -0.000 | +0.000 |
| varied_full | varied_masked | copy | nomatch | -0.000 | -0.000 | -0.000 | +0.000 |
| repeat_full | unique | copy | nomatch | +0.012 | +0.012 | -0.000 | +0.000 |
| repeat_full | support_control | copy | nomatch | +0.017 | +0.017 | -0.000 | +0.000 |
| varied_full | unique | copy | nomatch | +0.011 | +0.012 | -0.001 | +0.001 |
| varied_full | support_control | copy | nomatch | +0.016 | +0.017 | -0.001 | +0.001 |
| repeat_full | varied_full | copy | nomatch | +0.001 | +0.000 | +0.000 | -0.000 |
| repeat_full | repeat_masked | copy | wrongmatch | -0.000 | +0.000 | -0.000 | +0.000 |
| varied_full | varied_masked | copy | wrongmatch | -0.000 | -0.000 | -0.000 | +0.000 |
| repeat_full | unique | copy | wrongmatch | +0.012 | +0.012 | +0.000 | -0.000 |
| repeat_full | support_control | copy | wrongmatch | +0.017 | +0.017 | +0.000 | -0.000 |
| varied_full | unique | copy | wrongmatch | +0.011 | +0.012 | -0.000 | +0.000 |
| varied_full | support_control | copy | wrongmatch | +0.016 | +0.017 | -0.000 | +0.000 |
| repeat_full | varied_full | copy | wrongmatch | +0.001 | +0.000 | +0.000 | -0.000 |

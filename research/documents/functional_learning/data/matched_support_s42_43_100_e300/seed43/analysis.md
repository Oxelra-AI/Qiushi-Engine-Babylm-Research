# matched support source contrast result matched-support source contrast — seed 43

Train/probe support repair: one matched and one unmatched final slot per relation-arm sequence; matched target slot is uniformly slot 6/7; source slot is uniform 0..5 and not sorted. Probes use the same support. `U_nomatch` is now in train support for relation arms. `U_wrongmatch` is an extra diagnostic with the same entity but a wrong source attribute.

## Final epoch table

| condition | copy_gain_no | content_gain_no | copy_T | copy_U_no | content_T | content_U_no | content_U_wrong | matched_train | unmatched_train | loss |
|---|---|---|---|---|---|---|---|---|---|---|
| unique | +0.000 | -0.000 | 3.477 | 3.477 | 3.477 | 3.477 | 3.477 |  | 3.459 | 1.831 |
| support_control | +0.000 | +0.000 | 3.461 | 3.462 | 3.461 | 3.461 | 3.461 | 3.457 | 3.467 | 1.848 |
| repeat_full | +1.688 | +0.283 | 2.115 | 3.803 | 3.580 | 3.863 | 3.854 | 1.971 | 3.728 | 1.819 |
| repeat_masked | +0.001 | +0.000 | 3.461 | 3.462 | 3.463 | 3.463 | 3.462 | 3.457 | 3.474 | 1.849 |
| varied_full | +0.330 | +1.767 | 3.474 | 3.804 | 2.052 | 3.818 | 3.947 | 1.973 | 3.723 | 1.820 |
| varied_masked | +0.000 | -0.000 | 3.464 | 3.465 | 3.464 | 3.464 | 3.464 | 3.461 | 3.460 | 1.835 |
| wrong_full | -0.000 | +0.000 | 3.467 | 3.467 | 3.468 | 3.468 | 3.468 | 3.468 | 3.465 | 1.847 |

## Source-specific contrasts

Positive `excess_true_cost` means A is worse on true-source use beyond its unrelated-source change: `(T_A-T_B)-(U_A-U_B) = -(gain_A-gain_B)`.

| A | B | relation | U | T_delta | U_delta | excess_true_cost | gain_delta |
|---|---|---|---|---:|---:|---:|---:|
| repeat_full | repeat_masked | content | nomatch | +0.118 | +0.400 | -0.282 | +0.282 |
| varied_full | varied_masked | content | nomatch | -1.412 | +0.355 | -1.767 | +1.767 |
| repeat_full | unique | content | nomatch | +0.103 | +0.386 | -0.283 | +0.283 |
| repeat_full | support_control | content | nomatch | +0.119 | +0.402 | -0.282 | +0.282 |
| varied_full | unique | content | nomatch | -1.425 | +0.342 | -1.767 | +1.767 |
| varied_full | support_control | content | nomatch | -1.409 | +0.357 | -1.766 | +1.766 |
| repeat_full | varied_full | content | nomatch | +1.528 | +0.045 | +1.484 | -1.484 |
| repeat_full | repeat_masked | content | wrongmatch | +0.118 | +0.392 | -0.274 | +0.274 |
| varied_full | varied_masked | content | wrongmatch | -1.412 | +0.483 | -1.895 | +1.895 |
| repeat_full | unique | content | wrongmatch | +0.103 | +0.378 | -0.274 | +0.274 |
| repeat_full | support_control | content | wrongmatch | +0.119 | +0.394 | -0.274 | +0.274 |
| varied_full | unique | content | wrongmatch | -1.425 | +0.470 | -1.895 | +1.895 |
| varied_full | support_control | content | wrongmatch | -1.409 | +0.486 | -1.895 | +1.895 |
| repeat_full | varied_full | content | wrongmatch | +1.528 | -0.093 | +1.621 | -1.621 |
| repeat_full | repeat_masked | copy | nomatch | -1.346 | +0.341 | -1.688 | +1.688 |
| varied_full | varied_masked | copy | nomatch | +0.010 | +0.339 | -0.329 | +0.329 |
| repeat_full | unique | copy | nomatch | -1.362 | +0.326 | -1.688 | +1.688 |
| repeat_full | support_control | copy | nomatch | -1.346 | +0.341 | -1.688 | +1.688 |
| varied_full | unique | copy | nomatch | -0.003 | +0.327 | -0.329 | +0.329 |
| varied_full | support_control | copy | nomatch | +0.013 | +0.342 | -0.329 | +0.329 |
| repeat_full | varied_full | copy | nomatch | -1.360 | -0.001 | -1.359 | +1.359 |
| repeat_full | repeat_masked | copy | wrongmatch | -1.346 | +0.424 | -1.770 | +1.770 |
| varied_full | varied_masked | copy | wrongmatch | +0.010 | +0.342 | -0.332 | +0.332 |
| repeat_full | unique | copy | wrongmatch | -1.362 | +0.408 | -1.770 | +1.770 |
| repeat_full | support_control | copy | wrongmatch | -1.346 | +0.424 | -1.770 | +1.770 |
| varied_full | unique | copy | wrongmatch | -0.003 | +0.330 | -0.332 | +0.332 |
| varied_full | support_control | copy | wrongmatch | +0.013 | +0.346 | -0.332 | +0.332 |
| repeat_full | varied_full | copy | wrongmatch | -1.360 | +0.078 | -1.438 | +1.438 |

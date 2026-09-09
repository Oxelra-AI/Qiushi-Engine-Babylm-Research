# matched support source contrast result matched-support source contrast — seed 42

Train/probe support repair: one matched and one unmatched final slot per relation-arm sequence; matched target slot is uniformly slot 6/7; source slot is uniform 0..5 and not sorted. Probes use the same support. `U_nomatch` is now in train support for relation arms. `U_wrongmatch` is an extra diagnostic with the same entity but a wrong source attribute.

## Final epoch table

| condition | copy_gain_no | content_gain_no | copy_T | copy_U_no | content_T | content_U_no | content_U_wrong | matched_train | unmatched_train | loss |
|---|---|---|---|---|---|---|---|---|---|---|
| unique | -0.001 | -0.001 | 3.479 | 3.479 | 3.479 | 3.479 | 3.479 |  | 3.466 | 1.834 |
| support_control | +0.001 | +0.000 | 3.469 | 3.469 | 3.470 | 3.470 | 3.470 | 3.470 | 3.455 | 1.849 |
| repeat_full | +1.825 | +0.341 | 2.058 | 3.883 | 3.480 | 3.821 | 3.884 | 1.886 | 3.701 | 1.818 |
| repeat_masked | +0.000 | +0.000 | 3.475 | 3.476 | 3.476 | 3.476 | 3.476 | 3.449 | 3.464 | 1.850 |
| varied_full | +0.354 | +1.770 | 3.470 | 3.823 | 2.049 | 3.819 | 3.924 | 1.973 | 3.724 | 1.818 |
| varied_masked | +0.000 | +0.000 | 3.477 | 3.477 | 3.477 | 3.477 | 3.477 | 3.460 | 3.467 | 1.849 |
| wrong_full | +0.000 | -0.000 | 3.478 | 3.478 | 3.477 | 3.477 | 3.477 | 3.450 | 3.474 | 1.849 |

## Source-specific contrasts

Positive `excess_true_cost` means A is worse on true-source use beyond its unrelated-source change: `(T_A-T_B)-(U_A-U_B) = -(gain_A-gain_B)`.

| A | B | relation | U | T_delta | U_delta | excess_true_cost | gain_delta |
|---|---|---|---|---:|---:|---:|---:|
| repeat_full | repeat_masked | content | nomatch | +0.004 | +0.344 | -0.341 | +0.341 |
| varied_full | varied_masked | content | nomatch | -1.428 | +0.342 | -1.770 | +1.770 |
| repeat_full | unique | content | nomatch | +0.000 | +0.342 | -0.342 | +0.342 |
| repeat_full | support_control | content | nomatch | +0.010 | +0.351 | -0.341 | +0.341 |
| varied_full | unique | content | nomatch | -1.431 | +0.341 | -1.771 | +1.771 |
| varied_full | support_control | content | nomatch | -1.421 | +0.349 | -1.770 | +1.770 |
| repeat_full | varied_full | content | nomatch | +1.431 | +0.001 | +1.430 | -1.430 |
| repeat_full | repeat_masked | content | wrongmatch | +0.004 | +0.408 | -0.404 | +0.404 |
| varied_full | varied_masked | content | wrongmatch | -1.428 | +0.448 | -1.876 | +1.876 |
| repeat_full | unique | content | wrongmatch | +0.000 | +0.405 | -0.405 | +0.405 |
| repeat_full | support_control | content | wrongmatch | +0.010 | +0.414 | -0.405 | +0.405 |
| varied_full | unique | content | wrongmatch | -1.431 | +0.445 | -1.876 | +1.876 |
| varied_full | support_control | content | wrongmatch | -1.421 | +0.455 | -1.876 | +1.876 |
| repeat_full | varied_full | content | wrongmatch | +1.431 | -0.040 | +1.471 | -1.471 |
| repeat_full | repeat_masked | copy | nomatch | -1.417 | +0.407 | -1.824 | +1.824 |
| varied_full | varied_masked | copy | nomatch | -0.007 | +0.346 | -0.353 | +0.353 |
| repeat_full | unique | copy | nomatch | -1.421 | +0.404 | -1.825 | +1.825 |
| repeat_full | support_control | copy | nomatch | -1.411 | +0.414 | -1.824 | +1.824 |
| varied_full | unique | copy | nomatch | -0.010 | +0.345 | -0.354 | +0.354 |
| varied_full | support_control | copy | nomatch | +0.001 | +0.354 | -0.353 | +0.353 |
| repeat_full | varied_full | copy | nomatch | -1.412 | +0.059 | -1.471 | +1.471 |
| repeat_full | repeat_masked | copy | wrongmatch | -1.417 | +0.503 | -1.920 | +1.920 |
| varied_full | varied_masked | copy | wrongmatch | -0.007 | +0.401 | -0.408 | +0.408 |
| repeat_full | unique | copy | wrongmatch | -1.421 | +0.498 | -1.920 | +1.920 |
| repeat_full | support_control | copy | wrongmatch | -1.411 | +0.509 | -1.920 | +1.920 |
| varied_full | unique | copy | wrongmatch | -0.010 | +0.399 | -0.408 | +0.408 |
| varied_full | support_control | copy | wrongmatch | +0.001 | +0.409 | -0.408 | +0.408 |
| repeat_full | varied_full | copy | wrongmatch | -1.412 | +0.100 | -1.512 | +1.512 |

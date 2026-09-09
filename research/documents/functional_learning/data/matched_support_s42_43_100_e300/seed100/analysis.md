# matched support source contrast result matched-support source contrast — seed 100

Train/probe support repair: one matched and one unmatched final slot per relation-arm sequence; matched target slot is uniformly slot 6/7; source slot is uniform 0..5 and not sorted. Probes use the same support. `U_nomatch` is now in train support for relation arms. `U_wrongmatch` is an extra diagnostic with the same entity but a wrong source attribute.

## Final epoch table

| condition | copy_gain_no | content_gain_no | copy_T | copy_U_no | content_T | content_U_no | content_U_wrong | matched_train | unmatched_train | loss |
|---|---|---|---|---|---|---|---|---|---|---|
| unique | +0.000 | +0.000 | 3.477 | 3.478 | 3.477 | 3.477 | 3.477 |  | 3.463 | 1.830 |
| support_control | +0.000 | +0.000 | 3.466 | 3.467 | 3.466 | 3.466 | 3.466 | 3.472 | 3.464 | 1.850 |
| repeat_full | +1.622 | +0.405 | 2.223 | 3.845 | 3.458 | 3.863 | 3.884 | 2.016 | 3.782 | 1.818 |
| repeat_masked | +0.001 | +0.000 | 3.466 | 3.466 | 3.467 | 3.468 | 3.467 | 3.463 | 3.466 | 1.849 |
| varied_full | +0.347 | +1.860 | 3.573 | 3.919 | 2.086 | 3.946 | 4.005 | 1.890 | 3.781 | 1.818 |
| varied_masked | +0.001 | +0.000 | 3.472 | 3.473 | 3.471 | 3.472 | 3.472 | 3.456 | 3.463 | 1.849 |
| wrong_full | +0.000 | +0.000 | 3.476 | 3.477 | 3.476 | 3.476 | 3.476 | 3.461 | 3.460 | 1.850 |

## Source-specific contrasts

Positive `excess_true_cost` means A is worse on true-source use beyond its unrelated-source change: `(T_A-T_B)-(U_A-U_B) = -(gain_A-gain_B)`.

| A | B | relation | U | T_delta | U_delta | excess_true_cost | gain_delta |
|---|---|---|---|---:|---:|---:|---:|
| repeat_full | repeat_masked | content | nomatch | -0.009 | +0.395 | -0.405 | +0.405 |
| varied_full | varied_masked | content | nomatch | -1.386 | +0.474 | -1.860 | +1.860 |
| repeat_full | unique | content | nomatch | -0.019 | +0.386 | -0.405 | +0.405 |
| repeat_full | support_control | content | nomatch | -0.008 | +0.397 | -0.405 | +0.405 |
| varied_full | unique | content | nomatch | -1.391 | +0.469 | -1.860 | +1.860 |
| varied_full | support_control | content | nomatch | -1.380 | +0.480 | -1.860 | +1.860 |
| repeat_full | varied_full | content | nomatch | +1.372 | -0.083 | +1.455 | -1.455 |
| repeat_full | repeat_masked | content | wrongmatch | -0.009 | +0.416 | -0.426 | +0.426 |
| varied_full | varied_masked | content | wrongmatch | -1.386 | +0.534 | -1.919 | +1.919 |
| repeat_full | unique | content | wrongmatch | -0.019 | +0.406 | -0.425 | +0.425 |
| repeat_full | support_control | content | wrongmatch | -0.008 | +0.418 | -0.426 | +0.426 |
| varied_full | unique | content | wrongmatch | -1.391 | +0.528 | -1.919 | +1.919 |
| varied_full | support_control | content | wrongmatch | -1.380 | +0.539 | -1.919 | +1.919 |
| repeat_full | varied_full | content | wrongmatch | +1.372 | -0.122 | +1.494 | -1.494 |
| repeat_full | repeat_masked | copy | nomatch | -1.243 | +0.378 | -1.621 | +1.621 |
| varied_full | varied_masked | copy | nomatch | +0.101 | +0.447 | -0.346 | +0.346 |
| repeat_full | unique | copy | nomatch | -1.254 | +0.367 | -1.621 | +1.621 |
| repeat_full | support_control | copy | nomatch | -1.243 | +0.378 | -1.621 | +1.621 |
| varied_full | unique | copy | nomatch | +0.095 | +0.441 | -0.346 | +0.346 |
| varied_full | support_control | copy | nomatch | +0.106 | +0.453 | -0.346 | +0.346 |
| repeat_full | varied_full | copy | nomatch | -1.350 | -0.075 | -1.275 | +1.275 |
| repeat_full | repeat_masked | copy | wrongmatch | -1.243 | +0.443 | -1.686 | +1.686 |
| varied_full | varied_masked | copy | wrongmatch | +0.101 | +0.465 | -0.364 | +0.364 |
| repeat_full | unique | copy | wrongmatch | -1.254 | +0.432 | -1.686 | +1.686 |
| repeat_full | support_control | copy | wrongmatch | -1.243 | +0.443 | -1.686 | +1.686 |
| varied_full | unique | copy | wrongmatch | +0.095 | +0.459 | -0.363 | +0.363 |
| varied_full | support_control | copy | wrongmatch | +0.106 | +0.470 | -0.364 | +0.364 |
| repeat_full | varied_full | copy | wrongmatch | -1.350 | -0.027 | -1.323 | +1.323 |

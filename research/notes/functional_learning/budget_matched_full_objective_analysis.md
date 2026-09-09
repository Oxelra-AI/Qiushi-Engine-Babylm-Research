# budget matched design note analysis: budget-matched full-objective trajectories

Read from `experiments/archive/functional_learning/data/budget_matched_full_objective/results.json`. This analysis uses one unified trained-entity success definition: top4≥0.95, B-swap mean≥5, Q-swap mean≥5, B/Q positive fractions≥0.95, Q-swap both≥0.90, and query-novel selectivity≥0.80. It also reports a held-entity behavior summary using held top4, held B-swap, and held corruption; held Q-swap is mixed because one side queries a trained entity.

## Arm-level timing and endpoint summary

| arm | success at P+500 | success at 1000 | sustained success at 1000 | mean first strong | mean post-switch first strong | mean P+500 top4 | mean final top4 | mean final held_top4 | mean final held_B | mean final held_sel |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fresh_qfirst_full | 2/3 | 2/3 | 2/3 | 487.5 | 75.5 | 0.745 | 0.757 | 0.243 | 0.375 | -0.018 |
| prep_bound_ans_then_full | 3/3 | 3/3 | 3/3 | 400.0 | 275.0 | 0.997 | 1.000 | 0.389 | 3.119 | 0.188 |
| prep_bag_ans_then_full | 3/3 | 3/3 | 2/3 | 883.3 | 483.3 | 0.997 | 0.997 | 0.115 | -1.816 | -0.145 |
| prep_ctx_then_full | 2/3 | 2/3 | 2/3 | 650.0 | 250.0 | 0.870 | 0.867 | 0.189 | -1.003 | -0.075 |

## Per-seed states at preparation end P

| seed | arm | P | phase/mode | top4 | Bmean | Bfrac | Qmean | Qfrac | Qboth | sel | held4 | heldB | heldSel |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | fresh_qfirst_full | 300 | full/full | 0.258 | 0.007 | 0.506 | -0.001 | 0.504 | 0.012 | -0.002 | 0.211 | -0.060 | -0.012 |
| 42 | prep_bound_ans_then_full | 300 | prep/answer_only | 0.986 | 10.180 | 0.998 | 9.850 | 1.000 | 0.965 | 0.953 | 0.484 | 3.833 | 0.365 |
| 42 | prep_bag_ans_then_full | 300 | prep/answer_only | 0.215 | -0.002 | 0.490 | -0.007 | 0.445 | 0.004 | 0.003 | 0.227 | 0.017 | 0.012 |
| 42 | prep_ctx_then_full | 300 | prep/context_only | 0.275 | -0.012 | 0.451 | -0.026 | 0.465 | 0.000 | -0.000 | 0.254 | -0.016 | -0.000 |
| 43 | fresh_qfirst_full | 500 | full/full | 1.000 | 9.903 | 1.000 | 9.578 | 1.000 | 1.000 | 0.996 | 0.379 | 1.589 | 0.132 |
| 43 | prep_bound_ans_then_full | 500 | prep/answer_only | 1.000 | 7.645 | 1.000 | 7.649 | 1.000 | 0.996 | 0.983 | 0.820 | 5.504 | 0.726 |
| 43 | prep_bag_ans_then_full | 500 | prep/answer_only | 0.238 | -0.001 | 0.508 | -0.002 | 0.453 | 0.000 | -0.006 | 0.242 | 0.002 | -0.003 |
| 43 | prep_ctx_then_full | 500 | prep/context_only | 0.232 | 0.002 | 0.514 | 0.004 | 0.512 | 0.000 | 0.000 | 0.254 | -0.008 | -0.000 |
| 100 | fresh_qfirst_full | 400 | full/full | 0.297 | 0.011 | 0.523 | -0.000 | 0.500 | 0.000 | -0.001 | 0.238 | 0.089 | 0.010 |
| 100 | prep_bound_ans_then_full | 400 | prep/answer_only | 1.000 | 9.462 | 1.000 | 9.498 | 1.000 | 1.000 | 0.997 | 0.953 | 7.774 | 0.914 |
| 100 | prep_bag_ans_then_full | 400 | prep/answer_only | 0.250 | 0.006 | 0.514 | -0.003 | 0.484 | 0.008 | 0.002 | 0.262 | -0.011 | -0.003 |
| 100 | prep_ctx_then_full | 400 | prep/context_only | 0.232 | -0.011 | 0.471 | -0.018 | 0.469 | 0.000 | -0.000 | 0.250 | 0.001 | -0.000 |

## Per-seed matched branching experiment-total state P+500

| seed | arm | epoch | top4 | Bmean | Bfrac | Qmean | Qfrac | Qboth | sel | held4 | heldB | heldBfrac | heldQ | heldSel | strong? |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | fresh_qfirst_full | 800 | 0.234 | 0.037 | 0.553 | 0.003 | 0.527 | 0.012 | 0.005 | 0.219 | -0.061 | 0.438 | 0.006 | -0.026 | False |
| 42 | prep_bound_ans_then_full | 800 | 0.992 | 14.972 | 1.000 | 14.913 | 1.000 | 0.992 | 0.995 | 0.461 | 5.046 | 0.641 | 9.513 | 0.293 | True |
| 42 | prep_bag_ans_then_full | 800 | 1.000 | 7.311 | 1.000 | 7.091 | 1.000 | 1.000 | 0.979 | 0.000 | -2.853 | 0.121 | 2.760 | -0.290 | True |
| 42 | prep_ctx_then_full | 800 | 1.000 | 11.894 | 1.000 | 12.004 | 1.000 | 1.000 | 0.999 | 0.000 | -3.790 | 0.246 | 4.095 | -0.335 | True |
| 43 | fresh_qfirst_full | 1000 | 1.000 | 12.251 | 1.000 | 12.024 | 1.000 | 1.000 | 0.999 | 0.418 | 2.619 | 0.652 | 6.591 | 0.181 | True |
| 43 | prep_bound_ans_then_full | 1000 | 1.000 | 8.837 | 0.998 | 8.987 | 1.000 | 1.000 | 0.984 | 0.457 | 3.022 | 0.723 | 5.208 | 0.325 | True |
| 43 | prep_bag_ans_then_full | 1000 | 0.990 | 5.920 | 0.998 | 5.904 | 1.000 | 0.977 | 0.922 | 0.223 | 0.264 | 0.523 | 2.839 | 0.014 | True |
| 43 | prep_ctx_then_full | 1000 | 1.000 | 11.391 | 1.000 | 11.399 | 1.000 | 1.000 | 0.999 | 0.258 | 0.503 | 0.594 | 5.456 | -0.031 | True |
| 100 | fresh_qfirst_full | 900 | 1.000 | 11.204 | 1.000 | 11.260 | 1.000 | 1.000 | 0.998 | 0.121 | -1.246 | 0.492 | 4.756 | -0.169 | True |
| 100 | prep_bound_ans_then_full | 900 | 0.998 | 12.217 | 1.000 | 12.736 | 1.000 | 0.996 | 0.990 | 0.246 | 1.400 | 0.617 | 5.828 | 0.000 | True |
| 100 | prep_bag_ans_then_full | 900 | 1.000 | 8.741 | 1.000 | 8.874 | 1.000 | 1.000 | 0.993 | 0.109 | -1.777 | 0.250 | 3.748 | -0.158 | True |
| 100 | prep_ctx_then_full | 900 | 0.609 | 3.588 | 0.895 | 3.774 | 0.820 | 0.402 | 0.495 | 0.324 | 0.577 | 0.598 | 1.737 | 0.099 | False |

## Per-seed final state at 1000 cumulative epochs

| seed | arm | top4 | Bmean | Bfrac | Qmean | Qfrac | Qboth | sel | held4 | heldB | heldBfrac | heldQ | heldSel | strong? | held-behavior? | first strong | sustained first | post-switch first |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|
| 42 | fresh_qfirst_full | 0.270 | 0.039 | 0.578 | 0.017 | 0.559 | 0.027 | 0.008 | 0.223 | -0.114 | 0.359 | 0.012 | -0.033 | False | False |  |  |  |
| 42 | prep_bound_ans_then_full | 1.000 | 15.457 | 1.000 | 15.560 | 1.000 | 1.000 | 0.999 | 0.473 | 5.300 | 0.668 | 9.858 | 0.268 | True | False | 300 | 450 | 150 |
| 42 | prep_bag_ans_then_full | 1.000 | 10.124 | 1.000 | 9.907 | 1.000 | 0.996 | 0.995 | 0.000 | -3.904 | 0.086 | 3.541 | -0.306 | True | False | 800 | 800 | 500 |
| 42 | prep_ctx_then_full | 1.000 | 13.094 | 1.000 | 13.132 | 1.000 | 1.000 | 0.999 | 0.000 | -4.379 | 0.227 | 4.314 | -0.339 | True | False | 575 | 575 | 275 |
| 43 | fresh_qfirst_full | 1.000 | 12.251 | 1.000 | 12.024 | 1.000 | 1.000 | 0.999 | 0.418 | 2.619 | 0.652 | 6.591 | 0.181 | True | False | 425 | 425 | 1 |
| 43 | prep_bound_ans_then_full | 1.000 | 8.837 | 0.998 | 8.987 | 1.000 | 1.000 | 0.984 | 0.457 | 3.022 | 0.723 | 5.208 | 0.325 | True | False | 500 | 900 | 400 |
| 43 | prep_bag_ans_then_full | 0.990 | 5.920 | 0.998 | 5.904 | 1.000 | 0.977 | 0.922 | 0.223 | 0.264 | 0.523 | 2.839 | 0.014 | True | False | 1000 |  | 500 |
| 43 | prep_ctx_then_full | 1.000 | 11.391 | 1.000 | 11.399 | 1.000 | 1.000 | 0.999 | 0.258 | 0.503 | 0.594 | 5.456 | -0.031 | True | False | 725 | 725 | 225 |
| 100 | fresh_qfirst_full | 1.000 | 11.914 | 1.000 | 11.959 | 1.000 | 1.000 | 0.999 | 0.090 | -1.379 | 0.461 | 4.972 | -0.204 | True | False | 550 | 550 | 150 |
| 100 | prep_bound_ans_then_full | 1.000 | 13.901 | 1.000 | 14.285 | 1.000 | 1.000 | 0.997 | 0.238 | 1.035 | 0.594 | 6.320 | -0.029 | True | False | 400 | 675 | 275 |
| 100 | prep_bag_ans_then_full | 1.000 | 10.035 | 1.000 | 10.277 | 1.000 | 1.000 | 0.997 | 0.121 | -1.808 | 0.250 | 4.441 | -0.144 | True | False | 850 | 850 | 450 |
| 100 | prep_ctx_then_full | 0.602 | 4.020 | 0.871 | 4.213 | 0.836 | 0.336 | 0.489 | 0.309 | 0.867 | 0.645 | 2.109 | 0.146 | False | False |  |  |  |

## query first binding compact summary qfirst weighted-full w16 baseline at 500 epochs

This is not budget-matched to the 1000-epoch budget matched design note run, but it tests whether simply upweighting the answer within the full objective solved the task by 500 epochs in the previous experiment.

| seed | arm | top4 | Bmean | Qmean | Qboth | sel | held4 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 42 | qfirst_full_500 | 0.246 | -0.000 | -0.003 | 0.000 | 0.001 | 0.180 |
| 43 | qfirst_full_500 | 1.000 | 9.721 | 9.766 | 1.000 | 0.998 | 0.414 |
| 100 | qfirst_full_500 | 0.307 | 0.130 | 0.121 | 0.035 | 0.034 | 0.254 |
| 42 | qfirst_w16_500 | 0.254 | 0.000 | 0.002 | 0.000 | 0.004 | 0.227 |
| 43 | qfirst_w16_500 | 0.221 | -0.004 | 0.000 | 0.000 | -0.006 | 0.195 |
| 100 | qfirst_w16_500 | 0.289 | 0.018 | 0.014 | 0.004 | 0.003 | 0.254 |
| 42 | qfirst_ans_only_500 | 1.000 | 15.413 | 15.620 | 1.000 | 1.000 | 0.504 |
| 43 | qfirst_ans_only_500 | 0.994 | 7.578 | 7.667 | 0.996 | 0.978 | 0.809 |
| 100 | qfirst_ans_only_500 | 1.000 | 11.122 | 11.090 | 1.000 | 1.000 | 0.965 |

## Direct interpretation from the observed budget matched design note pattern

- At the matched branching experiment-total point P+500, bound preparation succeeds in 3/3 seeds and fresh full succeeds in 2/3 seeds under the unified criterion.
- The equally trained unbound bag-answer preparation succeeds in 3/3 seeds at P+500.
- The context-only preparation succeeds in 2/3 seeds at P+500.
- Bound post-switch full-objective epochs to first strong: [150, 400, 275]. Fresh cumulative full-objective epochs to first strong: [, 425, 550].
- Treat these numbers as paired seed trajectories, not population rates. The next scientific step depends on whether bound preparation is uniquely earlier/stronger than fresh and unbound controls, and whether held-entity probes move with trained-entity binding.

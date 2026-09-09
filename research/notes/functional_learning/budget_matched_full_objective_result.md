# budget matched design note result: budget-matched full-objective comparison

This note is auto-generated from `data/budget_matched_full_objective/results.json`. It compares fresh query-first full-objective training with bound and unbound preparation under the same cumulative epoch budget.

## Arm-level summary

| arm | success at P+500 | final success | mean P+500 top4 | mean final top4 | mean held final top4 | first strong epochs | first strong after full epochs |
|---|---:|---:|---:|---:|---:|---|---|
| fresh_qfirst_full | 2/3 | 2/3 | 0.745 | 0.757 | 0.243 | [None, 425, 550] | [None, 501, 550] |
| prep_bound_ans_then_full | 3/3 | 3/3 | 0.997 | 1.000 | 0.389 | [300, 500, 400] | [450, 900, 675] |
| prep_bag_ans_then_full | 3/3 | 3/3 | 0.997 | 0.997 | 0.115 | [800, 1000, 850] | [800, 1000, 850] |
| prep_ctx_then_full | 2/3 | 2/3 | 0.870 | 0.867 | 0.189 | [575, 725, None] | [575, 725, None] |

## Per-seed matched branching experiment-total readout (epoch P+500)

| seed | arm | epoch | top4 | B-swap | Q-swap | held_top4 | held_B-swap | held_Q-swap |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | fresh_qfirst_full | 800 | 0.234 | 0.037 | 0.003 | 0.219 | -0.061 | 0.006 |
| 42 | prep_bound_ans_then_full | 800 | 0.992 | 14.972 | 14.913 | 0.461 | 5.046 | 9.513 |
| 42 | prep_bag_ans_then_full | 800 | 1.000 | 7.311 | 7.091 | 0.000 | -2.853 | 2.760 |
| 42 | prep_ctx_then_full | 800 | 1.000 | 11.894 | 12.004 | 0.000 | -3.790 | 4.095 |
| 43 | fresh_qfirst_full | 1000 | 1.000 | 12.251 | 12.024 | 0.418 | 2.619 | 6.591 |
| 43 | prep_bound_ans_then_full | 1000 | 1.000 | 8.837 | 8.987 | 0.457 | 3.022 | 5.208 |
| 43 | prep_bag_ans_then_full | 1000 | 0.990 | 5.920 | 5.904 | 0.223 | 0.264 | 2.839 |
| 43 | prep_ctx_then_full | 1000 | 1.000 | 11.391 | 11.399 | 0.258 | 0.503 | 5.456 |
| 100 | fresh_qfirst_full | 900 | 1.000 | 11.204 | 11.260 | 0.121 | -1.246 | 4.756 |
| 100 | prep_bound_ans_then_full | 900 | 0.998 | 12.217 | 12.736 | 0.246 | 1.400 | 5.828 |
| 100 | prep_bag_ans_then_full | 900 | 1.000 | 8.741 | 8.874 | 0.109 | -1.777 | 3.748 |
| 100 | prep_ctx_then_full | 900 | 0.609 | 3.588 | 3.774 | 0.324 | 0.577 | 1.737 |

## Per-seed final readout

| seed | arm | epoch | top4 | B-swap | Q-swap | held_top4 | held_B-swap | held_Q-swap | first strong | first strong after full |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | fresh_qfirst_full | 1000 | 0.270 | 0.039 | 0.017 | 0.223 | -0.114 | 0.012 |  |  |
| 42 | prep_bound_ans_then_full | 1000 | 1.000 | 15.457 | 15.560 | 0.473 | 5.300 | 9.858 | 300 | 450 |
| 42 | prep_bag_ans_then_full | 1000 | 1.000 | 10.124 | 9.907 | 0.000 | -3.904 | 3.541 | 800 | 800 |
| 42 | prep_ctx_then_full | 1000 | 1.000 | 13.094 | 13.132 | 0.000 | -4.379 | 4.314 | 575 | 575 |
| 43 | fresh_qfirst_full | 1000 | 1.000 | 12.251 | 12.024 | 0.418 | 2.619 | 6.591 | 425 | 501 |
| 43 | prep_bound_ans_then_full | 1000 | 1.000 | 8.837 | 8.987 | 0.457 | 3.022 | 5.208 | 500 | 900 |
| 43 | prep_bag_ans_then_full | 1000 | 0.990 | 5.920 | 5.904 | 0.223 | 0.264 | 2.839 | 1000 | 1000 |
| 43 | prep_ctx_then_full | 1000 | 1.000 | 11.391 | 11.399 | 0.258 | 0.503 | 5.456 | 725 | 725 |
| 100 | fresh_qfirst_full | 1000 | 1.000 | 11.914 | 11.959 | 0.090 | -1.379 | 4.972 | 550 | 550 |
| 100 | prep_bound_ans_then_full | 1000 | 1.000 | 13.901 | 14.285 | 0.238 | 1.035 | 6.320 | 400 | 675 |
| 100 | prep_bag_ans_then_full | 1000 | 1.000 | 10.035 | 10.277 | 0.121 | -1.808 | 4.441 | 850 | 850 |
| 100 | prep_ctx_then_full | 1000 | 0.602 | 4.020 | 4.213 | 0.309 | 0.867 | 2.109 |  |  |

## Interpretation criteria

- If fresh full reaches the same strong-binding state by the same cumulative budget, the endpoint 3/3 versus 1/3 result from branching experiment cannot be used as a curriculum-efficiency result; only timing and held/generalization differences remain.
- If bound preparation reaches strong full-objective binding earlier than fresh full and earlier than unbound bag/context preparation, that supports retained selector structure increasing the value of later full-objective examples.
- If unbound preparation matches bound preparation, the advantage is more likely generic representation/bag-output pretraining rather than retained entity-specific selector knowledge.
- Held-entity B/Q-swap probes are included here because held standard top4 alone can mix token-generalization with bag/family effects.

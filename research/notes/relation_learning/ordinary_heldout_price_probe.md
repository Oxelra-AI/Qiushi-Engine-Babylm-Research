# ordinary heldout price probe ordinary held-out MLM price probe

This readout scores deterministic-mask MLM loss on the same 6,992 ordinary held-out rows for seed43022 CLEAN, REPEAT, VIEW, REPEAT_SPLIT, and VIEW_SPLIT. It tests whether the local-vs-split price seen on compact-rewrite N targets also appears on ordinary held-out text.

## Late mean loss

| role | arm | mean loss | n rows |
|---|---|---:|---:|
| C | D_C_43022 | 2.5102 | 6992 |
| R | D_R_43022 | 2.6176 | 6992 |
| RS | D_RS_43022 | 2.5966 | 6992 |
| V | D_V_43022 | 2.6096 | 6992 |
| VS | D_VS_43022 | 2.5894 | 6992 |

## Arm contrasts

| contrast | late Δloss | row-paired Δloss | reading |
|---|---:|---:|---|
| RminusRS | +0.0210 | +0.0210 | local exact recurrence versus split exact recurrence |
| VminusVS | +0.0203 | +0.0203 | local restatement versus split restatement |
| RminusC | +0.1074 | +0.1074 | local exact recurrence versus CLEAN |
| RSminusC | +0.0864 | +0.0864 | split exact recurrence versus CLEAN |
| VminusC | +0.0994 | +0.0994 | local restatement versus CLEAN |
| VSminusC | +0.0791 | +0.0791 | split restatement versus CLEAN |
| VminusR | -0.0080 | -0.0080 |  |
| VSminusRS | -0.0073 | -0.0073 |  |

The compact-rewrite N price was about +0.3027 for R−RS and +0.2688 for V−VS. On ordinary held-out rows the corresponding deltas are R−RS +0.0210 and V−VS +0.0203. If these ordinary deltas are near zero while compact N is large, the price is concentrated on the compact companion target family rather than broad held-out language fit.

Data outputs: `experiments/archive/relation_learning/data/ordinary_heldout_price_probe`.

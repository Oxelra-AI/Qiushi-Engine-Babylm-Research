# ordinary heldout price probe ordinary held-out MLM price probe

This readout scores deterministic-mask MLM loss on the same 6,992 ordinary held-out rows for seed43022 CLEAN, REPEAT, VIEW, REPEAT_SPLIT, and VIEW_SPLIT. It tests whether the local-vs-split price seen on compact-rewrite N targets also appears on ordinary held-out text.

## Late mean loss

| role | arm | mean loss | n rows |
|---|---|---:|---:|
| ALN | ALN | 2.5084 | 6992 |
| OFF | OFF | 2.5212 | 6992 |

## Arm contrasts

| contrast | late Δloss | row-paired Δloss | reading |
|---|---:|---:|---|

The compact-rewrite N price was about +0.3027 for R−RS and +0.2688 for V−VS. On ordinary held-out rows the corresponding deltas are R−RS +nan and V−VS +nan. If these ordinary deltas are near zero while compact N is large, the price is concentrated on the compact companion target family rather than broad held-out language fit.

Data outputs: `experiments/archive/relation_learning/data/ordinary_heldout_price_probe`.

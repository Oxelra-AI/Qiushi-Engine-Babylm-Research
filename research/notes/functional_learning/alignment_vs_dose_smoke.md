# alignment vs dose alignment versus context-dose controls

## Final means

| arm | target/kind | lr | final h4 | final hB | final hSel | train4 | blocked4 | ctx CE | ans CE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| context_only_lr_half | bound/context_only | 1.5e-04 | 0.625 | +5.267 | +0.746 | 1.000 | 0.208 | 24.143 | 0.005 |
| slot0_static_1over17 | slot0/static | 3.0e-04 | 0.271 | +1.288 | +0.082 | 0.406 | 0.250 | 21.625 | 1.271 |

## Per-seed final h4/hB

| seed | context_only_lr_half | slot0_static_1over17 |
|---:|---:|---:|
| 100 | 0.625/+5.27 | 0.271/+1.29 |

## Interpretation scaffold

Compare these endpoints to Step021b's bound static w=1/17 arm (mean held_top4 0.694, held_B +7.548, context CE 1.238). If context_only_lr_half collapses, then reducing context-update scale without answer credit is insufficient. If slot0_static_1over17 collapses while learning its visible deterministic answer distribution, then learnable answer/RWT pressure at the matched coefficient is insufficient unless the answer remains compatible with the query-conditioned binding relation.

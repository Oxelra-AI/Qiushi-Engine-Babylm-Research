# alignment vs dose alignment versus context-dose controls

## Final means

| arm | target/kind | lr | final h4 | final hB | final hSel | train4 | blocked4 | ctx CE | ans CE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| context_only_lr_half | bound/context_only | 1.5e-04 | 0.272 | -0.144 | +0.015 | 0.280 | 0.257 | 1.237 | 3.964 |
| slot0_static_1over17 | slot0/static | 3.0e-04 | 0.238 | +0.578 | -0.035 | 0.221 | 0.221 | 1.232 | 0.000 |

## Per-seed final h4/hB

| seed | context_only_lr_half | slot0_static_1over17 |
|---:|---:|---:|
| 42 | 0.281/-0.38 | 0.238/+0.52 |
| 43 | 0.258/+0.04 | 0.238/+0.68 |
| 100 | 0.277/-0.09 | 0.238/+0.54 |

## Interpretation scaffold

Compare these endpoints to Step021b's bound static w=1/17 arm (mean held_top4 0.694, held_B +7.548, context CE 1.238). If context_only_lr_half collapses, then reducing context-update scale without answer credit is insufficient. If slot0_static_1over17 collapses while learning its visible deterministic answer distribution, then learnable answer/RWT pressure at the matched coefficient is insufficient unless the answer remains compatible with the query-conditioned binding relation.

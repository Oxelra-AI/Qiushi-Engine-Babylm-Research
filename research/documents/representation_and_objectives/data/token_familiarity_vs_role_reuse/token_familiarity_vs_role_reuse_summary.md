# orbit pair audit and slot reuse update token familiarity versus relation-role reuse

Device `cuda`, elapsed 89.41 s, 4 seeds per cell.

## Held-pool token exposure in training

| mode | covered held-pool types | total occ | mean occ/type | min | max |
|---|---:|---:|---:|---:|---:|
| absent | 0/32 | 0 | 0.0 | 0 | 0 |
| neutral | 32/32 | 19200 | 600.0 | 240 | 1040 |
| anchor_true | 32/32 | 19200 | 600.0 | 240 | 1040 |
| anchor_flip | 32/32 | 19200 | 600.0 | 240 | 1040 |

## Held-family probe accuracy with held-pool argument fillers

| mode | arm | fit | train | probe-heldpool | anchor-heldpool | heldtemplate-heldpool | trainpool-heldfam probe | trainpool-trainfam probe |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| absent | zero | 4/4 | 1.000 | 0.625 | 0.826 | 0.626 | 0.737 | 0.740 |
| absent | true | 4/4 | 1.000 | 0.745 | 0.844 | 0.688 | 0.843 | 0.881 |
| absent | shuffled | 4/4 | 0.999 | 0.368 | 0.854 | 0.627 | 0.326 | 0.322 |
| neutral | zero | 4/4 | 1.000 | 0.716 | 0.902 | 0.747 | 0.737 | 0.772 |
| neutral | true | 4/4 | 1.000 | 0.812 | 0.883 | 0.740 | 0.875 | 0.889 |
| neutral | shuffled | 4/4 | 0.999 | 0.342 | 0.918 | 0.731 | 0.309 | 0.297 |
| anchor_true | zero | 4/4 | 1.000 | 0.754 | 0.943 | 0.792 | 0.741 | 0.781 |
| anchor_true | true | 4/4 | 1.000 | 0.860 | 0.936 | 0.758 | 0.843 | 0.882 |
| anchor_true | shuffled | 4/4 | 0.999 | 0.320 | 0.938 | 0.679 | 0.325 | 0.319 |
| anchor_flip | zero | 4/4 | 1.000 | 0.260 | 0.099 | 0.268 | 0.754 | 0.777 |
| anchor_flip | true | 4/4 | 1.000 | 0.141 | 0.080 | 0.243 | 0.841 | 0.892 |
| anchor_flip | shuffled | 4/4 | 1.000 | 0.573 | 0.093 | 0.264 | 0.350 | 0.357 |

## Probe-coordinate separation

| mode | zero | true | shuffled | true-shuffled | true-zero | held token train coverage | mean occ/type |
|---|---:|---:|---:|---:|---:|---:|---:|
| absent | 0.625 | 0.745 | 0.368 | +0.377 | +0.119 | 0.000 | 0.0 |
| neutral | 0.716 | 0.812 | 0.342 | +0.471 | +0.096 | 1.000 | 600.0 |
| anchor_true | 0.754 | 0.860 | 0.320 | +0.540 | +0.106 | 1.000 | 600.0 |
| anchor_flip | 0.260 | 0.141 | 0.573 | -0.432 | -0.119 | 1.000 | 600.0 |

Summary JSON: `experiments/archive/representation_and_objectives/data/token_familiarity_vs_role_reuse/token_familiarity_vs_role_reuse_summary.json`

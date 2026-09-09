# format sensitivity boundary inline-role bridge

Tests whether co-located role annotations avoid the coordinate collision
that role coordinate collision and route established for separated declaration sentences.
Held readouts meaningful only after sparse changed-focal rows fit.

## inline_direct

Best train: 0.5000

Fit by kind: `{"base_stable_anchor|ctx=inline_role|q=direct_tag": 0.5, "base_stable_train|ctx=inline_role|q=direct_tag": 0.5, "sparse_changed_focal|ctx=inline_role|q=direct_tag": 0.5, "sparse_stable_focal|ctx=inline_role|q=direct_tag": 0.5}`

| eval | con | fb | fa | sb | sa | fa_m | sa_m |
|---|---:|---:|---:|---:|---:|---:|---:|
| inline_direct_hC_hS_direct_hH | 0.438 | 0.250 | 1.000 | 0.250 | 0.250 | 0.00 | -0.00 |
| inline_direct_hC_hS_rolePara_hH | 0.562 | 1.000 | 0.000 | 0.750 | 0.500 | -0.00 | -0.00 |
| inline_direct_hC_hS_roleSwap_direct_hH | 0.250 | 1.000 | 0.000 | 0.000 | 0.000 | -0.00 | -0.00 |
| inline_direct_hC_hS_roleSwap_role_hH | 0.250 | 1.000 | 0.000 | 0.000 | 0.000 | -0.00 | -0.00 |
| inline_direct_hC_hS_role_hH | 0.250 | 0.000 | 1.000 | 0.000 | 0.000 | 0.00 | -0.00 |


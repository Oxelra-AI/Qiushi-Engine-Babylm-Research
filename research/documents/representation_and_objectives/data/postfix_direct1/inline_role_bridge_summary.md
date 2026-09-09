# format sensitivity boundary inline-role bridge

Tests whether co-located role annotations avoid the coordinate collision
that role coordinate collision and route established for separated declaration sentences.
Held readouts meaningful only after sparse changed-focal rows fit.

## postfix_direct

Best train: 0.9776

Fit by kind: `{"base_stable_anchor|ctx=postfix_role|q=direct_tag": 1.0, "base_stable_train|ctx=postfix_role|q=direct_tag": 1.0, "sparse_changed_focal|ctx=postfix_role|q=direct_tag": 0.5078125, "sparse_stable_focal|ctx=postfix_role|q=direct_tag": 1.0}`

| eval | con | fb | fa | sb | sa | fa_m | sa_m |
|---|---:|---:|---:|---:|---:|---:|---:|
| postfix_direct_hC_hS_direct_hH | 0.750 | 0.000 | 1.000 | 1.000 | 1.000 | 0.97 | 6.13 |
| postfix_direct_hC_hS_rolePara_hH | 0.750 | 0.500 | 0.500 | 1.000 | 1.000 | 2.19 | 4.36 |
| postfix_direct_hC_hS_roleSwap_direct_hH | 0.750 | 1.000 | 0.000 | 1.000 | 1.000 | -5.10 | 4.32 |
| postfix_direct_hC_hS_roleSwap_role_hH | 0.750 | 1.000 | 0.000 | 1.000 | 1.000 | -1.77 | 3.57 |
| postfix_direct_hC_hS_role_hH | 0.750 | 0.250 | 0.750 | 1.000 | 1.000 | -1.68 | 4.40 |


# address bootstrap and role query role-query bootstrap probe

Question: after direct address retrieval is fitted, can sparse role-query training learn to select records by context role words rather than by a tag in the hypothesis?

Mode: `inline_role`; shared namespace: `s269_inline_direct` suffix `_direct_tag`.

## Fits

### scratch_role

Best train acc: 1.0

Fit: `{"base_stable_anchor|ctx=inline_role|q=role_inline": 1.0, "base_stable_train|ctx=inline_role|q=role_inline": 1.0, "sparse_changed_focal|ctx=inline_role|q=role_inline": 1.0, "sparse_stable_focal|ctx=inline_role|q=role_inline": 1.0}`

## Evaluations

### scratch_after_role

| eval | con | fb | fa | sb | sa | fa_m | sa_m |
|---|---:|---:|---:|---:|---:|---:|---:|
| inline_role_hC_hS_direct_hH | 0.750 | 0.000 | 1.000 | 1.000 | 1.000 | 13.80 | 24.38 |
| inline_role_hC_hS_rolePara_hH | 0.750 | 0.000 | 1.000 | 1.000 | 1.000 | 13.28 | 25.07 |
| inline_role_hC_hS_roleSwap_direct_hH | 0.750 | 0.000 | 1.000 | 1.000 | 1.000 | 15.16 | 23.86 |
| inline_role_hC_hS_roleSwap_role_hH | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 21.09 | 25.29 |
| inline_role_hC_hS_role_hH | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 26.25 | 24.76 |
| inline_role_tC_hS_direct_hH | 0.750 | 0.500 | 0.500 | 1.000 | 1.000 | 0.75 | 22.60 |
| inline_role_tC_hS_rolePara_hH | 0.750 | 0.500 | 0.500 | 1.000 | 1.000 | 1.07 | 24.78 |
| inline_role_tC_hS_roleSwap_direct_hH | 0.688 | 0.500 | 0.250 | 1.000 | 1.000 | -0.29 | 23.27 |
| inline_role_tC_hS_roleSwap_role_hH | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 23.05 | 24.60 |
| inline_role_tC_hS_role_hH | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 23.21 | 24.40 |


# address bootstrap and role query role-query bootstrap probe

Question: after direct address retrieval is fitted, can sparse role-query training learn to select records by context role words rather than by a tag in the hypothesis?

Mode: `inline_role`; shared namespace: `s269_inline_direct` suffix `_direct_tag`.

## Fits

### scratch_role

Best train acc: 0.5

Fit: `{"base_stable_anchor|ctx=inline_role|q=role_inline": 0.5, "base_stable_train|ctx=inline_role|q=role_inline": 0.5, "sparse_changed_focal|ctx=inline_role|q=role_inline": 0.5, "sparse_stable_focal|ctx=inline_role|q=role_inline": 0.5}`

## Evaluations

### scratch_after_role

| eval | con | fb | fa | sb | sa | fa_m | sa_m |
|---|---:|---:|---:|---:|---:|---:|---:|
| inline_role_hC_hS_direct_hH | 0.312 | 0.000 | 1.000 | 0.250 | 0.000 | 0.00 | -0.00 |
| inline_role_hC_hS_rolePara_hH | 0.250 | 0.000 | 1.000 | 0.000 | 0.000 | 0.00 | -0.00 |
| inline_role_hC_hS_roleSwap_direct_hH | 0.312 | 0.000 | 1.000 | 0.250 | 0.000 | 0.00 | -0.00 |
| inline_role_hC_hS_roleSwap_role_hH | 0.250 | 1.000 | 0.000 | 0.000 | 0.000 | -0.00 | -0.00 |
| inline_role_hC_hS_role_hH | 0.250 | 0.000 | 1.000 | 0.000 | 0.000 | 0.00 | -0.00 |
| inline_role_tC_hS_direct_hH | 0.250 | 0.000 | 1.000 | 0.000 | 0.000 | 0.00 | -0.00 |
| inline_role_tC_hS_rolePara_hH | 0.250 | 0.000 | 1.000 | 0.000 | 0.000 | 0.00 | -0.00 |
| inline_role_tC_hS_roleSwap_direct_hH | 0.250 | 0.000 | 1.000 | 0.000 | 0.000 | 0.00 | -0.00 |
| inline_role_tC_hS_roleSwap_role_hH | 0.250 | 1.000 | 0.000 | 0.000 | 0.000 | -0.00 | -0.00 |
| inline_role_tC_hS_role_hH | 0.250 | 0.000 | 1.000 | 0.000 | 0.000 | 0.00 | -0.00 |


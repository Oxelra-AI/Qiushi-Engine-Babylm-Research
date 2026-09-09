# address bootstrap and role query role-query bootstrap probe

Question: after direct address retrieval is fitted, can sparse role-query training learn to select records by context role words rather than by a tag in the hypothesis?

Mode: `inline_role`; shared namespace: `s269_inline_direct` suffix `_direct_tag`.

## Fits

### warm_tag_pretrain

Best train acc: 0.9996448863636364

Fit: `{"base_stable_anchor|ctx=tag_only|q=direct_tag": 1.0, "base_stable_train|ctx=tag_only|q=direct_tag": 1.0, "sparse_changed_focal|ctx=tag_only|q=direct_tag": 0.9921875, "sparse_stable_focal|ctx=tag_only|q=direct_tag": 1.0}`

### warm_inline_direct

Best train acc: 1.0

Fit: `{"base_stable_anchor|ctx=inline_role|q=direct_tag": 1.0, "base_stable_train|ctx=inline_role|q=direct_tag": 1.0, "sparse_changed_focal|ctx=inline_role|q=direct_tag": 1.0, "sparse_stable_focal|ctx=inline_role|q=direct_tag": 1.0}`

### warm_role_query

Best train acc: 0.9786931818181818

Fit: `{"base_stable_anchor|ctx=inline_role|q=role_inline": 1.0, "base_stable_train|ctx=inline_role|q=role_inline": 1.0, "sparse_changed_focal|ctx=inline_role|q=role_inline": 0.53125, "sparse_stable_focal|ctx=inline_role|q=role_inline": 1.0}`

### warm_role_tag_fit_retention

Best train acc: None

Fit: `{"base_stable_anchor|ctx=tag_only|q=direct_tag": 1.0, "base_stable_train|ctx=tag_only|q=direct_tag": 1.0, "sparse_changed_focal|ctx=tag_only|q=direct_tag": 0.5234375, "sparse_stable_focal|ctx=tag_only|q=direct_tag": 1.0}`

## Evaluations

### warm_after_tag_on_role_suite

| eval | con | fb | fa | sb | sa | fa_m | sa_m |
|---|---:|---:|---:|---:|---:|---:|---:|
| inline_role_hC_hS_direct_hH | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 14.91 | 23.15 |
| inline_role_hC_hS_rolePara_hH | 0.750 | 1.000 | 0.000 | 1.000 | 1.000 | -5.38 | 15.36 |
| inline_role_hC_hS_roleSwap_direct_hH | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 13.77 | 22.89 |
| inline_role_hC_hS_roleSwap_role_hH | 0.750 | 0.000 | 1.000 | 1.000 | 1.000 | 6.67 | 15.14 |
| inline_role_hC_hS_role_hH | 0.750 | 1.000 | 0.000 | 1.000 | 1.000 | -5.16 | 15.54 |
| inline_role_tC_hS_direct_hH | 0.938 | 0.750 | 1.000 | 1.000 | 1.000 | 16.50 | 20.97 |
| inline_role_tC_hS_rolePara_hH | 0.750 | 0.750 | 0.250 | 1.000 | 1.000 | -1.07 | 11.99 |
| inline_role_tC_hS_roleSwap_direct_hH | 0.875 | 0.500 | 1.000 | 1.000 | 1.000 | 18.57 | 21.19 |
| inline_role_tC_hS_roleSwap_role_hH | 0.812 | 0.750 | 0.500 | 1.000 | 1.000 | -3.59 | 11.15 |
| inline_role_tC_hS_role_hH | 0.750 | 0.750 | 0.250 | 1.000 | 1.000 | -0.22 | 11.87 |

### warm_after_direct_on_role_suite

| eval | con | fb | fa | sb | sa | fa_m | sa_m |
|---|---:|---:|---:|---:|---:|---:|---:|
| inline_role_hC_hS_direct_hH | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 26.28 | 27.23 |
| inline_role_hC_hS_rolePara_hH | 0.625 | 0.250 | 0.250 | 1.000 | 1.000 | -0.18 | 26.80 |
| inline_role_hC_hS_roleSwap_direct_hH | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 26.32 | 27.22 |
| inline_role_hC_hS_roleSwap_role_hH | 0.812 | 0.750 | 0.500 | 1.000 | 1.000 | -0.02 | 26.56 |
| inline_role_hC_hS_role_hH | 0.625 | 0.250 | 0.250 | 1.000 | 1.000 | 0.57 | 26.13 |
| inline_role_tC_hS_direct_hH | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 19.48 | 27.26 |
| inline_role_tC_hS_rolePara_hH | 0.750 | 0.500 | 0.500 | 1.000 | 1.000 | -5.25 | 25.85 |
| inline_role_tC_hS_roleSwap_direct_hH | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 21.98 | 27.34 |
| inline_role_tC_hS_roleSwap_role_hH | 0.750 | 0.250 | 0.750 | 1.000 | 1.000 | 1.53 | 25.78 |
| inline_role_tC_hS_role_hH | 0.750 | 0.500 | 0.500 | 1.000 | 1.000 | -4.49 | 25.85 |

### warm_after_role_on_role_suite

| eval | con | fb | fa | sb | sa | fa_m | sa_m |
|---|---:|---:|---:|---:|---:|---:|---:|
| inline_role_hC_hS_direct_hH | 0.750 | 0.250 | 0.750 | 1.000 | 1.000 | 9.58 | 15.18 |
| inline_role_hC_hS_rolePara_hH | 0.750 | 0.500 | 0.500 | 1.000 | 1.000 | 5.50 | 15.16 |
| inline_role_hC_hS_roleSwap_direct_hH | 0.812 | 0.500 | 0.750 | 1.000 | 1.000 | 6.83 | 16.32 |
| inline_role_hC_hS_roleSwap_role_hH | 0.688 | 0.500 | 0.250 | 1.000 | 1.000 | -4.13 | 15.66 |
| inline_role_hC_hS_role_hH | 0.750 | 0.500 | 0.500 | 1.000 | 1.000 | 5.53 | 13.18 |
| inline_role_tC_hS_direct_hH | 0.875 | 1.000 | 0.500 | 1.000 | 1.000 | -5.37 | 12.64 |
| inline_role_tC_hS_rolePara_hH | 0.875 | 1.000 | 0.500 | 1.000 | 1.000 | -4.30 | 10.24 |
| inline_role_tC_hS_roleSwap_direct_hH | 0.750 | 0.750 | 0.250 | 1.000 | 1.000 | -4.41 | 9.55 |
| inline_role_tC_hS_roleSwap_role_hH | 0.812 | 0.250 | 1.000 | 1.000 | 1.000 | 6.10 | 11.08 |
| inline_role_tC_hS_role_hH | 0.812 | 1.000 | 0.250 | 1.000 | 1.000 | -6.19 | 12.74 |

### warm_after_role_direct_retention

| eval | con | fb | fa | sb | sa | fa_m | sa_m |
|---|---:|---:|---:|---:|---:|---:|---:|
| inline_role_heldChanged_heldStable_direct_hH | 0.312 | 1.000 | 0.000 | 0.000 | 0.250 | -1.22 | -0.03 |
| inline_role_trainChanged_heldStable_direct_hH | 0.938 | 0.750 | 1.000 | 1.000 | 1.000 | 3.83 | 4.85 |


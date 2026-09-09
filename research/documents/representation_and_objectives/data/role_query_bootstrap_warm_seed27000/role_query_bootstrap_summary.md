# address bootstrap and role query role-query bootstrap probe

Question: after direct address retrieval is fitted, can sparse role-query training learn to select records by context role words rather than by a tag in the hypothesis?

Mode: `inline_role`; shared namespace: `s269_inline_direct` suffix `_direct_tag`.

## Fits

### warm_tag_pretrain

Best train acc: 1.0

Fit: `{"base_stable_anchor|ctx=tag_only|q=direct_tag": 1.0, "base_stable_train|ctx=tag_only|q=direct_tag": 1.0, "sparse_changed_focal|ctx=tag_only|q=direct_tag": 1.0, "sparse_stable_focal|ctx=tag_only|q=direct_tag": 1.0}`

### warm_inline_direct

Best train acc: 1.0

Fit: `{"base_stable_anchor|ctx=inline_role|q=direct_tag": 1.0, "base_stable_train|ctx=inline_role|q=direct_tag": 1.0, "sparse_changed_focal|ctx=inline_role|q=direct_tag": 1.0, "sparse_stable_focal|ctx=inline_role|q=direct_tag": 1.0}`

### warm_role_query

Best train acc: 1.0

Fit: `{"base_stable_anchor|ctx=inline_role|q=role_inline": 1.0, "base_stable_train|ctx=inline_role|q=role_inline": 1.0, "sparse_changed_focal|ctx=inline_role|q=role_inline": 1.0, "sparse_stable_focal|ctx=inline_role|q=role_inline": 1.0}`

### warm_role_tag_fit_retention

Best train acc: None

Fit: `{"base_stable_anchor|ctx=tag_only|q=direct_tag": 1.0, "base_stable_train|ctx=tag_only|q=direct_tag": 1.0, "sparse_changed_focal|ctx=tag_only|q=direct_tag": 0.8515625, "sparse_stable_focal|ctx=tag_only|q=direct_tag": 1.0}`

## Evaluations

### warm_after_tag_on_role_suite

| eval | con | fb | fa | sb | sa | fa_m | sa_m |
|---|---:|---:|---:|---:|---:|---:|---:|
| inline_role_hC_hS_direct_hH | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 18.93 | 24.12 |
| inline_role_hC_hS_rolePara_hH | 0.812 | 0.500 | 0.750 | 1.000 | 1.000 | 0.25 | 19.49 |
| inline_role_hC_hS_roleSwap_direct_hH | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 18.20 | 23.51 |
| inline_role_hC_hS_roleSwap_role_hH | 0.875 | 0.500 | 1.000 | 1.000 | 1.000 | 1.60 | 19.10 |
| inline_role_hC_hS_role_hH | 0.938 | 0.750 | 1.000 | 1.000 | 1.000 | 0.56 | 19.88 |
| inline_role_tC_hS_direct_hH | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 22.17 | 24.54 |
| inline_role_tC_hS_rolePara_hH | 0.750 | 0.750 | 0.250 | 1.000 | 1.000 | -0.48 | 18.53 |
| inline_role_tC_hS_roleSwap_direct_hH | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 21.58 | 24.18 |
| inline_role_tC_hS_roleSwap_role_hH | 0.750 | 0.250 | 0.750 | 1.000 | 1.000 | 2.27 | 19.57 |
| inline_role_tC_hS_role_hH | 0.750 | 0.750 | 0.250 | 1.000 | 1.000 | -0.13 | 18.68 |

### warm_after_direct_on_role_suite

| eval | con | fb | fa | sb | sa | fa_m | sa_m |
|---|---:|---:|---:|---:|---:|---:|---:|
| inline_role_hC_hS_direct_hH | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 26.07 | 27.30 |
| inline_role_hC_hS_rolePara_hH | 0.688 | 0.250 | 0.500 | 1.000 | 1.000 | -0.78 | 22.47 |
| inline_role_hC_hS_roleSwap_direct_hH | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 25.88 | 27.08 |
| inline_role_hC_hS_roleSwap_role_hH | 0.812 | 0.750 | 0.500 | 1.000 | 1.000 | 3.16 | 23.23 |
| inline_role_hC_hS_role_hH | 0.750 | 0.250 | 0.750 | 1.000 | 1.000 | 1.68 | 22.95 |
| inline_role_tC_hS_direct_hH | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 20.36 | 19.33 |
| inline_role_tC_hS_rolePara_hH | 0.812 | 0.750 | 0.500 | 1.000 | 1.000 | -2.10 | 9.83 |
| inline_role_tC_hS_roleSwap_direct_hH | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 23.52 | 19.42 |
| inline_role_tC_hS_roleSwap_role_hH | 0.812 | 0.250 | 1.000 | 1.000 | 1.000 | 15.42 | 11.81 |
| inline_role_tC_hS_role_hH | 0.812 | 0.750 | 0.500 | 1.000 | 1.000 | -1.13 | 9.04 |

### warm_after_role_on_role_suite

| eval | con | fb | fa | sb | sa | fa_m | sa_m |
|---|---:|---:|---:|---:|---:|---:|---:|
| inline_role_hC_hS_direct_hH | 0.812 | 0.500 | 0.750 | 1.000 | 1.000 | 5.42 | 22.62 |
| inline_role_hC_hS_rolePara_hH | 0.750 | 0.000 | 1.000 | 1.000 | 1.000 | 22.23 | 22.77 |
| inline_role_hC_hS_roleSwap_direct_hH | 0.812 | 0.750 | 0.500 | 1.000 | 1.000 | -5.04 | 25.74 |
| inline_role_hC_hS_roleSwap_role_hH | 0.750 | 0.000 | 1.000 | 1.000 | 1.000 | 24.13 | 24.85 |
| inline_role_hC_hS_role_hH | 0.875 | 0.500 | 1.000 | 1.000 | 1.000 | 24.27 | 22.67 |
| inline_role_tC_hS_direct_hH | 0.875 | 0.750 | 0.750 | 1.000 | 1.000 | 7.88 | 29.59 |
| inline_role_tC_hS_rolePara_hH | 0.812 | 0.250 | 1.000 | 1.000 | 1.000 | 21.54 | 29.80 |
| inline_role_tC_hS_roleSwap_direct_hH | 0.875 | 1.000 | 0.500 | 1.000 | 1.000 | 2.10 | 29.57 |
| inline_role_tC_hS_roleSwap_role_hH | 0.750 | 0.000 | 1.000 | 1.000 | 1.000 | 24.83 | 29.86 |
| inline_role_tC_hS_role_hH | 0.875 | 0.500 | 1.000 | 1.000 | 1.000 | 24.41 | 29.82 |

### warm_after_role_direct_retention

| eval | con | fb | fa | sb | sa | fa_m | sa_m |
|---|---:|---:|---:|---:|---:|---:|---:|
| inline_role_heldChanged_heldStable_direct_hH | 0.812 | 0.250 | 1.000 | 1.000 | 1.000 | 14.70 | 23.42 |
| inline_role_trainChanged_heldStable_direct_hH | 0.812 | 0.500 | 0.750 | 1.000 | 1.000 | 3.45 | 23.12 |


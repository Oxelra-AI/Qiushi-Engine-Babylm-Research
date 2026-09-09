# address bootstrap and role query prefix warm-start probe

Question: does a previously learned `Entry TAG:` address route make a minimally prefixed `The entry TAG:` format learnable under identical supervision?

Prefix mode: `inline_role`; tag pretrain mode: `tag_only`.

## Construction

- tag_train: {'n': 5632, 'by_query': {'focal_after': 1536, 'focal_before': 1536, 'secondary_after': 1280, 'secondary_before': 1280}, 'by_label': {'0': 2816, '1': 2816}, 'by_kind': {'base_stable_anchor|ctx=tag_only|q=direct_tag': 2560, 'base_stable_train|ctx=tag_only|q=direct_tag': 2560, 'sparse_changed_focal|ctx=tag_only|q=direct_tag': 256, 'sparse_stable_focal|ctx=tag_only|q=direct_tag': 256}, 'by_changed_focal': {'False': 5376, 'True': 256}, 'by_stable_secondary': {'True': 5632}}
- prefix_train: {'n': 5632, 'by_query': {'focal_after': 1536, 'focal_before': 1536, 'secondary_after': 1280, 'secondary_before': 1280}, 'by_label': {'0': 2816, '1': 2816}, 'by_kind': {'base_stable_anchor|ctx=inline_role|q=direct_tag': 2560, 'base_stable_train|ctx=inline_role|q=direct_tag': 2560, 'sparse_changed_focal|ctx=inline_role|q=direct_tag': 256, 'sparse_stable_focal|ctx=inline_role|q=direct_tag': 256}, 'by_changed_focal': {'False': 5376, 'True': 256}, 'by_stable_secondary': {'True': 5632}}

## Main fits

### tag_pretrain

Best train acc: 0.9998224431818182

Fit: `{"base_stable_anchor|ctx=tag_only|q=direct_tag": 1.0, "base_stable_train|ctx=tag_only|q=direct_tag": 1.0, "sparse_changed_focal|ctx=tag_only|q=direct_tag": 0.99609375, "sparse_stable_focal|ctx=tag_only|q=direct_tag": 1.0}`

### warm_prefix

Best train acc: 0.9992897727272727

Fit: `{"base_stable_anchor|ctx=inline_role|q=direct_tag": 1.0, "base_stable_train|ctx=inline_role|q=direct_tag": 1.0, "sparse_changed_focal|ctx=inline_role|q=direct_tag": 0.984375, "sparse_stable_focal|ctx=inline_role|q=direct_tag": 1.0}`

### optional_tag_refresh

Best train acc: None

Fit: `{"base_stable_anchor|ctx=tag_only|q=direct_tag": 0.99921875, "base_stable_train|ctx=tag_only|q=direct_tag": 1.0, "sparse_changed_focal|ctx=tag_only|q=direct_tag": 1.0, "sparse_stable_focal|ctx=tag_only|q=direct_tag": 1.0}`

## Evaluation summary

### after_tag_on_tag

| eval | con | fb | fa | sb | sa | fa_m | sa_m |
|---|---:|---:|---:|---:|---:|---:|---:|
| tag_only_heldChanged_heldStable_direct_hH | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 16.74 | 20.23 |
| tag_only_trainChanged_heldStable_direct_hH | 0.938 | 1.000 | 0.750 | 1.000 | 1.000 | 9.94 | 22.33 |

### after_tag_on_prefix_before_continuation

| eval | con | fb | fa | sb | sa | fa_m | sa_m |
|---|---:|---:|---:|---:|---:|---:|---:|
| inline_role_heldChanged_heldStable_direct_hH | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 19.51 | 20.65 |
| inline_role_trainChanged_heldStable_direct_hH | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 13.52 | 23.93 |

### after_warm_on_prefix

| eval | con | fb | fa | sb | sa | fa_m | sa_m |
|---|---:|---:|---:|---:|---:|---:|---:|
| inline_role_heldChanged_heldStable_direct_hH | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 21.06 | 19.73 |
| inline_role_trainChanged_heldStable_direct_hH | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 15.70 | 22.48 |

### after_warm_on_tag_retention

| eval | con | fb | fa | sb | sa | fa_m | sa_m |
|---|---:|---:|---:|---:|---:|---:|---:|
| tag_only_heldChanged_heldStable_direct_hH | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 21.69 | 18.22 |
| tag_only_trainChanged_heldStable_direct_hH | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 16.43 | 23.48 |


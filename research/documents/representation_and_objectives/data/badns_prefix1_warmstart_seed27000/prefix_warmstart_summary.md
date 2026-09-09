# address bootstrap and role query prefix warm-start probe

Question: does a previously learned `Entry TAG:` address route make a minimally prefixed `The entry TAG:` format learnable under identical supervision?

Prefix mode: `prefix_one`; tag pretrain mode: `tag_only`.

## Construction

- tag_train: {'n': 5632, 'by_query': {'focal_after': 1536, 'focal_before': 1536, 'secondary_after': 1280, 'secondary_before': 1280}, 'by_label': {'0': 2816, '1': 2816}, 'by_kind': {'base_stable_anchor|ctx=tag_only|q=direct_tag': 2560, 'base_stable_train|ctx=tag_only|q=direct_tag': 2560, 'sparse_changed_focal|ctx=tag_only|q=direct_tag': 256, 'sparse_stable_focal|ctx=tag_only|q=direct_tag': 256}, 'by_changed_focal': {'False': 5376, 'True': 256}, 'by_stable_secondary': {'True': 5632}}
- prefix_train: {'n': 5632, 'by_query': {'focal_after': 1536, 'focal_before': 1536, 'secondary_after': 1280, 'secondary_before': 1280}, 'by_label': {'0': 2816, '1': 2816}, 'by_kind': {'base_stable_anchor|ctx=prefix_one|q=direct_tag': 2560, 'base_stable_train|ctx=prefix_one|q=direct_tag': 2560, 'sparse_changed_focal|ctx=prefix_one|q=direct_tag': 256, 'sparse_stable_focal|ctx=prefix_one|q=direct_tag': 256}, 'by_changed_focal': {'False': 5376, 'True': 256}, 'by_stable_secondary': {'True': 5632}}

## Main fits

### tag_pretrain

Best train acc: 1.0

Fit: `{"base_stable_anchor|ctx=tag_only|q=direct_tag": 1.0, "base_stable_train|ctx=tag_only|q=direct_tag": 1.0, "sparse_changed_focal|ctx=tag_only|q=direct_tag": 1.0, "sparse_stable_focal|ctx=tag_only|q=direct_tag": 1.0}`

### warm_prefix

Best train acc: 0.9996448863636364

Fit: `{"base_stable_anchor|ctx=prefix_one|q=direct_tag": 1.0, "base_stable_train|ctx=prefix_one|q=direct_tag": 1.0, "sparse_changed_focal|ctx=prefix_one|q=direct_tag": 0.9921875, "sparse_stable_focal|ctx=prefix_one|q=direct_tag": 1.0}`

### optional_tag_refresh

Best train acc: None

Fit: `{"base_stable_anchor|ctx=tag_only|q=direct_tag": 1.0, "base_stable_train|ctx=tag_only|q=direct_tag": 1.0, "sparse_changed_focal|ctx=tag_only|q=direct_tag": 0.99609375, "sparse_stable_focal|ctx=tag_only|q=direct_tag": 1.0}`

## Evaluation summary

### after_tag_on_tag

| eval | con | fb | fa | sb | sa | fa_m | sa_m |
|---|---:|---:|---:|---:|---:|---:|---:|
| tag_only_heldChanged_heldStable_direct_hH | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 28.41 | 29.89 |
| tag_only_trainChanged_heldStable_direct_hH | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 27.75 | 30.41 |

### after_tag_on_prefix_before_continuation

| eval | con | fb | fa | sb | sa | fa_m | sa_m |
|---|---:|---:|---:|---:|---:|---:|---:|
| prefix_one_heldChanged_heldStable_direct_hH | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 27.84 | 29.45 |
| prefix_one_trainChanged_heldStable_direct_hH | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 27.61 | 30.50 |

### after_warm_on_prefix

| eval | con | fb | fa | sb | sa | fa_m | sa_m |
|---|---:|---:|---:|---:|---:|---:|---:|
| prefix_one_heldChanged_heldStable_direct_hH | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 31.18 | 31.41 |
| prefix_one_trainChanged_heldStable_direct_hH | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 25.86 | 32.19 |

### after_warm_on_tag_retention

| eval | con | fb | fa | sb | sa | fa_m | sa_m |
|---|---:|---:|---:|---:|---:|---:|---:|
| tag_only_heldChanged_heldStable_direct_hH | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 31.05 | 31.37 |
| tag_only_trainChanged_heldStable_direct_hH | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 24.12 | 32.08 |


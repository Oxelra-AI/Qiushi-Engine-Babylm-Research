# address bootstrap and role query prefix warm-start probe

Question: does a previously learned `Entry TAG:` address route make a minimally prefixed `The entry TAG:` format learnable under identical supervision?

Prefix mode: `prefix_one`; tag pretrain mode: `tag_only`.

## Construction

- tag_train: {'n': 5632, 'by_query': {'focal_after': 1536, 'focal_before': 1536, 'secondary_after': 1280, 'secondary_before': 1280}, 'by_label': {'0': 2816, '1': 2816}, 'by_kind': {'base_stable_anchor|ctx=tag_only|q=direct_tag': 2560, 'base_stable_train|ctx=tag_only|q=direct_tag': 2560, 'sparse_changed_focal|ctx=tag_only|q=direct_tag': 256, 'sparse_stable_focal|ctx=tag_only|q=direct_tag': 256}, 'by_changed_focal': {'False': 5376, 'True': 256}, 'by_stable_secondary': {'True': 5632}}
- prefix_train: {'n': 5632, 'by_query': {'focal_after': 1536, 'focal_before': 1536, 'secondary_after': 1280, 'secondary_before': 1280}, 'by_label': {'0': 2816, '1': 2816}, 'by_kind': {'base_stable_anchor|ctx=prefix_one|q=direct_tag': 2560, 'base_stable_train|ctx=prefix_one|q=direct_tag': 2560, 'sparse_changed_focal|ctx=prefix_one|q=direct_tag': 256, 'sparse_stable_focal|ctx=prefix_one|q=direct_tag': 256}, 'by_changed_focal': {'False': 5376, 'True': 256}, 'by_stable_secondary': {'True': 5632}}

## Main fits

### tag_pretrain

Best train acc: 0.5005326704545454

Fit: `{"base_stable_anchor|ctx=tag_only|q=direct_tag": 0.5, "base_stable_train|ctx=tag_only|q=direct_tag": 0.5015625, "sparse_changed_focal|ctx=tag_only|q=direct_tag": 0.49609375, "sparse_stable_focal|ctx=tag_only|q=direct_tag": 0.5}`

### warm_prefix

Best train acc: 0.5

Fit: `{"base_stable_anchor|ctx=prefix_one|q=direct_tag": 0.5, "base_stable_train|ctx=prefix_one|q=direct_tag": 0.5, "sparse_changed_focal|ctx=prefix_one|q=direct_tag": 0.5, "sparse_stable_focal|ctx=prefix_one|q=direct_tag": 0.5}`

### optional_tag_refresh

Best train acc: None

Fit: `{"base_stable_anchor|ctx=tag_only|q=direct_tag": 0.5, "base_stable_train|ctx=tag_only|q=direct_tag": 0.5, "sparse_changed_focal|ctx=tag_only|q=direct_tag": 0.5, "sparse_stable_focal|ctx=tag_only|q=direct_tag": 0.5}`

## Evaluation summary

### after_tag_on_tag

| eval | con | fb | fa | sb | sa | fa_m | sa_m |
|---|---:|---:|---:|---:|---:|---:|---:|
| tag_only_heldChanged_heldStable_direct_hH | 0.375 | 1.000 | 0.000 | 0.250 | 0.250 | -0.00 | -0.00 |
| tag_only_trainChanged_heldStable_direct_hH | 0.375 | 0.000 | 1.000 | 0.000 | 0.500 | 0.00 | -0.00 |

### after_tag_on_prefix_before_continuation

| eval | con | fb | fa | sb | sa | fa_m | sa_m |
|---|---:|---:|---:|---:|---:|---:|---:|
| prefix_one_heldChanged_heldStable_direct_hH | 0.750 | 1.000 | 0.000 | 1.000 | 1.000 | -0.00 | 0.00 |
| prefix_one_trainChanged_heldStable_direct_hH | 0.250 | 1.000 | 0.000 | 0.000 | 0.000 | -0.00 | -0.00 |

### after_warm_on_prefix

| eval | con | fb | fa | sb | sa | fa_m | sa_m |
|---|---:|---:|---:|---:|---:|---:|---:|
| prefix_one_heldChanged_heldStable_direct_hH | 0.750 | 0.000 | 1.000 | 1.000 | 1.000 | 0.00 | 0.00 |
| prefix_one_trainChanged_heldStable_direct_hH | 0.250 | 0.250 | 0.750 | 0.000 | 0.000 | 0.00 | -0.00 |

### after_warm_on_tag_retention

| eval | con | fb | fa | sb | sa | fa_m | sa_m |
|---|---:|---:|---:|---:|---:|---:|---:|
| tag_only_heldChanged_heldStable_direct_hH | 0.250 | 0.000 | 1.000 | 0.000 | 0.000 | 0.00 | -0.00 |
| tag_only_trainChanged_heldStable_direct_hH | 0.250 | 0.750 | 0.250 | 0.000 | 0.000 | -0.00 | -0.00 |


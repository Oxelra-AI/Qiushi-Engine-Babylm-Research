# address bootstrap and role query prefix warm-start probe

Question: does a previously learned `Entry TAG:` address route make a minimally prefixed `The entry TAG:` format learnable under identical supervision?

Prefix mode: `inline_role`; tag pretrain mode: `tag_only`.

## Construction

- tag_train: {'n': 5632, 'by_query': {'focal_after': 1536, 'focal_before': 1536, 'secondary_after': 1280, 'secondary_before': 1280}, 'by_label': {'0': 2816, '1': 2816}, 'by_kind': {'base_stable_anchor|ctx=tag_only|q=direct_tag': 2560, 'base_stable_train|ctx=tag_only|q=direct_tag': 2560, 'sparse_changed_focal|ctx=tag_only|q=direct_tag': 256, 'sparse_stable_focal|ctx=tag_only|q=direct_tag': 256}, 'by_changed_focal': {'False': 5376, 'True': 256}, 'by_stable_secondary': {'True': 5632}}
- prefix_train: {'n': 5632, 'by_query': {'focal_after': 1536, 'focal_before': 1536, 'secondary_after': 1280, 'secondary_before': 1280}, 'by_label': {'0': 2816, '1': 2816}, 'by_kind': {'base_stable_anchor|ctx=inline_role|q=direct_tag': 2560, 'base_stable_train|ctx=inline_role|q=direct_tag': 2560, 'sparse_changed_focal|ctx=inline_role|q=direct_tag': 256, 'sparse_stable_focal|ctx=inline_role|q=direct_tag': 256}, 'by_changed_focal': {'False': 5376, 'True': 256}, 'by_stable_secondary': {'True': 5632}}

## Main fits

### scratch_prefix

Best train acc: 0.8600852272727273

Fit: `{"base_stable_anchor|ctx=inline_role|q=direct_tag": 0.87578125, "base_stable_train|ctx=inline_role|q=direct_tag": 0.878515625, "sparse_changed_focal|ctx=inline_role|q=direct_tag": 0.51171875, "sparse_stable_focal|ctx=inline_role|q=direct_tag": 0.8671875}`

## Evaluation summary

### scratch_prefix_on_prefix

| eval | con | fb | fa | sb | sa | fa_m | sa_m |
|---|---:|---:|---:|---:|---:|---:|---:|
| inline_role_heldChanged_heldStable_direct_hH | 0.688 | 0.500 | 0.250 | 1.000 | 1.000 | -0.08 | 3.10 |
| inline_role_trainChanged_heldStable_direct_hH | 0.750 | 0.500 | 0.500 | 1.000 | 1.000 | 0.42 | 2.31 |


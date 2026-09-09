# prephase alignment design and bias analysis prephase alignment — aligned

Seed: 27000. Prephase rows: 5632. Continuation rows: 5632.

## prephase

Best train acc: 1.000000

Fit: `{"base_stable_anchor|ctx=tag_only|q=direct_tag": 1.0, "base_stable_train|ctx=tag_only|q=direct_tag": 1.0, "sparse_changed_focal|ctx=tag_only|q=direct_tag": 1.0, "sparse_stable_focal|ctx=tag_only|q=direct_tag": 1.0}`

## continuation

Best train acc: 1.000000

Fit: `{"base_stable_anchor|ctx=inline_role|q=direct_tag": 1.0, "base_stable_train|ctx=inline_role|q=direct_tag": 1.0, "sparse_changed_focal|ctx=inline_role|q=direct_tag": 1.0, "sparse_stable_focal|ctx=inline_role|q=direct_tag": 1.0}`

## Evaluations

### after_continuation_inline

| eval | con | fb | fa | sb | sa | fa_m | sa_m |
|---|---:|---:|---:|---:|---:|---:|---:|
| inline_role_heldChanged_heldStable_direct_hH | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 24.60 | 28.89 |
| inline_role_trainChanged_heldStable_direct_hH | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 26.11 | 28.83 |

### after_continuation_tag

| eval | con | fb | fa | sb | sa | fa_m | sa_m |
|---|---:|---:|---:|---:|---:|---:|---:|
| tag_only_heldChanged_heldStable_direct_hH | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 19.73 | 27.48 |
| tag_only_trainChanged_heldStable_direct_hH | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 21.30 | 26.67 |

### after_prephase_inline

| eval | con | fb | fa | sb | sa | fa_m | sa_m |
|---|---:|---:|---:|---:|---:|---:|---:|
| inline_role_heldChanged_heldStable_direct_hH | 0.938 | 0.750 | 1.000 | 1.000 | 1.000 | 21.30 | 13.73 |
| inline_role_trainChanged_heldStable_direct_hH | 0.938 | 0.750 | 1.000 | 1.000 | 1.000 | 15.31 | 21.46 |

### after_prephase_tag

| eval | con | fb | fa | sb | sa | fa_m | sa_m |
|---|---:|---:|---:|---:|---:|---:|---:|
| tag_only_heldChanged_heldStable_direct_hH | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 24.31 | 20.57 |
| tag_only_trainChanged_heldStable_direct_hH | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 16.21 | 26.44 |

## Prephase retention after continuation

Fit: `{"base_stable_anchor|ctx=tag_only|q=direct_tag": 1.0, "base_stable_train|ctx=tag_only|q=direct_tag": 1.0, "sparse_changed_focal|ctx=tag_only|q=direct_tag": 1.0, "sparse_stable_focal|ctx=tag_only|q=direct_tag": 1.0}`


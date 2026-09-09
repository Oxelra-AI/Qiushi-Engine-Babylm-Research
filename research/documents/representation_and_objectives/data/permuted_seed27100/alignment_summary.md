# prephase alignment design and bias analysis prephase alignment — permuted

Seed: 27100. Prephase rows: 5632. Continuation rows: 5632.

## prephase

Best train acc: 0.500000

Fit: `{"base_stable_anchor|ctx=tag_only|q=direct_tag": 0.5, "base_stable_train|ctx=tag_only|q=direct_tag": 0.5, "sparse_changed_focal|ctx=tag_only|q=direct_tag": 0.5, "sparse_stable_focal|ctx=tag_only|q=direct_tag": 0.5}`

## continuation

Best train acc: 0.500000

Fit: `{"base_stable_anchor|ctx=inline_role|q=direct_tag": 0.5, "base_stable_train|ctx=inline_role|q=direct_tag": 0.5, "sparse_changed_focal|ctx=inline_role|q=direct_tag": 0.5, "sparse_stable_focal|ctx=inline_role|q=direct_tag": 0.5}`

## Evaluations

### after_continuation_inline

| eval | con | fb | fa | sb | sa | fa_m | sa_m |
|---|---:|---:|---:|---:|---:|---:|---:|
| inline_role_heldChanged_heldStable_direct_hH | 0.688 | 0.750 | 0.000 | 1.000 | 1.000 | -0.00 | 0.00 |
| inline_role_trainChanged_heldStable_direct_hH | 0.312 | 0.500 | 0.000 | 0.500 | 0.250 | -0.00 | -0.00 |

### after_continuation_tag

| eval | con | fb | fa | sb | sa | fa_m | sa_m |
|---|---:|---:|---:|---:|---:|---:|---:|
| tag_only_heldChanged_heldStable_direct_hH | 0.688 | 0.750 | 0.000 | 1.000 | 1.000 | -0.00 | 0.00 |
| tag_only_trainChanged_heldStable_direct_hH | 0.562 | 0.750 | 0.000 | 0.750 | 0.750 | -0.00 | 0.00 |

### after_prephase_inline

| eval | con | fb | fa | sb | sa | fa_m | sa_m |
|---|---:|---:|---:|---:|---:|---:|---:|
| inline_role_heldChanged_heldStable_direct_hH | 0.812 | 1.000 | 0.250 | 1.000 | 1.000 | -0.00 | 0.00 |
| inline_role_trainChanged_heldStable_direct_hH | 0.625 | 0.250 | 0.750 | 0.750 | 0.750 | 0.00 | 0.00 |

### after_prephase_tag

| eval | con | fb | fa | sb | sa | fa_m | sa_m |
|---|---:|---:|---:|---:|---:|---:|---:|
| tag_only_heldChanged_heldStable_direct_hH | 0.938 | 1.000 | 0.750 | 1.000 | 1.000 | 0.00 | 0.00 |
| tag_only_trainChanged_heldStable_direct_hH | 0.688 | 0.250 | 1.000 | 0.750 | 0.750 | 0.00 | 0.00 |

## Prephase retention after continuation

Fit: `{"base_stable_anchor|ctx=tag_only|q=direct_tag": 0.5, "base_stable_train|ctx=tag_only|q=direct_tag": 0.5, "sparse_changed_focal|ctx=tag_only|q=direct_tag": 0.5, "sparse_stable_focal|ctx=tag_only|q=direct_tag": 0.5}`


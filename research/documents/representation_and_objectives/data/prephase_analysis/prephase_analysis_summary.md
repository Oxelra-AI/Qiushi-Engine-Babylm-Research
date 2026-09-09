# prephase alignment design and bias analysis Prephase Alignment Analysis

## Decision table

| arm | pre fit changed | cont fit changed | cont_inline hC_hS con | cont inline fa | cont inline fb | micro_eebf | secondary | tag retention |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| aligned | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 |
| permuted | 0.996 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 |
| disjoint | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 |
| scratch | – | 0.500 | 0.688 | 0.750 | 0.000 | 0.750 | 1.000 | 0.750 |

Micro-EEBF = focal_after_acc − focal_before_acc on held changed/stable queries.
Near zero → genuine before/after discrimination; large positive → after-state bias.

## aligned (seed 27000)

**prephase** best_acc=1.000000
  - base_stable_anchor|ctx=tag_only|q=direct_tag: 1.0000
  - base_stable_train|ctx=tag_only|q=direct_tag: 1.0000
  - sparse_changed_focal|ctx=tag_only|q=direct_tag: 1.0000
  - sparse_stable_focal|ctx=tag_only|q=direct_tag: 1.0000

**continuation** best_acc=1.000000
  - base_stable_anchor|ctx=inline_role|q=direct_tag: 1.0000
  - base_stable_train|ctx=inline_role|q=direct_tag: 1.0000
  - sparse_changed_focal|ctx=inline_role|q=direct_tag: 1.0000
  - sparse_stable_focal|ctx=inline_role|q=direct_tag: 1.0000

### after_prephase_inline

| eval | con | fb | fa | sb | sa | micro_eebf | fa_m | sa_m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| inline_role_heldChanged_heldStable_direct_hH | 0.938 | 0.750 | 1.000 | 1.000 | 1.000 | 0.250 | 21.30 | 13.73 |
| inline_role_trainChanged_heldStable_direct_hH | 0.938 | 0.750 | 1.000 | 1.000 | 1.000 | 0.250 | 15.31 | 21.46 |

### after_prephase_tag

| eval | con | fb | fa | sb | sa | micro_eebf | fa_m | sa_m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tag_only_heldChanged_heldStable_direct_hH | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 24.31 | 20.57 |
| tag_only_trainChanged_heldStable_direct_hH | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 16.21 | 26.44 |

### after_continuation_inline

| eval | con | fb | fa | sb | sa | micro_eebf | fa_m | sa_m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| inline_role_heldChanged_heldStable_direct_hH | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 24.60 | 28.89 |
| inline_role_trainChanged_heldStable_direct_hH | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 26.11 | 28.83 |

### after_continuation_tag

| eval | con | fb | fa | sb | sa | micro_eebf | fa_m | sa_m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tag_only_heldChanged_heldStable_direct_hH | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 19.73 | 27.48 |
| tag_only_trainChanged_heldStable_direct_hH | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 21.30 | 26.67 |

## permuted (seed 27000)

**prephase** best_acc=0.999822
  - base_stable_anchor|ctx=tag_only|q=direct_tag: 1.0000
  - base_stable_train|ctx=tag_only|q=direct_tag: 1.0000
  - sparse_changed_focal|ctx=tag_only|q=direct_tag: 0.9961
  - sparse_stable_focal|ctx=tag_only|q=direct_tag: 1.0000

**continuation** best_acc=1.000000
  - base_stable_anchor|ctx=inline_role|q=direct_tag: 1.0000
  - base_stable_train|ctx=inline_role|q=direct_tag: 1.0000
  - sparse_changed_focal|ctx=inline_role|q=direct_tag: 1.0000
  - sparse_stable_focal|ctx=inline_role|q=direct_tag: 1.0000

### after_prephase_inline

| eval | con | fb | fa | sb | sa | micro_eebf | fa_m | sa_m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| inline_role_heldChanged_heldStable_direct_hH | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 19.50 | 22.47 |
| inline_role_trainChanged_heldStable_direct_hH | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 17.96 | 21.77 |

### after_prephase_tag

| eval | con | fb | fa | sb | sa | micro_eebf | fa_m | sa_m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tag_only_heldChanged_heldStable_direct_hH | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 21.89 | 21.85 |
| tag_only_trainChanged_heldStable_direct_hH | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 19.51 | 22.12 |

### after_continuation_inline

| eval | con | fb | fa | sb | sa | micro_eebf | fa_m | sa_m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| inline_role_heldChanged_heldStable_direct_hH | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 16.93 | 22.00 |
| inline_role_trainChanged_heldStable_direct_hH | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 17.16 | 20.32 |

### after_continuation_tag

| eval | con | fb | fa | sb | sa | micro_eebf | fa_m | sa_m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tag_only_heldChanged_heldStable_direct_hH | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 18.44 | 22.16 |
| tag_only_trainChanged_heldStable_direct_hH | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 17.67 | 21.41 |

## disjoint (seed 27000)

**prephase** best_acc=1.000000
  - base_stable_anchor|ctx=tag_only|q=direct_tag: 1.0000
  - base_stable_train|ctx=tag_only|q=direct_tag: 1.0000
  - sparse_changed_focal|ctx=tag_only|q=direct_tag: 1.0000
  - sparse_stable_focal|ctx=tag_only|q=direct_tag: 1.0000

**continuation** best_acc=1.000000
  - base_stable_anchor|ctx=inline_role|q=direct_tag: 1.0000
  - base_stable_train|ctx=inline_role|q=direct_tag: 1.0000
  - sparse_changed_focal|ctx=inline_role|q=direct_tag: 1.0000
  - sparse_stable_focal|ctx=inline_role|q=direct_tag: 1.0000

### after_prephase_inline

| eval | con | fb | fa | sb | sa | micro_eebf | fa_m | sa_m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| inline_role_heldChanged_heldStable_direct_hH | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 16.99 | 25.79 |
| inline_role_trainChanged_heldStable_direct_hH | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 14.02 | 23.37 |

### after_prephase_tag

| eval | con | fb | fa | sb | sa | micro_eebf | fa_m | sa_m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tag_only_heldChanged_heldStable_direct_hH | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 17.32 | 27.23 |
| tag_only_trainChanged_heldStable_direct_hH | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 18.87 | 22.33 |

### after_continuation_inline

| eval | con | fb | fa | sb | sa | micro_eebf | fa_m | sa_m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| inline_role_heldChanged_heldStable_direct_hH | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 26.92 | 31.10 |
| inline_role_trainChanged_heldStable_direct_hH | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 23.87 | 33.11 |

### after_continuation_tag

| eval | con | fb | fa | sb | sa | micro_eebf | fa_m | sa_m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tag_only_heldChanged_heldStable_direct_hH | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 27.85 | 31.64 |
| tag_only_trainChanged_heldStable_direct_hH | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 25.40 | 33.95 |

## scratch (seed 27000)

**continuation** best_acc=0.500888
  - base_stable_anchor|ctx=inline_role|q=direct_tag: 0.5000
  - base_stable_train|ctx=inline_role|q=direct_tag: 0.5020
  - sparse_changed_focal|ctx=inline_role|q=direct_tag: 0.5000
  - sparse_stable_focal|ctx=inline_role|q=direct_tag: 0.5000

### after_prephase_inline

| eval | con | fb | fa | sb | sa | micro_eebf | fa_m | sa_m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| inline_role_heldChanged_heldStable_direct_hH | 0.750 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.00 | 0.00 |
| inline_role_trainChanged_heldStable_direct_hH | 0.750 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.00 | 0.00 |

### after_prephase_tag

| eval | con | fb | fa | sb | sa | micro_eebf | fa_m | sa_m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tag_only_heldChanged_heldStable_direct_hH | 0.688 | 0.000 | 1.000 | 1.000 | 0.750 | 1.000 | 0.00 | 0.00 |
| tag_only_trainChanged_heldStable_direct_hH | 0.688 | 0.000 | 1.000 | 1.000 | 0.750 | 1.000 | 0.00 | 0.00 |

### after_continuation_inline

| eval | con | fb | fa | sb | sa | micro_eebf | fa_m | sa_m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| inline_role_heldChanged_heldStable_direct_hH | 0.688 | 0.000 | 0.750 | 1.000 | 1.000 | 0.750 | 0.00 | 0.00 |
| inline_role_trainChanged_heldStable_direct_hH | 0.250 | 0.500 | 0.500 | 0.000 | 0.000 | 0.000 | 0.00 | -0.00 |

### after_continuation_tag

| eval | con | fb | fa | sb | sa | micro_eebf | fa_m | sa_m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tag_only_heldChanged_heldStable_direct_hH | 0.750 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.00 | 0.00 |
| tag_only_trainChanged_heldStable_direct_hH | 0.375 | 0.750 | 0.250 | 0.250 | 0.250 | -0.500 | -0.00 | -0.00 |


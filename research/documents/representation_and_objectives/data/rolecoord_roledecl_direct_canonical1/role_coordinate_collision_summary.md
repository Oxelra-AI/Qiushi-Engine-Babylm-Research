# role coordinate collision and route role-coordinate collision probe

Small matched bridge separating context burden, duplicated tag mentions, declaration language, and role-to-coordinate indirection. Held readouts are meaningful only when sparse changed-focal rows fit.

## roledecl_direct

Best train acc: 0.500

Train fit by kind: `{"base_stable_anchor|ctx=roledecl|q=direct_tag": 0.5, "base_stable_train|ctx=roledecl|q=direct_tag": 0.5, "sparse_changed_focal|ctx=roledecl|q=direct_tag": 0.5, "sparse_stable_focal|ctx=roledecl|q=direct_tag": 0.5}`

| eval_set | con_acc | focal_before | focal_after | secondary_before | secondary_after | focal_after_margin | secondary_after_margin |
|---|---:|---:|---:|---:|---:|---:|---:|
| roledecl_direct_heldChanged_heldStable_direct_tag_heldHyp | 0.250 | 1.000 | 0.000 | 0.000 | 0.000 | -0.00 | -0.00 |
| roledecl_direct_heldChanged_heldStable_direct_tag_roleSwap_heldHyp | 0.688 | 0.750 | 0.250 | 1.000 | 0.750 | -0.00 | 0.00 |
| roledecl_direct_heldChanged_heldStable_role_exact_heldHyp | 0.438 | 1.000 | 0.250 | 0.250 | 0.250 | -0.00 | -0.00 |
| roledecl_direct_heldChanged_heldStable_role_exact_roleSwap_heldHyp | 0.750 | 0.000 | 1.000 | 1.000 | 1.000 | 0.00 | 0.00 |
| roledecl_direct_heldChanged_heldStable_role_sem_heldHyp | 0.750 | 0.000 | 1.000 | 1.000 | 1.000 | 0.00 | 0.00 |
| roledecl_direct_heldChanged_heldStable_role_sem_roleSwap_heldHyp | 0.750 | 1.000 | 0.000 | 1.000 | 1.000 | -0.00 | 0.00 |


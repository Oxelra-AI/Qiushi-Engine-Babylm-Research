# temporal addressability bridge two-hop temporal indirect address probe

Context binds a temporal/discourse role to a random entry tag; the queried relation is stored under that tag. Role-query rows ask via the role, not the tag. Direct-tag rows name the tag. Role-swap rows swap the declarations while labels retain intended temporal truth.

Train role families: ['background_update']

Eval role families: ['background_update', 'original_current_exact']

Train rows: base=1280 update=512 direct_tag_support=True

Best train accuracy: 0.500

Train fit by kind: `{"base_stable_direct|background_update|direct_tag_query": 0.5, "base_stable|background_update|role_query": 0.5, "sparse_changed_direct|background_update|direct_tag_query": 0.5, "sparse_changed_focal|background_update|role_query": 0.5, "sparse_stable_direct|background_update|direct_tag_query": 0.5, "sparse_stable_focal|background_update|role_query": 0.5}`

| eval_set | con_acc | focal_before | focal_after | secondary_before | secondary_after | focal_after_margin | secondary_after_margin |
|---|---:|---:|---:|---:|---:|---:|---:|
| background_update_heldChanged_heldStable_directTag_roleSwap_heldHyp | 0.312 | 0.750 | 0.500 | 0.000 | 0.000 | 0.00 | -0.00 |
| background_update_heldChanged_heldStable_direct_tag_heldHyp | 0.375 | 1.000 | 0.500 | 0.000 | 0.000 | -0.00 | -0.00 |
| background_update_heldChanged_heldStable_roleSwap_heldHyp | 0.750 | 0.000 | 1.000 | 1.000 | 1.000 | 0.00 | 0.00 |
| background_update_heldChanged_heldStable_role_heldHyp | 0.750 | 1.000 | 0.000 | 1.000 | 1.000 | -0.00 | 0.00 |
| original_current_exact_heldChanged_heldStable_directTag_roleSwap_heldHyp | 0.750 | 1.000 | 0.000 | 1.000 | 1.000 | -0.00 | 0.00 |
| original_current_exact_heldChanged_heldStable_direct_tag_heldHyp | 0.625 | 1.000 | 0.000 | 0.750 | 0.750 | -0.00 | 0.00 |
| original_current_exact_heldChanged_heldStable_roleSwap_heldHyp | 0.375 | 0.250 | 0.750 | 0.250 | 0.250 | 0.00 | 0.00 |
| original_current_exact_heldChanged_heldStable_role_heldHyp | 0.688 | 0.000 | 0.750 | 1.000 | 1.000 | 0.00 | 0.00 |

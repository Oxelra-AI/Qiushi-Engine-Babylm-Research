# temporal addressability bridge two-hop temporal indirect address probe

Context binds a temporal/discourse role to a random entry tag; the queried relation is stored under that tag. Role-query rows ask via the role, not the tag. Direct-tag rows name the tag. Role-swap rows swap the declarations while labels retain intended temporal truth.

Train role families: ['background_update', 'initial_latest', 'old_new']

Eval role families: ['original_current_exact', 'prior_revised_to_original_current', 'earlier_later_to_original_current']

Train rows: base=7680 update=3072 direct_tag_support=True

Best train accuracy: 0.502

Train fit by kind: `{"base_stable_direct|background_update|direct_tag_query": 0.5, "base_stable_direct|initial_latest|direct_tag_query": 0.50078125, "base_stable_direct|old_new|direct_tag_query": 0.49921875, "base_stable|background_update|role_query": 0.5125, "base_stable|initial_latest|role_query": 0.50234375, "base_stable|old_new|role_query": 0.5, "sparse_changed_direct|background_update|direct_tag_query": 0.50390625, "sparse_changed_direct|initial_latest|direct_tag_query": 0.49609375, "sparse_changed_direct|old_new|direct_tag_query": 0.5, "sparse_changed_focal|background_update|role_query": 0.49609375, "sparse_changed_focal|initial_latest|role_query": 0.5, "sparse_changed_focal|old_new|role_query": 0.5, "sparse_stable_direct|background_update|direct_tag_query": 0.5, "sparse_stable_direct|initial_latest|direct_tag_query": 0.49609375, "sparse_stable_direct|old_new|direct_tag_query": 0.5, "sparse_stable_focal|background_update|role_query": 0.5078125, "sparse_stable_focal|initial_latest|role_query": 0.5, "sparse_stable_focal|old_new|role_query": 0.50390625}`

| eval_set | con_acc | focal_before | focal_after | secondary_before | secondary_after | focal_after_margin | secondary_after_margin |
|---|---:|---:|---:|---:|---:|---:|---:|
| earlier_later_to_original_current_heldChanged_heldStable_directTag_roleSwap_heldHyp | 0.312 | 0.750 | 0.250 | 0.250 | 0.000 | -0.00 | -0.00 |
| earlier_later_to_original_current_heldChanged_heldStable_direct_tag_heldHyp | 0.688 | 0.250 | 0.750 | 1.000 | 0.750 | 0.00 | 0.00 |
| earlier_later_to_original_current_heldChanged_heldStable_roleSwap_heldHyp | 0.375 | 1.000 | 0.000 | 0.250 | 0.250 | -0.00 | -0.00 |
| earlier_later_to_original_current_heldChanged_heldStable_role_heldHyp | 0.438 | 0.250 | 0.750 | 0.500 | 0.250 | 0.00 | -0.00 |
| original_current_exact_heldChanged_heldStable_directTag_roleSwap_heldHyp | 0.250 | 0.000 | 1.000 | 0.000 | 0.000 | 0.00 | -0.00 |
| original_current_exact_heldChanged_heldStable_direct_tag_heldHyp | 0.688 | 0.250 | 0.750 | 1.000 | 0.750 | 0.00 | 0.00 |
| original_current_exact_heldChanged_heldStable_roleSwap_heldHyp | 0.188 | 0.000 | 0.750 | 0.000 | 0.000 | 0.00 | -0.00 |
| original_current_exact_heldChanged_heldStable_role_heldHyp | 0.750 | 0.500 | 0.500 | 1.000 | 1.000 | 0.00 | 0.00 |
| prior_revised_to_original_current_heldChanged_heldStable_directTag_roleSwap_heldHyp | 0.750 | 0.750 | 0.500 | 0.750 | 1.000 | -0.00 | 0.00 |
| prior_revised_to_original_current_heldChanged_heldStable_direct_tag_heldHyp | 0.688 | 1.000 | 0.000 | 0.750 | 1.000 | -0.00 | 0.00 |
| prior_revised_to_original_current_heldChanged_heldStable_roleSwap_heldHyp | 0.750 | 1.000 | 0.000 | 1.000 | 1.000 | -0.00 | 0.00 |
| prior_revised_to_original_current_heldChanged_heldStable_role_heldHyp | 0.250 | 1.000 | 0.000 | 0.000 | 0.000 | -0.00 | -0.00 |

# earlier analysis dense-mask/sparse-label causal reader

Created: 2026-09-07T22:30:41Z

This reader compares sparse `(S,S)`, dense-mask/sparse-label `(M,S)`, and dense `(M,M)` when real outputs exist. Missing dense-mask files are not interpreted.

## Arm status

| arm | present | role | focus targets | focus groups | fast mean | Entity | source-help Δ | common Δ | both-source |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| sparse_seed62064_SS | True | sparse labels, sparse masks | 28590 | 21479 | 44.545 | 27.8 | -0.0030773413105851487 | -0.010176720517347495 | 21 |
| dense_seed62064_MM | True | dense labels, dense masks | 176607 | 132283 | 44.80857142857143 | 28.44 | 0.10663173413263989 | 0.20787532545847906 | 21 |
| dense_seed62065_MM | True | dense labels, dense masks; fixed-policy replication | 176607 | 132283 | 44.777142857142856 | 28.43 | 0.11037357639799796 | 0.20564517308957875 | 21 |
| densemask_sparselabel_seed62064_MS | False | dense masks, sparse labels; causal arm | None | None | None | None | None | None | None |

## Core comparisons

### MS_minus_SS_input_mask_at_sparse_labels

Present: `False`

Reason: one_or_both_arms_missing

### MM64_minus_SS_dense_total_effect

Present: `True`

Fast Δ a-b: `{"BLiMP": -0.5699999999999932, "Supplement": -0.4000000000000057, "EWoK": 0.7299999999999969, "Entity": 0.6400000000000006, "COMPS": -0.00999999999999801, "GlobalPIQA_mean": 1.4849999999999994, "Reading": -0.030000000000001137, "equal_valid_mean": 0.26357142857143145}`

Qwen Δ a-b: `{"view_only_delta_nll_vs_parent": 0.16337536427635913, "with_source_delta_nll_vs_parent": 0.05366628883313408, "source_help_delta_vs_parent": 0.10970907544322504}`

Common Δ a-b: `{"mean_delta_expected_margin_vs_parent": 0.21805204597582656, "condition_deltas": {"no_source": 0.0019640217508588367, "source_altered": 0.39734482791940007, "source_original": 0.2548472882572207}, "split_deltas": {"held_source": 0.38461800676304847, "trained_content": 0.1396680644288986}, "both_source_conditions_correct_delta": 0.0, "mean_source_follow_swing_delta": 0.6521921161766198}`

### MS_minus_MM64_sparse_labels_vs_dense_coverage

Present: `False`

Reason: one_or_both_arms_missing

### MM65_minus_MM64_dense_replication

Present: `True`

Fast Δ a-b: `{"BLiMP": -0.020000000000010232, "Supplement": 0.0, "EWoK": -0.17999999999999972, "Entity": -0.010000000000001563, "COMPS": 0.0, "GlobalPIQA_mean": 0.0, "Reading": -0.00999999999999801, "equal_valid_mean": -0.031428571428577357}`

Qwen Δ a-b: `{"view_only_delta_nll_vs_parent": 0.0028485793415504446, "with_source_delta_nll_vs_parent": -0.000893262923807623, "source_help_delta_vs_parent": 0.003741842265358064}`

Common Δ a-b: `{"mean_delta_expected_margin_vs_parent": -0.002230152368900312, "condition_deltas": {"no_source": 0.0006430544455846565, "source_altered": -0.0009438199456781038, "source_original": -0.006389691606607462}, "split_deltas": {"held_source": 0.0006970790725770448, "trained_content": -0.0036076730472425766}, "both_source_conditions_correct_delta": 0.0, "mean_source_follow_swing_delta": -0.007333511552284122}`

## Verified implementation geometry

Verification present: `True` path `experiments/archive/functional_learning/data/densemask_sparselabel_fast_verify/densemask_sparselabel_fast_verify.json`

Core invariants: `{"rows_checked": 3831, "no_rows_over_dense_cap": true, "seed62064_labels_equal_original_sparse_by_group_logic": true, "seed62064_all_labels_masked": true, "seed62064_masks_equal_original_dense": true, "seed62064_masks_equal_original_dense_union_sparse": true, "seed62065_labels_equal_original_sparse_by_group_logic": true, "seed62065_all_labels_masked": true, "seed62065_masks_equal_original_dense": true, "sample_actual_constructor_status": "sample_actual_constructor_pass"}`

Dense/sparse target ratio: `6.177229800629591`; dense-mask/dense target ratio: `0.1618848630009003`; per-label focus-weight ratio dense-to-dm: `6.188765426100415`.

## Interpretation handle

Dense-mask/sparse-label outputs are not yet present. The script currently preserves the exact baselines and verified implementation geometry so that the causal-arm result can be interpreted when complete; it does not infer whether input masking or dense supervision caused dense gains.

# transition background for seed43122 signed-transition window background

CPU/file-only comparison of every eligible late three-checkpoint window against the fixed reference 82M→84M→86M signed-transition signature. This addresses the selection-induced rise/fall hazard in own-peak windows.

Reference: `scale1p75_seed43022_reference` fixed window `chck_82M->chck_84M->chck_86M` from `experiments/archive/frontier_consolidation/data/selected_trajectory_eval_scale1p75_seed43022_reference_common2M`.

## Target summaries

| target | complete endpoints | windows | own-peak window | fixed 82→84→86 present? |
|---|---:|---:|---|---|
| scale1p75_seed43022_reference | 16 | 14 | chck_82M->chck_84M->chck_86M | True |
| scale1p25_seed43022_dense | 16 | 14 | chck_84M->chck_86M->chck_88M | True |

## Key ranks against each target's own all-window background

Higher similarity is better. Rank 1 means the row is the most similar window to the fixed reference signature within that target trajectory. The fixed row is the same exposure coordinate 82→84→86; own-peak is selected by the target trajectory's cheap7.

| target | role | level | window | metric | value | rank | percentile | background median | background max |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| scale1p75_seed43022_reference | own_peak | column_official_subtask_mean | rise+fall_pattern | weighted_pearson_net_pct | 1.000000 | 1 | 1.000000 | 0.006533 | 1.000000 |
| scale1p75_seed43022_reference | fixed_82_84_86 | column_official_subtask_mean | rise+fall_pattern | weighted_pearson_net_pct | 1.000000 | 1 | 1.000000 | 0.006533 | 1.000000 |
| scale1p75_seed43022_reference | own_peak | column_official_subtask_mean | rise+fall_pattern | cosine_net_pct | 1.000000 | 1 | 1.000000 | -0.012526 | 1.000000 |
| scale1p75_seed43022_reference | fixed_82_84_86 | column_official_subtask_mean | rise+fall_pattern | cosine_net_pct | 1.000000 | 1 | 1.000000 | -0.012526 | 1.000000 |
| scale1p75_seed43022_reference | own_peak | column_subtask | rise+fall_pattern | weighted_pearson_net_pct | 1.000000 | 1 | 1.000000 | 0.047183 | 1.000000 |
| scale1p75_seed43022_reference | fixed_82_84_86 | column_subtask | rise+fall_pattern | weighted_pearson_net_pct | 1.000000 | 1 | 1.000000 | 0.047183 | 1.000000 |
| scale1p75_seed43022_reference | own_peak | column_subtask | rise+fall_pattern | cosine_net_pct | 1.000000 | 1 | 1.000000 | 0.041617 | 1.000000 |
| scale1p75_seed43022_reference | fixed_82_84_86 | column_subtask | rise+fall_pattern | cosine_net_pct | 1.000000 | 1 | 1.000000 | 0.041617 | 1.000000 |
| scale1p75_seed43022_reference | own_peak | item_pattern | rise+fall_pattern | signed_transition_pattern_cosine | 1.000000 | 1 | 1.000000 | -0.011650 | 1.000000 |
| scale1p75_seed43022_reference | fixed_82_84_86 | item_pattern | rise+fall_pattern | signed_transition_pattern_cosine | 1.000000 | 1 | 1.000000 | -0.011650 | 1.000000 |
| scale1p75_seed43022_reference | own_peak | item_pattern | rise+fall_pattern | same_signed_transition_on_ref_changed_fraction | 1.000000 | 1 | 1.000000 | 0.071524 | 1.000000 |
| scale1p75_seed43022_reference | fixed_82_84_86 | item_pattern | rise+fall_pattern | same_signed_transition_on_ref_changed_fraction | 1.000000 | 1 | 1.000000 | 0.071524 | 1.000000 |
| scale1p25_seed43022_dense | own_peak | column_official_subtask_mean | rise+fall_pattern | weighted_pearson_net_pct | 0.266155 | 5 | 0.714286 | 0.034973 | 0.358092 |
| scale1p25_seed43022_dense | fixed_82_84_86 | column_official_subtask_mean | rise+fall_pattern | weighted_pearson_net_pct | 0.332673 | 2 | 0.928571 | 0.034973 | 0.358092 |
| scale1p25_seed43022_dense | own_peak | column_official_subtask_mean | rise+fall_pattern | cosine_net_pct | 0.706816 | 1 | 1.000000 | -0.041442 | 0.706816 |
| scale1p25_seed43022_dense | fixed_82_84_86 | column_official_subtask_mean | rise+fall_pattern | cosine_net_pct | 0.228478 | 6 | 0.642857 | -0.041442 | 0.706816 |
| scale1p25_seed43022_dense | own_peak | column_subtask | rise+fall_pattern | weighted_pearson_net_pct | -0.241053 | 13 | 0.142857 | -0.024042 | 0.335592 |
| scale1p25_seed43022_dense | fixed_82_84_86 | column_subtask | rise+fall_pattern | weighted_pearson_net_pct | 0.335592 | 1 | 1.000000 | -0.024042 | 0.335592 |
| scale1p25_seed43022_dense | own_peak | column_subtask | rise+fall_pattern | cosine_net_pct | -0.202040 | 12 | 0.214286 | -0.029627 | 0.296070 |
| scale1p25_seed43022_dense | fixed_82_84_86 | column_subtask | rise+fall_pattern | cosine_net_pct | 0.296070 | 1 | 1.000000 | -0.029627 | 0.296070 |
| scale1p25_seed43022_dense | own_peak | item_pattern | rise+fall_pattern | signed_transition_pattern_cosine | -0.005165 | 13 | 0.142857 | 0.000335 | 0.016419 |
| scale1p25_seed43022_dense | fixed_82_84_86 | item_pattern | rise+fall_pattern | signed_transition_pattern_cosine | 0.016419 | 1 | 1.000000 | 0.000335 | 0.016419 |
| scale1p25_seed43022_dense | own_peak | item_pattern | rise+fall_pattern | same_signed_transition_on_ref_changed_fraction | 0.030956 | 8 | 0.500000 | 0.038739 | 0.080729 |
| scale1p25_seed43022_dense | fixed_82_84_86 | item_pattern | rise+fall_pattern | same_signed_transition_on_ref_changed_fraction | 0.049773 | 5 | 0.714286 | 0.038739 | 0.080729 |

## Best-matching windows by target

For a future seed43122 readout, recurring late-phase structure should make the own-peak and/or fixed 82→84→86 window stand out against these internal backgrounds at official-like column and subtask levels, not merely show an automatic rise/fall from peak selection.

| target | level | window | metric | best target window | best value | own-peak value | fixed value |
|---|---|---|---|---|---:|---:|---:|
| scale1p75_seed43022_reference | column_official_subtask_mean | rise | cosine_net_pct | chck_82M->chck_84M->chck_86M | 1.000000 | 1.000000 | 1.000000 |
| scale1p75_seed43022_reference | column_official_subtask_mean | rise | weighted_pearson_net_pct | chck_82M->chck_84M->chck_86M | 1.000000 | 1.000000 | 1.000000 |
| scale1p75_seed43022_reference | column_official_subtask_mean | fall | cosine_net_pct | chck_82M->chck_84M->chck_86M | 1.000000 | 1.000000 | 1.000000 |
| scale1p75_seed43022_reference | column_official_subtask_mean | fall | weighted_pearson_net_pct | chck_82M->chck_84M->chck_86M | 1.000000 | 1.000000 | 1.000000 |
| scale1p75_seed43022_reference | column_official_subtask_mean | rise+fall_pattern | cosine_net_pct | chck_82M->chck_84M->chck_86M | 1.000000 | 1.000000 | 1.000000 |
| scale1p75_seed43022_reference | column_official_subtask_mean | rise+fall_pattern | weighted_pearson_net_pct | chck_82M->chck_84M->chck_86M | 1.000000 | 1.000000 | 1.000000 |
| scale1p75_seed43022_reference | column_subtask | rise | cosine_net_pct | chck_82M->chck_84M->chck_86M | 1.000000 | 1.000000 | 1.000000 |
| scale1p75_seed43022_reference | column_subtask | rise | weighted_pearson_net_pct | chck_82M->chck_84M->chck_86M | 1.000000 | 1.000000 | 1.000000 |
| scale1p75_seed43022_reference | column_subtask | fall | cosine_net_pct | chck_82M->chck_84M->chck_86M | 1.000000 | 1.000000 | 1.000000 |
| scale1p75_seed43022_reference | column_subtask | fall | weighted_pearson_net_pct | chck_82M->chck_84M->chck_86M | 1.000000 | 1.000000 | 1.000000 |
| scale1p75_seed43022_reference | column_subtask | rise+fall_pattern | cosine_net_pct | chck_82M->chck_84M->chck_86M | 1.000000 | 1.000000 | 1.000000 |
| scale1p75_seed43022_reference | column_subtask | rise+fall_pattern | weighted_pearson_net_pct | chck_82M->chck_84M->chck_86M | 1.000000 | 1.000000 | 1.000000 |
| scale1p25_seed43022_dense | column_official_subtask_mean | rise | cosine_net_pct | chck_82M->chck_84M->chck_86M | 0.720579 | -0.016266 | 0.720579 |
| scale1p25_seed43022_dense | column_official_subtask_mean | rise | weighted_pearson_net_pct | chck_92M->chck_94M->chck_96M | 0.737668 | -0.248842 | 0.604738 |
| scale1p25_seed43022_dense | column_official_subtask_mean | fall | cosine_net_pct | chck_84M->chck_86M->chck_88M | 0.933815 | 0.933815 | 0.045583 |
| scale1p25_seed43022_dense | column_official_subtask_mean | fall | weighted_pearson_net_pct | chck_84M->chck_86M->chck_88M | 0.773866 | 0.773866 | 0.231925 |
| scale1p25_seed43022_dense | column_official_subtask_mean | rise+fall_pattern | cosine_net_pct | chck_84M->chck_86M->chck_88M | 0.706816 | 0.706816 | 0.228478 |
| scale1p25_seed43022_dense | column_official_subtask_mean | rise+fall_pattern | weighted_pearson_net_pct | chck_88M->chck_90M->chck_92M | 0.358092 | 0.266155 | 0.332673 |
| scale1p25_seed43022_dense | column_subtask | rise | cosine_net_pct | chck_82M->chck_84M->chck_86M | 0.276678 | -0.369552 | 0.276678 |
| scale1p25_seed43022_dense | column_subtask | rise | weighted_pearson_net_pct | chck_82M->chck_84M->chck_86M | 0.291999 | -0.376681 | 0.291999 |
| scale1p25_seed43022_dense | column_subtask | fall | cosine_net_pct | chck_82M->chck_84M->chck_86M | 0.327253 | 0.189981 | 0.327253 |
| scale1p25_seed43022_dense | column_subtask | fall | weighted_pearson_net_pct | chck_82M->chck_84M->chck_86M | 0.384164 | 0.147471 | 0.384164 |
| scale1p25_seed43022_dense | column_subtask | rise+fall_pattern | cosine_net_pct | chck_82M->chck_84M->chck_86M | 0.296070 | -0.202040 | 0.296070 |
| scale1p25_seed43022_dense | column_subtask | rise+fall_pattern | weighted_pearson_net_pct | chck_82M->chck_84M->chck_86M | 0.335592 | -0.241053 | 0.335592 |

## Files

- window_group_signature_similarity_csv: `experiments/archive/frontier_consolidation/data/transition_window_background_reference_scale125_calibration_v3/window_group_signature_similarity.csv` (133843 bytes, sha256 03a54df7b644919eddf42e590fc83bab3030f9d57948e8bbe047a67cbfa83687)
- window_item_signed_transition_similarity_csv: `experiments/archive/frontier_consolidation/data/transition_window_background_reference_scale125_calibration_v3/window_item_signed_transition_similarity.csv` (196046 bytes, sha256 6e5dfb865eaf2989b9c79aee3f8f85b6ec58d585ef62151b8590c83ff9b13773)
- window_item_pattern_similarity_csv: `experiments/archive/frontier_consolidation/data/transition_window_background_reference_scale125_calibration_v3/window_item_pattern_similarity.csv` (11155 bytes, sha256 75b9f822b15676920637f18b42f4169b6a90947e9ecc6ac0777e1a3ef3369d59)

## Reading rule

When seed43122 arrives, run this same script with seed43122 as a target together with the signed transition readout ready/195/196 tools. If its own-peak similarity is ordinary under the all-window background, or if the fixed 82→84→86 coordinate is not similar, the seed43022 chck84 late peak should be treated as a trajectory-specific endpoint asset. If the fixed and own-peak windows both stand out with matching columns/subtasks and item signed transitions, then the residual-capacity late phase is a more robust learning-dynamics object. After that readout, make the route decision rather than adding more analysis layers.

JSON: `experiments/archive/frontier_consolidation/data/transition_window_background_reference_scale125_calibration_v3/transition_window_background_summary.json`

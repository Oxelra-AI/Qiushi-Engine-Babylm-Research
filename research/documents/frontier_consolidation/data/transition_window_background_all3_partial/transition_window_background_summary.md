# transition background for seed43122 signed-transition window background

CPU/file-only comparison of every eligible late three-checkpoint window against the fixed reference 82M→84M→86M signed-transition signature. This addresses the selection-induced rise/fall hazard in own-peak windows.

Reference: `scale1p75_seed43022_reference` fixed window `chck_82M->chck_84M->chck_86M` from `experiments/archive/frontier_consolidation/data/selected_trajectory_eval_scale1p75_seed43022_reference_common2M`.

## Target summaries

| target | complete endpoints | windows | own-peak window | fixed 82→84→86 present? |
|---|---:|---:|---|---|
| scale1p75_seed43022_reference | 16 | 14 | chck_82M->chck_84M->chck_86M | True |
| scale1p25_seed43022_dense | 16 | 14 | chck_84M->chck_86M->chck_88M | True |
| scale1p75_seed43122_dense | 11 | 9 | chck_86M->chck_88M->chck_90M | True |

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
| scale1p75_seed43122_dense | own_peak | column_official_subtask_mean | rise+fall_pattern | weighted_pearson_net_pct | 0.216952 | 3 | 0.777778 | 0.065507 | 0.352335 |
| scale1p75_seed43122_dense | fixed_82_84_86 | column_official_subtask_mean | rise+fall_pattern | weighted_pearson_net_pct | 0.129241 | 4 | 0.666667 | 0.065507 | 0.352335 |
| scale1p75_seed43122_dense | own_peak | column_official_subtask_mean | rise+fall_pattern | cosine_net_pct | 0.549415 | 2 | 0.888889 | 0.115822 | 0.562952 |
| scale1p75_seed43122_dense | fixed_82_84_86 | column_official_subtask_mean | rise+fall_pattern | cosine_net_pct | 0.357309 | 3 | 0.777778 | 0.115822 | 0.562952 |
| scale1p75_seed43122_dense | own_peak | column_subtask | rise+fall_pattern | weighted_pearson_net_pct | -0.100663 | 7 | 0.333333 | -0.034998 | 0.258389 |
| scale1p75_seed43122_dense | fixed_82_84_86 | column_subtask | rise+fall_pattern | weighted_pearson_net_pct | -0.084906 | 6 | 0.444444 | -0.034998 | 0.258389 |
| scale1p75_seed43122_dense | own_peak | column_subtask | rise+fall_pattern | cosine_net_pct | -0.109132 | 7 | 0.333333 | -0.062918 | 0.249387 |
| scale1p75_seed43122_dense | fixed_82_84_86 | column_subtask | rise+fall_pattern | cosine_net_pct | -0.064549 | 6 | 0.444444 | -0.062918 | 0.249387 |
| scale1p75_seed43122_dense | own_peak | item_pattern | rise+fall_pattern | signed_transition_pattern_cosine | -0.002442 | 8 | 0.222222 | 0.000154 | 0.005417 |
| scale1p75_seed43122_dense | fixed_82_84_86 | item_pattern | rise+fall_pattern | signed_transition_pattern_cosine | 0.002236 | 3 | 0.777778 | 0.000154 | 0.005417 |
| scale1p75_seed43122_dense | own_peak | item_pattern | rise+fall_pattern | same_signed_transition_on_ref_changed_fraction | 0.017598 | 9 | 0.111111 | 0.044256 | 0.067662 |
| scale1p75_seed43122_dense | fixed_82_84_86 | item_pattern | rise+fall_pattern | same_signed_transition_on_ref_changed_fraction | 0.030549 | 7 | 0.333333 | 0.044256 | 0.067662 |

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
| scale1p75_seed43122_dense | column_official_subtask_mean | rise | cosine_net_pct | chck_74M->chck_76M->chck_78M | 0.557525 | 0.483295 | 0.343572 |
| scale1p75_seed43122_dense | column_official_subtask_mean | rise | weighted_pearson_net_pct | chck_80M->chck_82M->chck_84M | 0.314033 | -0.086429 | -0.017566 |
| scale1p75_seed43122_dense | column_official_subtask_mean | fall | cosine_net_pct | chck_82M->chck_84M->chck_86M | 0.990060 | 0.921876 | 0.990060 |
| scale1p75_seed43122_dense | column_official_subtask_mean | fall | weighted_pearson_net_pct | chck_82M->chck_84M->chck_86M | 0.943874 | 0.696881 | 0.943874 |
| scale1p75_seed43122_dense | column_official_subtask_mean | rise+fall_pattern | cosine_net_pct | chck_74M->chck_76M->chck_78M | 0.562952 | 0.549415 | 0.357309 |
| scale1p75_seed43122_dense | column_official_subtask_mean | rise+fall_pattern | weighted_pearson_net_pct | chck_74M->chck_76M->chck_78M | 0.352335 | 0.216952 | 0.129241 |
| scale1p75_seed43122_dense | column_subtask | rise | cosine_net_pct | chck_84M->chck_86M->chck_88M | 0.346682 | -0.209377 | 0.035323 |
| scale1p75_seed43122_dense | column_subtask | rise | weighted_pearson_net_pct | chck_84M->chck_86M->chck_88M | 0.361930 | -0.154133 | 0.019568 |
| scale1p75_seed43122_dense | column_subtask | fall | cosine_net_pct | chck_78M->chck_80M->chck_82M | 0.245353 | 0.074213 | -0.257336 |
| scale1p75_seed43122_dense | column_subtask | fall | weighted_pearson_net_pct | chck_78M->chck_80M->chck_82M | 0.224437 | 0.026908 | -0.287991 |
| scale1p75_seed43122_dense | column_subtask | rise+fall_pattern | cosine_net_pct | chck_84M->chck_86M->chck_88M | 0.249387 | -0.109132 | -0.064549 |
| scale1p75_seed43122_dense | column_subtask | rise+fall_pattern | weighted_pearson_net_pct | chck_84M->chck_86M->chck_88M | 0.258389 | -0.100663 | -0.084906 |

## Files

- window_group_signature_similarity_csv: `experiments/archive/frontier_consolidation/data/transition_window_background_all3_partial/window_group_signature_similarity.csv` (175848 bytes, sha256 20cafffa33f5b4538013620850d7a8f7581e230eb4fc2f84300b97112f9739cb)
- window_item_signed_transition_similarity_csv: `experiments/archive/frontier_consolidation/data/transition_window_background_all3_partial/window_item_signed_transition_similarity.csv` (259024 bytes, sha256 a94c1a675151fd83212d161a9783a6219f0a434b62a8bb8a93ecd6e4eb90e2f9)
- window_item_pattern_similarity_csv: `experiments/archive/frontier_consolidation/data/transition_window_background_all3_partial/window_item_pattern_similarity.csv` (14523 bytes, sha256 db963a312ac73220852733661e6f7490758225da7bf0ea1488c3d4dcc59750ca)

## Reading rule

When seed43122 arrives, run this same script with seed43122 as a target together with the signed transition readout ready/195/196 tools. If its own-peak similarity is ordinary under the all-window background, or if the fixed 82→84→86 coordinate is not similar, the seed43022 chck84 late peak should be treated as a trajectory-specific endpoint asset. If the fixed and own-peak windows both stand out with matching columns/subtasks and item signed transitions, then the residual-capacity late phase is a more robust learning-dynamics object. After that readout, make the route decision rather than adding more analysis layers.

JSON: `experiments/archive/frontier_consolidation/data/transition_window_background_all3_partial/transition_window_background_summary.json`

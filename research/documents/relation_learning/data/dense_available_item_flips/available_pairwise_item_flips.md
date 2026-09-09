# earlier analysis available-column paired item flips

Only the intersection of available discrete columns is compared; missing SuperGLUE/AoA or missing per-target tasks are not inferred.

## Scores
| label | BLiMP | Supplement | EWoK | Entity | COMPS | GlobalPIQA | Reading |
|---|---:|---:|---:|---:|---:|---:|---:|
| chck82 | 68.49128403651986 | 62.9378112562002 | 50.05545332553276 | 28.314041930298774 | 52.19117509443596 | 37.57766990291262 | 8.148713589261902 |
| coherent86_s43022 | 68.51 | 63.64 | 50.02 | 28.32 | 52.05 | 38.565 | 8.165 |
| dense62064_zero_reading | 68.06 | 63.04 | 49.92 | 29.37 | 52.15 | 40.05 | 8.219999999999999 |
| dense62065_zero_reading | 68.02 | 63.09 | 49.77 | 29.42 | 52.15 | 40.05 | 8.21 |

## chck82_minus_coherent86_s43022
Available discrete columns: `['BLiMP', 'Supplement', 'EWoK', 'Entity', 'COMPS', 'GlobalPIQA']`; missing: `[]`.
Aggregate over available discrete items: gains `2418`, losses `2332`, net `86` over `170722` items; available-column payload mean Δ `-0.25626074234997215`.

| column | payload Δ | reconstructed Δ | common | gains | losses | net | best groups | worst groups |
|---|---:|---:|---:|---:|---:|---:|---|---|
| BLiMP | -0.019 | -0.025 | 59875 | 545 | 565 | -20 | npi_present_2 +42/914, npi_present_1 +20/909, ellipsis_n_bar_1 +10/802, animate_subject_trans +11/923 | wh_island -23/960, principle_A_domain_1 -20/914, superlative_quantifiers_2 -20/986, principle_A_c_command -12/946 |
| Supplement | -0.702 | -0.698 | 5218 | 37 | 27 | +10 | subject_aux_inversion +14/3867, turn_taking +0/280, hypernym -1/842, qa_congruence_tricky -1/165 | qa_congruence_easy -2/64, qa_congruence_tricky -1/165, hypernym -1/842, turn_taking +0/280 |
| EWoK | +0.035 | +0.036 | 7618 | 113 | 117 | -4 | social-properties +4/328, physical-dynamics +1/120, spatial-relations +2/490, quantitative-properties +1/314 | material-properties -2/170, social-interactions -2/294, social-relations -9/1548, physical-relations -1/818 |
| Entity | -0.006 | -0.008 | 6780 | 54 | 50 | +4 | move_contents_3_ops +5/406, regular_2_ops +4/405, ambiref_0_ops +5/508, ambiref_5_ops +1/123 | regular_5_ops -1/94, ambiref_2_ops -4/413, move_contents_5_ops -1/116, move_contents_4_ops -3/353 |
| COMPS | +0.141 | +0.144 | 91028 | 1668 | 1570 | +98 | wugs_dist_before +74/13896, wugs +22/13896, base +25/49340, wugs_dist_in_between -23/13896 | wugs_dist_in_between -23/13896, base +25/49340, wugs +22/13896, wugs_dist_before +74/13896 |
| GlobalPIQA | -0.987 | -0.985 | 203 | 1 | 3 | -2 | GlobalPIQA_parallel -1/103, GlobalPIQA_nonparallel -1/100 | GlobalPIQA_nonparallel -1/100, GlobalPIQA_parallel -1/103 |

## dense62064_zero_reading_minus_coherent86_s43022
Available discrete columns: `['BLiMP', 'Supplement', 'EWoK', 'Entity', 'COMPS', 'GlobalPIQA']`; missing: `[]`.
Aggregate over available discrete items: gains `4819`, losses `4917`, net `-98` over `170722` items; available-column payload mean Δ `0.2474999999999993`.

| column | payload Δ | reconstructed Δ | common | gains | losses | net | best groups | worst groups |
|---|---:|---:|---:|---:|---:|---:|---|---|
| BLiMP | -0.450 | -0.450 | 59875 | 776 | 1048 | -272 | matrix_question_npi_licensor_present +36/929, existential_there_quantifiers_2 +28/911, wh_questions_object_gap +15/859, ellipsis_n_bar_2 +13/828 | principle_A_domain_1 -32/914, distractor_agreement_relational_noun -24/788, superlative_quantifiers_2 -28/986, only_npi_licensor_present -25/882 |
| Supplement | -0.600 | -0.598 | 5218 | 54 | 59 | -5 | subject_aux_inversion +5/3867, turn_taking +0/280, qa_congruence_tricky -1/165, hypernym -8/842 | qa_congruence_easy -1/64, hypernym -8/842, qa_congruence_tricky -1/165, turn_taking +0/280 |
| EWoK | -0.100 | -0.097 | 7618 | 236 | 223 | +13 | material-dynamics +32/770, social-interactions +2/294, social-relations +5/1548, spatial-relations +1/490 | physical-dynamics -2/120, physical-interactions -9/556, social-properties -5/328, material-properties -1/170 |
| Entity | +1.050 | +1.045 | 6780 | 219 | 177 | +42 | ambiref_5_ops +6/123, move_contents_2_ops +15/399, move_contents_4_ops +13/353, ambiref_4_ops +13/434 | regular_0_ops -25/517, move_contents_0_ops -16/516, ambiref_0_ops -7/508, ambiref_1_ops -4/428 |
| COMPS | +0.100 | +0.107 | 91028 | 3530 | 3409 | +121 | wugs_dist_before +147/13896, base +86/49340, wugs -46/13896, wugs_dist_in_between -66/13896 | wugs_dist_in_between -66/13896, wugs -46/13896, base +86/49340, wugs_dist_before +147/13896 |
| GlobalPIQA | +1.485 | +1.485 | 203 | 4 | 1 | +3 | GlobalPIQA_nonparallel +2/100, GlobalPIQA_parallel +1/103 | GlobalPIQA_parallel +1/103, GlobalPIQA_nonparallel +2/100 |

## dense62065_zero_reading_minus_coherent86_s43022
Available discrete columns: `['BLiMP', 'Supplement', 'EWoK', 'Entity', 'COMPS', 'GlobalPIQA']`; missing: `[]`.
Aggregate over available discrete items: gains `4886`, losses `5018`, net `-132` over `170722` items; available-column payload mean Δ `None`.

| column | payload Δ | reconstructed Δ | common | gains | losses | net | best groups | worst groups |
|---|---:|---:|---:|---:|---:|---:|---|---|
| BLiMP | NA | -0.488 | 59875 | 785 | 1080 | -295 | matrix_question_npi_licensor_present +37/929, existential_there_quantifiers_2 +25/911, sentential_subject_island +17/961, ellipsis_n_bar_2 +13/828 | principle_A_domain_1 -30/914, distractor_agreement_relational_noun -25/788, only_npi_licensor_present -26/882, superlative_quantifiers_2 -29/986 |
| Supplement | NA | -0.542 | 5218 | 52 | 59 | -7 | turn_taking +1/280, subject_aux_inversion +2/3867, qa_congruence_tricky -1/165, hypernym -8/842 | qa_congruence_easy -1/64, hypernym -8/842, qa_congruence_tricky -1/165, subject_aux_inversion +2/3867 |
| EWoK | NA | -0.250 | 7618 | 235 | 231 | +4 | material-dynamics +34/770, spatial-relations +1/490, social-relations -2/1548, agent-properties -5/2210 | social-properties -6/328, physical-dynamics -2/120, physical-interactions -8/556, social-interactions -2/294 |
| Entity | NA | +1.093 | 6780 | 222 | 179 | +43 | ambiref_5_ops +7/123, move_contents_4_ops +15/353, move_contents_2_ops +15/399, ambiref_4_ops +13/434 | regular_0_ops -25/517, move_contents_0_ops -16/516, ambiref_0_ops -7/508, ambiref_1_ops -5/428 |
| COMPS | NA | +0.098 | 91028 | 3588 | 3468 | +120 | wugs_dist_before +151/13896, base +91/49340, wugs -48/13896, wugs_dist_in_between -74/13896 | wugs_dist_in_between -74/13896, wugs -48/13896, base +91/49340, wugs_dist_before +151/13896 |
| GlobalPIQA | NA | +1.485 | 203 | 4 | 1 | +3 | GlobalPIQA_nonparallel +2/100, GlobalPIQA_parallel +1/103 | GlobalPIQA_parallel +1/103, GlobalPIQA_nonparallel +2/100 |

JSON: `experiments/archive/relation_learning/data/dense_available_item_flips/available_pairwise_item_flips.json`

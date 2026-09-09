# fastpath closure and ordinary84 endpoint ordinary84 candidate synthesis

No-training analysis of saved official-compatible payloads. ordinary84 is the same seed43022 scale1.75 trajectory, not a new training route.

## Cheap7 scores

| arm | BLiMP | Supplement | EWoK | Entity | COMPS | GlobalPIQA | Reading | cheap7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| anchor80 | 68.120 | 62.620 | 49.240 | 28.200 | 52.110 | 38.105 | 8.300 | 43.813571 |
| chck82 | 68.491 | 62.938 | 50.055 | 28.314 | 52.191 | 37.578 | 8.149 | 43.959450 |
| ordinary84 | 68.240 | 63.480 | 50.070 | 28.580 | 52.210 | 38.120 | 8.155 | 44.122143 |
| ordinary86 | 68.470 | 62.680 | 50.140 | 28.580 | 52.290 | 36.135 | 8.100 | 43.770714 |
| coherent86 | 68.520 | 63.650 | 49.910 | 28.440 | 51.990 | 38.065 | 8.170 | 44.106429 |

## SuperGLUE projection thresholds for ordinary84

- ordinary84_sg_needed_to_tie_chck82_with_aoa0: `68.62733050647392`
- ordinary84_sg_needed_to_tie_coherent86_min_projected_with_aoa0: `69.67975515726181`
- ordinary84_sg_needed_to_tie_coherent86_max_projected_with_aoa0: `69.73279617564941`
- ordinary84_projected_overall_if_sg_equals_chck82_and_aoa0: `42.069020152367976`

## Item-transition comparisons

### ordinary84_vs_chck82
- total gain/loss/net: 4443/4501/-58 over 170722 common discrete items
- discrete payload mean delta: +0.188761; reconstructed mean delta: +0.190491
| column | payload Δ | recon Δ | gains | losses | net | best groups | worst groups |
|---|---:|---:|---:|---:|---:|---|---|
| BLiMP | -0.251 | -0.240 | 1015 | 1154 | -139 | existential_there_quantifiers_2 +69/911, left_branch_island_echo_question +36/947, wh_vs_that_with_gap_long_distance +18/910 | npi_present_2 -52/914, only_npi_licensor_present -45/882, npi_present_1 -38/909 |
| Supplement | +0.542 | +0.546 | 51 | 61 | -10 | qa_congruence_easy +1/64, qa_congruence_tricky +2/165, turn_taking +2/280 | hypernym -4/842, subject_aux_inversion -11/3867, turn_taking +2/280 |
| EWoK | +0.015 | +0.018 | 237 | 240 | -3 | material-dynamics +13/770, social-interactions +2/294, quantitative-properties +2/314 | physical-dynamics -2/120, social-properties -5/328, agent-properties -23/2210 |
| Entity | +0.266 | +0.261 | 134 | 123 | +11 | regular_5_ops +2/94, move_contents_3_ops +6/406, move_contents_2_ops +5/399 | regular_3_ops -5/425, ambiref_3_ops -4/409, ambiref_1_ops -4/428 |
| COMPS | +0.019 | +0.014 | 3001 | 2919 | +82 | wugs_dist_in_between +36/13896, base +103/49340, wugs -19/13896 | wugs_dist_before -38/13896, wugs -19/13896, base +103/49340 |
| GlobalPIQA | +0.542 | +0.544 | 5 | 4 | +1 | GlobalPIQA_nonparallel +4/100, GlobalPIQA_parallel -3/103 | GlobalPIQA_parallel -3/103, GlobalPIQA_nonparallel +4/100 |

### ordinary84_vs_coherent86
- total gain/loss/net: 4535/4478/+57 over 170722 common discrete items
- discrete payload mean delta: +0.020833; reconstructed mean delta: +0.022053
| column | payload Δ | recon Δ | gains | losses | net | best groups | worst groups |
|---|---:|---:|---:|---:|---:|---|---|
| BLiMP | -0.280 | -0.278 | 973 | 1142 | -169 | existential_there_quantifiers_2 +50/911, left_branch_island_echo_question +28/947, animate_subject_trans +18/923 | only_npi_licensor_present -49/882, principle_A_c_command -34/946, only_npi_scope -22/837 |
| Supplement | -0.170 | -0.162 | 73 | 69 | +4 | turn_taking +3/280, subject_aux_inversion +6/3867, qa_congruence_tricky +0/165 | qa_congruence_easy -1/64, hypernym -4/842, qa_congruence_tricky +0/165 |
| EWoK | +0.160 | +0.167 | 242 | 243 | -1 | quantitative-properties +5/314, material-dynamics +12/770, spatial-relations +4/490 | physical-interactions -4/556, agent-properties -15/2210, material-properties -1/170 |
| Entity | +0.140 | +0.135 | 145 | 141 | +4 | regular_5_ops +3/94, move_contents_3_ops +11/406, regular_2_ops +5/405 | move_contents_5_ops -3/116, regular_3_ops -8/425, ambiref_1_ops -4/428 |
| COMPS | +0.220 | +0.212 | 3096 | 2877 | +219 | wugs_dist_before +47/13896, base +141/49340, wugs_dist_in_between +24/13896 | wugs +7/13896, wugs_dist_in_between +24/13896, base +141/49340 |
| GlobalPIQA | +0.055 | +0.058 | 6 | 6 | +0 | GlobalPIQA_nonparallel +4/100, GlobalPIQA_parallel -4/103 | GlobalPIQA_parallel -4/103, GlobalPIQA_nonparallel +4/100 |

### ordinary84_vs_ordinary86
- total gain/loss/net: 4049/4225/-176 over 170722 common discrete items
- discrete payload mean delta: +0.400833; reconstructed mean delta: +0.401655
| column | payload Δ | recon Δ | gains | losses | net | best groups | worst groups |
|---|---:|---:|---:|---:|---:|---|---|
| BLiMP | -0.230 | -0.222 | 859 | 986 | -127 | existential_there_quantifiers_2 +48/911, left_branch_island_simple_question +29/951, wh_island +21/960 | only_npi_licensor_present -28/882, principle_A_domain_1 -27/914, tough_vs_raising_2 -19/920 |
| Supplement | +0.800 | +0.808 | 47 | 64 | -17 | qa_congruence_tricky +4/165, qa_congruence_easy +1/64, turn_taking +3/280 | subject_aux_inversion -21/3867, hypernym -4/842, turn_taking +3/280 |
| EWoK | -0.070 | -0.067 | 206 | 234 | -28 | social-interactions +4/294, quantitative-properties +2/314, material-properties +1/170 | social-relations -20/1548, physical-dynamics -1/120, spatial-relations -4/490 |
| Entity | +0.000 | -0.007 | 109 | 119 | -10 | move_contents_3_ops +12/406, regular_5_ops +2/94, move_contents_5_ops +2/116 | regular_4_ops -9/388, move_contents_4_ops -4/353, ambiref_4_ops -4/434 |
| COMPS | -0.080 | -0.087 | 2822 | 2820 | +2 | wugs_dist_before +64/13896, base +70/49340, wugs -37/13896 | wugs_dist_in_between -95/13896, wugs -37/13896, base +70/49340 |
| GlobalPIQA | +1.985 | +1.985 | 6 | 2 | +4 | GlobalPIQA_nonparallel +3/100, GlobalPIQA_parallel +1/103 | GlobalPIQA_parallel +1/103, GlobalPIQA_nonparallel +3/100 |

### ordinary84_vs_anchor80
- total gain/loss/net: 6476/6170/+306 over 170722 common discrete items
- discrete payload mean delta: +0.384167; reconstructed mean delta: +0.384698
| column | payload Δ | recon Δ | gains | losses | net | best groups | worst groups |
|---|---:|---:|---:|---:|---:|---|---|
| BLiMP | +0.120 | +0.134 | 1453 | 1366 | +87 | npi_present_1 +60/909, matrix_question_npi_licensor_present +47/929, left_branch_island_echo_question +45/947 | principle_A_domain_1 -54/914, wh_island -47/960, tough_vs_raising_1 -39/948 |
| Supplement | +0.860 | +0.859 | 103 | 93 | +10 | qa_congruence_tricky +6/165, hypernym +6/842, qa_congruence_easy +0/64 | subject_aux_inversion -2/3867, qa_congruence_easy +0/64, turn_taking +0/280 |
| EWoK | +0.830 | +0.830 | 336 | 289 | +47 | material-properties +6/170, physical-relations +23/818, material-dynamics +17/770 | social-properties -6/328, quantitative-properties -1/314, social-relations -3/1548 |
| Entity | +0.380 | +0.377 | 178 | 156 | +22 | regular_5_ops +4/94, move_contents_2_ops +5/399, move_contents_3_ops +5/406 | ambiref_5_ops -4/123, regular_3_ops -5/425, move_contents_1_ops -4/437 |
| COMPS | +0.100 | +0.093 | 4400 | 4260 | +140 | wugs_dist_in_between +379/13896, base +123/49340, wugs -8/13896 | wugs_dist_before -354/13896, wugs -8/13896, base +123/49340 |
| GlobalPIQA | +0.015 | +0.015 | 6 | 6 | +0 | GlobalPIQA_nonparallel +1/100, GlobalPIQA_parallel -1/103 | GlobalPIQA_parallel -1/103, GlobalPIQA_nonparallel +1/100 |

## Scientific reading
- ordinary84 is not a new mechanism; it is an existing same-seed scale1.75 trajectory checkpoint found while constructing the fast-path control.
- Its cheap7 exceeds protected chck82 and coherent86, so SuperGLUE-only is the lowest-cost decision-changing endpoint check before any AoA/materialization.
- The item-transition comparisons should be read as trajectory endpoint selection and redistribution, not a data-efficient learning principle or binding-deficit repair.

JSON: `experiments/archive/representation_and_objectives/data/ordinary84_candidate_synthesis/ordinary84_candidate_synthesis.json`

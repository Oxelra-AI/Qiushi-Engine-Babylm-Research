# earlier analysis official-coordinate transition comparison: dense_seed62064_zero_reading_current vs coherent86_official_zero_reading

## Score summary

Scores labelled `computed` are recomputed from prediction files using the official macro over subtasks or SuperGLUE primary metrics where applicable. `payload` is the saved evaluator score. Item flips below are micro counts and should not be read as official score movement when subtasks have unequal sizes.

| column | A payload | B payload | payload B-A | A computed | B computed | computed B-A | A payload-computed | B payload-computed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BLiMP | 68.510000 | 68.060000 | -0.450000 | 68.515970 | 68.065520 | -0.450450 | -0.005970 | -0.005520 |
| Supplement | 63.640000 | 63.040000 | -0.600000 | 63.635369 | 63.037493 | -0.597876 | +0.004631 | +0.002507 |
| EWoK | 50.020000 | 49.920000 | -0.100000 | 50.019616 | 49.922478 | -0.097138 | +0.000384 | -0.002478 |
| Entity | 28.320000 | 29.370000 | +1.050000 | 28.322219 | 29.366952 | +1.044733 | -0.002219 | +0.003048 |
| COMPS | 52.050000 | 52.150000 | +0.100000 | 52.047175 | 52.153718 | +0.106543 | +0.002825 | -0.003718 |
| GlobalPIQA_parallel | 29.130000 | 30.100000 | +0.970000 | 29.126214 | 30.097087 | +0.970874 | +0.003786 | +0.002913 |
| GlobalPIQA_nonparallel | 48.000000 | 50.000000 | +2.000000 | 48.000000 | 50.000000 | +2.000000 | +0.000000 | +0.000000 |
| GlobalPIQA | 38.565000 | 40.050000 | +1.485000 | 38.563107 | 40.048544 | +1.485437 | +0.001893 | +0.001456 |
| Reading | 8.165000 | 8.220000 | +0.055000 | 8.165000 | 8.220000 | +0.055000 | +0.000000 | +0.000000 |
| SuperGLUE |  | 68.562691 |  |  |  |  |  |  |
| cheap7_mean | 44.181429 | 44.401429 | +0.220000 | 44.181208 | 44.402101 | +0.220893 | +0.000221 | -0.000672 |
| projected_overall_with_aoa0_if_missing |  | 42.152521 |  |  |  |  |  |  |

## Prediction-level transitions

| column | n | official-macro A | official-macro B | official-macro B-A | micro A | micro B | micro B-A | B-only | A-only | net item |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BLiMP | 59875 | 68.515970 | 68.065520 | -0.450450 | 68.404175 | 67.949896 | -0.454280 | 776 | 1048 | -272 |
| Supplement | 5218 | 63.635369 | 63.037493 | -0.597876 | 73.974703 | 73.878881 | -0.095822 | 54 | 59 | -5 |
| EWoK | 7618 | 50.019616 | 49.922478 | -0.097138 | 49.986873 | 50.157522 | +0.170648 | 236 | 223 | +13 |
| Entity | 6780 | 28.322219 | 29.366952 | +1.044733 | 27.831858 | 28.451327 | +0.619469 | 219 | 177 | +42 |
| COMPS | 91028 | 52.047175 | 52.153718 | +0.106543 | 52.513512 | 52.646438 | +0.132926 | 3530 | 3409 | +121 |
| GlobalPIQA_parallel | 103 | 29.126214 | 30.097087 | +0.970874 | 29.126214 | 30.097087 | +0.970874 | 2 | 1 | +1 |
| GlobalPIQA_nonparallel | 100 | 48.000000 | 50.000000 | +2.000000 | 48.000000 | 50.000000 | +2.000000 | 2 | 0 | +2 |

## Largest subtask movements by official-macro component


### BLiMP

| subtask | n | A acc | B acc | B-A acc | net items |
|---|---:|---:|---:|---:|---:|
| matrix_question_npi_licensor_present | 929 | 41.657696 | 45.532831 | +3.875135 | +36 |
| principle_A_domain_1 | 914 | 77.024070 | 73.522976 | -3.501094 | -32 |
| existential_there_quantifiers_2 | 911 | 40.944018 | 44.017563 | +3.073546 | +28 |
| distractor_agreement_relational_noun | 788 | 49.873096 | 46.827411 | -3.045685 | -24 |
| superlative_quantifiers_2 | 986 | 82.251521 | 79.411765 | -2.839757 | -28 |
| only_npi_licensor_present | 882 | 80.385488 | 77.551020 | -2.834467 | -25 |
| determiner_noun_agreement_1 | 929 | 88.912809 | 86.652314 | -2.260495 | -21 |
| wh_vs_that_no_gap | 861 | 88.734030 | 86.643438 | -2.090592 | -18 |
| left_branch_island_echo_question | 947 | 38.542767 | 36.642027 | -1.900739 | -18 |
| left_branch_island_simple_question | 951 | 51.209253 | 49.316509 | -1.892744 | -18 |
| sentential_negation_npi_licensor_present | 919 | 99.455930 | 97.606094 | -1.849837 | -17 |
| wh_questions_object_gap | 859 | 54.598370 | 56.344587 | +1.746217 | +15 |
| npi_present_1 | 909 | 31.463146 | 29.812981 | -1.650165 | -15 |
| coordinate_structure_constraint_object_extraction | 949 | 78.714436 | 77.133825 | -1.580611 | -15 |
| ellipsis_n_bar_2 | 828 | 96.376812 | 97.946860 | +1.570048 | +13 |

### Supplement

| subtask | n | A acc | B acc | B-A acc | net items |
|---|---:|---:|---:|---:|---:|
| qa_congruence_easy | 64 | 70.312500 | 68.750000 | -1.562500 | -1 |
| hypernym | 842 | 49.881235 | 48.931116 | -0.950119 | -8 |
| qa_congruence_tricky | 165 | 49.696970 | 49.090909 | -0.606061 | -1 |
| subject_aux_inversion | 3867 | 80.786139 | 80.915438 | +0.129299 | +5 |
| turn_taking | 280 | 67.500000 | 67.500000 | +0.000000 | +0 |

### EWoK

| subtask | n | A acc | B acc | B-A acc | net items |
|---|---:|---:|---:|---:|---:|
| material-dynamics | 770 | 50.519481 | 54.675325 | +4.155844 | +32 |
| physical-dynamics | 120 | 52.500000 | 50.833333 | -1.666667 | -2 |
| physical-interactions | 556 | 48.561151 | 46.942446 | -1.618705 | -9 |
| social-properties | 328 | 46.646341 | 45.121951 | -1.524390 | -5 |
| social-interactions | 294 | 54.081633 | 54.761905 | +0.680272 | +2 |
| material-properties | 170 | 51.764706 | 51.176471 | -0.588235 | -1 |
| physical-relations | 818 | 49.877751 | 49.388753 | -0.488998 | -4 |
| social-relations | 1548 | 50.645995 | 50.968992 | +0.322997 | +5 |
| quantitative-properties | 314 | 48.726115 | 48.407643 | -0.318471 | -1 |
| agent-properties | 2210 | 50.361991 | 50.135747 | -0.226244 | -5 |
| spatial-relations | 490 | 46.530612 | 46.734694 | +0.204082 | +1 |

### Entity

| subtask | n | A acc | B acc | B-A acc | net items |
|---|---:|---:|---:|---:|---:|
| ambiref_5_ops | 123 | 31.707317 | 36.585366 | +4.878049 | +6 |
| regular_0_ops | 517 | 40.618956 | 35.783366 | -4.835590 | -25 |
| move_contents_2_ops | 399 | 21.553885 | 25.313283 | +3.759398 | +15 |
| move_contents_4_ops | 353 | 31.161473 | 34.844193 | +3.682720 | +13 |
| move_contents_0_ops | 516 | 45.542636 | 42.441860 | -3.100775 | -16 |
| ambiref_4_ops | 434 | 34.562212 | 37.557604 | +2.995392 | +13 |
| regular_4_ops | 388 | 27.577320 | 30.412371 | +2.835052 | +11 |
| regular_2_ops | 405 | 21.975309 | 24.444444 | +2.469136 | +10 |
| ambiref_3_ops | 409 | 26.650367 | 28.850856 | +2.200489 | +9 |
| move_contents_3_ops | 406 | 26.600985 | 28.571429 | +1.970443 | +8 |
| move_contents_5_ops | 116 | 43.103448 | 44.827586 | +1.724138 | +2 |
| regular_1_ops | 409 | 15.647922 | 17.359413 | +1.711491 | +7 |
| ambiref_0_ops | 508 | 25.196850 | 23.818898 | -1.377953 | -7 |
| regular_5_ops | 94 | 29.787234 | 30.851064 | +1.063830 | +1 |
| ambiref_1_ops | 428 | 14.953271 | 14.018692 | -0.934579 | -4 |

### COMPS

| subtask | n | A acc | B acc | B-A acc | net items |
|---|---:|---:|---:|---:|---:|
| wugs_dist_before | 13896 | 37.478411 | 38.536269 | +1.057858 | +147 |
| wugs_dist_in_between | 13896 | 65.443293 | 64.968336 | -0.474957 | -66 |
| wugs | 13896 | 52.022165 | 51.691134 | -0.331031 | -46 |
| base | 49340 | 53.244832 | 53.419133 | +0.174301 | +86 |

### GlobalPIQA_parallel

| subtask | n | A acc | B acc | B-A acc | net items |
|---|---:|---:|---:|---:|---:|
| global_piqa_parallel | 103 | 29.126214 | 30.097087 | +0.970874 | +1 |

### GlobalPIQA_nonparallel

| subtask | n | A acc | B acc | B-A acc | net items |
|---|---:|---:|---:|---:|---:|
| global_piqa_nonparallel | 100 | 48.000000 | 50.000000 | +2.000000 | +2 |

## Interpretation note

Official BabyLM columns use macro averages over subtasks and SuperGLUE primary metrics; micro item flips diagnose which decisions changed but do not directly equal leaderboard movement. Treat deltas as measurement evidence, not as a causal mechanism without training provenance and controls.

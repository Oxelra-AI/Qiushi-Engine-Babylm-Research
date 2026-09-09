# earlier analysis official-coordinate transition comparison: pa87_alpha050_official vs coherent86_alpha075_existing_official

## Score summary

Scores labelled `computed` are recomputed from prediction files using the official macro over subtasks or SuperGLUE primary metrics where applicable. `payload` is the saved evaluator score. Item flips below are micro counts and should not be read as official score movement when subtasks have unequal sizes.

| column | A payload | B payload | payload B-A | A computed | B computed | computed B-A | A payload-computed | B payload-computed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BLiMP | 68.510000 | 68.420000 | -0.090000 | 68.515970 | 68.419053 | -0.096917 | -0.005970 | +0.000947 |
| Supplement | 63.640000 | 63.780000 | +0.140000 | 63.635369 | 63.780333 | +0.144964 | +0.004631 | -0.000333 |
| EWoK | 50.020000 | 50.260000 | +0.240000 | 50.019616 | 50.262103 | +0.242487 | +0.000384 | -0.002103 |
| Entity | 28.320000 | 28.250000 | -0.070000 | 28.322219 | 28.248605 | -0.073614 | -0.002219 | +0.001395 |
| COMPS | 52.050000 | 52.150000 | +0.100000 | 52.047175 | 52.151595 | +0.104420 | +0.002825 | -0.001595 |
| GlobalPIQA_parallel | 29.130000 | 29.130000 | +0.000000 | 29.126214 | 29.126214 | +0.000000 | +0.003786 | +0.003786 |
| GlobalPIQA_nonparallel | 48.000000 | 47.000000 | -1.000000 | 48.000000 | 47.000000 | -1.000000 | +0.000000 | +0.000000 |
| GlobalPIQA | 38.565000 | 38.065000 | -0.500000 | 38.563107 | 38.063107 | -0.500000 | +0.001893 | +0.001893 |
| Reading | 8.165000 | 8.215000 | +0.050000 | 8.165000 | 8.215000 | +0.050000 | +0.000000 | +0.000000 |
| SuperGLUE | 69.819222 | 69.777968 | -0.041254 | 69.819222 | 69.777968 | -0.041254 | +0.000000 | +0.000000 |
| AoA |  | 0.000000 |  |  | 0.000000 |  |  | +0.000000 |
| cheap7_mean | 44.181429 | 44.162857 | -0.018571 | 44.181208 | 44.162828 | -0.018380 | +0.000221 | +0.000029 |
| projected_overall_with_aoa0_if_missing | 42.121025 | 42.101996 | -0.019028 | 42.120853 | 42.101974 | -0.018879 | +0.000172 | +0.000023 |
| SuperGLUE/boolq |  |  |  | 68.562691 | 68.562691 | +0.000000 |  |  |
| SuperGLUE/mnli |  |  |  | 59.800326 | 59.800326 | +0.000000 |  |  |
| SuperGLUE/mrpc |  |  |  | 88.737201 | 88.737201 | +0.000000 |  |  |
| SuperGLUE/multirc |  |  |  | 67.574257 | 67.285479 | -0.288779 |  |  |
| SuperGLUE/qqp |  |  |  | 71.519959 | 71.519959 | +0.000000 |  |  |
| SuperGLUE/rte |  |  |  | 63.309353 | 63.309353 | +0.000000 |  |  |
| SuperGLUE/wsc |  |  |  | 69.230769 | 69.230769 | +0.000000 |  |  |

## Prediction-level transitions

| column | n | official-macro A | official-macro B | official-macro B-A | micro A | micro B | micro B-A | B-only | A-only | net item |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BLiMP | 59875 | 68.515970 | 68.419053 | -0.096917 | 68.404175 | 68.302296 | -0.101879 | 314 | 375 | -61 |
| Supplement | 5218 | 63.635369 | 63.780333 | +0.144964 | 73.974703 | 74.089690 | +0.114987 | 23 | 17 | +6 |
| EWoK | 7618 | 50.019616 | 50.262103 | +0.242487 | 49.986873 | 50.210029 | +0.223156 | 100 | 83 | +17 |
| Entity | 6780 | 28.322219 | 28.248605 | -0.073614 | 27.831858 | 27.935103 | +0.103245 | 58 | 51 | +7 |
| COMPS | 91028 | 52.047175 | 52.151595 | +0.104420 | 52.513512 | 52.597003 | +0.083491 | 1188 | 1112 | +76 |
| GlobalPIQA_parallel | 103 | 29.126214 | 29.126214 | +0.000000 | 29.126214 | 29.126214 | +0.000000 | 1 | 1 | +0 |
| GlobalPIQA_nonparallel | 100 | 48.000000 | 47.000000 | -1.000000 | 48.000000 | 47.000000 | -1.000000 | 1 | 2 | -1 |

## SuperGLUE task transitions

| task | metric | n | A primary | B primary | B-A primary | A acc | B acc | B-A acc | B-only | A-only | net item |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| boolq | accuracy | 1635 | 68.562691 | 68.562691 | +0.000000 | 68.562691 | 68.562691 | +0.000000 | 0 | 0 | +0 |
| mnli | accuracy | 4908 | 59.800326 | 59.800326 | +0.000000 | 59.800326 | 59.800326 | +0.000000 | 0 | 0 | +0 |
| mrpc | f1 | 204 | 88.737201 | 88.737201 | +0.000000 | 83.823529 | 83.823529 | +0.000000 | 0 | 0 | +0 |
| multirc | accuracy | 2424 | 67.574257 | 67.285479 | -0.288779 | 67.574257 | 67.285479 | -0.288779 | 40 | 47 | -7 |
| qqp | f1 | 20215 | 71.519959 | 71.519959 | +0.000000 | 77.976750 | 77.976750 | +0.000000 | 0 | 0 | +0 |
| rte | accuracy | 139 | 63.309353 | 63.309353 | +0.000000 | 63.309353 | 63.309353 | +0.000000 | 0 | 0 | +0 |
| wsc | accuracy | 52 | 69.230769 | 69.230769 | +0.000000 | 69.230769 | 69.230769 | +0.000000 | 0 | 0 | +0 |

## Largest subtask movements by official-macro component


### BLiMP

| subtask | n | A acc | B acc | B-A acc | net items |
|---|---:|---:|---:|---:|---:|
| wh_island | 960 | 57.604167 | 55.833333 | -1.770833 | -17 |
| matrix_question_npi_licensor_present | 929 | 41.657696 | 40.581270 | -1.076426 | -10 |
| sentential_subject_island | 961 | 32.778356 | 31.737773 | -1.040583 | -10 |
| npi_present_2 | 914 | 34.901532 | 35.776805 | +0.875274 | +8 |
| principle_A_domain_1 | 914 | 77.024070 | 76.148796 | -0.875274 | -8 |
| wh_vs_that_with_gap | 919 | 43.960827 | 44.722524 | +0.761697 | +7 |
| principle_A_reconstruction | 967 | 41.882110 | 41.158221 | -0.723888 | -7 |
| existential_there_object_raising | 812 | 73.645320 | 73.029557 | -0.615764 | -5 |
| superlative_quantifiers_2 | 986 | 82.251521 | 81.643002 | -0.608519 | -6 |
| wh_questions_subject_gap_long_distance | 857 | 92.182030 | 92.765461 | +0.583431 | +5 |
| wh_vs_that_no_gap_long_distance | 875 | 95.200000 | 95.771429 | +0.571429 | +5 |
| only_npi_licensor_present | 882 | 80.385488 | 79.818594 | -0.566893 | -5 |
| passive_2 | 903 | 73.754153 | 73.200443 | -0.553710 | -5 |
| principle_A_domain_2 | 915 | 53.442623 | 52.896175 | -0.546448 | -5 |
| adjunct_island | 928 | 76.185345 | 75.646552 | -0.538793 | -5 |

### Supplement

| subtask | n | A acc | B acc | B-A acc | net items |
|---|---:|---:|---:|---:|---:|
| turn_taking | 280 | 67.500000 | 68.214286 | +0.714286 | +2 |
| subject_aux_inversion | 3867 | 80.786139 | 80.915438 | +0.129299 | +5 |
| hypernym | 842 | 49.881235 | 49.762470 | -0.118765 | -1 |
| qa_congruence_tricky | 165 | 49.696970 | 49.696970 | +0.000000 | +0 |
| qa_congruence_easy | 64 | 70.312500 | 70.312500 | +0.000000 | +0 |

### EWoK

| subtask | n | A acc | B acc | B-A acc | net items |
|---|---:|---:|---:|---:|---:|
| material-dynamics | 770 | 50.519481 | 52.727273 | +2.207792 | +17 |
| spatial-relations | 490 | 46.530612 | 47.959184 | +1.428571 | +7 |
| quantitative-properties | 314 | 48.726115 | 49.681529 | +0.955414 | +3 |
| physical-relations | 818 | 49.877751 | 49.022005 | -0.855746 | -7 |
| material-properties | 170 | 51.764706 | 51.176471 | -0.588235 | -1 |
| social-relations | 1548 | 50.645995 | 51.227390 | +0.581395 | +9 |
| agent-properties | 2210 | 50.361991 | 50.000000 | -0.361991 | -8 |
| physical-interactions | 556 | 48.561151 | 48.201439 | -0.359712 | -2 |
| social-interactions | 294 | 54.081633 | 53.741497 | -0.340136 | -1 |
| social-properties | 328 | 46.646341 | 46.646341 | +0.000000 | +0 |
| physical-dynamics | 120 | 52.500000 | 52.500000 | +0.000000 | +0 |

### Entity

| subtask | n | A acc | B acc | B-A acc | net items |
|---|---:|---:|---:|---:|---:|
| move_contents_5_ops | 116 | 43.103448 | 41.379310 | -1.724138 | -2 |
| move_contents_0_ops | 516 | 45.542636 | 47.093023 | +1.550388 | +8 |
| ambiref_0_ops | 508 | 25.196850 | 26.574803 | +1.377953 | +7 |
| regular_5_ops | 94 | 29.787234 | 28.723404 | -1.063830 | -1 |
| regular_3_ops | 425 | 31.058824 | 30.117647 | -0.941176 | -4 |
| move_contents_4_ops | 353 | 31.161473 | 30.311615 | -0.849858 | -3 |
| regular_0_ops | 517 | 40.618956 | 41.199226 | +0.580271 | +3 |
| regular_2_ops | 405 | 21.975309 | 22.469136 | +0.493827 | +2 |
| move_contents_3_ops | 406 | 26.600985 | 26.108374 | -0.492611 | -2 |
| ambiref_1_ops | 428 | 14.953271 | 15.420561 | +0.467290 | +2 |
| move_contents_2_ops | 399 | 21.553885 | 21.303258 | -0.250627 | -1 |
| ambiref_3_ops | 409 | 26.650367 | 26.894866 | +0.244499 | +1 |
| regular_1_ops | 409 | 15.647922 | 15.403423 | -0.244499 | -1 |
| ambiref_2_ops | 413 | 24.939467 | 24.697337 | -0.242131 | -1 |
| ambiref_4_ops | 434 | 34.562212 | 34.331797 | -0.230415 | -1 |

### COMPS

| subtask | n | A acc | B acc | B-A acc | net items |
|---|---:|---:|---:|---:|---:|
| wugs_dist_before | 13896 | 37.478411 | 37.938975 | +0.460564 | +64 |
| wugs_dist_in_between | 13896 | 65.443293 | 65.371330 | -0.071963 | -10 |
| base | 49340 | 53.244832 | 53.295501 | +0.050669 | +25 |
| wugs | 13896 | 52.022165 | 52.000576 | -0.021589 | -3 |

### GlobalPIQA_parallel

| subtask | n | A acc | B acc | B-A acc | net items |
|---|---:|---:|---:|---:|---:|
| global_piqa_parallel | 103 | 29.126214 | 29.126214 | +0.000000 | +0 |

### GlobalPIQA_nonparallel

| subtask | n | A acc | B acc | B-A acc | net items |
|---|---:|---:|---:|---:|---:|
| global_piqa_nonparallel | 100 | 48.000000 | 47.000000 | -1.000000 | -1 |

## Interpretation note

Official BabyLM columns use macro averages over subtasks and SuperGLUE primary metrics; micro item flips diagnose which decisions changed but do not directly equal leaderboard movement. Treat deltas as measurement evidence, not as a causal mechanism without training provenance and controls.

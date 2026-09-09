# earlier analysis official-coordinate transition comparison: pa87_alpha050_official vs coherent86_alpha050_existing_official

## Score summary

Scores labelled `computed` are recomputed from prediction files using the official macro over subtasks or SuperGLUE primary metrics where applicable. `payload` is the saved evaluator score. Item flips below are micro counts and should not be read as official score movement when subtasks have unequal sizes.

| column | A payload | B payload | payload B-A | A computed | B computed | computed B-A | A payload-computed | B payload-computed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BLiMP | 68.540000 | 68.420000 | -0.120000 | 68.544710 | 68.419053 | -0.125657 | -0.004710 | +0.000947 |
| Supplement | 63.150000 | 63.780000 | +0.630000 | 63.145914 | 63.780333 | +0.634419 | +0.004086 | -0.000333 |
| EWoK | 50.000000 | 50.260000 | +0.260000 | 49.996114 | 50.262103 | +0.265989 | +0.003886 | -0.002103 |
| Entity | 28.230000 | 28.250000 | +0.020000 | 28.230197 | 28.248605 | +0.018408 | -0.000197 | +0.001395 |
| COMPS | 52.110000 | 52.150000 | +0.040000 | 52.108975 | 52.151595 | +0.042620 | +0.001025 | -0.001595 |
| GlobalPIQA_parallel | 30.100000 | 29.130000 | -0.970000 | 30.097087 | 29.126214 | -0.970874 | +0.002913 | +0.003786 |
| GlobalPIQA_nonparallel | 48.000000 | 47.000000 | -1.000000 | 48.000000 | 47.000000 | -1.000000 | +0.000000 | +0.000000 |
| GlobalPIQA | 39.050000 | 38.065000 | -0.985000 | 39.048544 | 38.063107 | -0.985437 | +0.001456 | +0.001893 |
| Reading | 8.165000 | 8.215000 | +0.050000 | 8.165000 | 8.215000 | +0.050000 | +0.000000 | +0.000000 |
| SuperGLUE | 69.789755 | 69.777968 | -0.011787 | 69.789755 | 69.777968 | -0.011787 | +0.000000 | +0.000000 |
| AoA |  | 0.000000 |  |  | 0.000000 |  |  | +0.000000 |
| cheap7_mean | 44.177857 | 44.162857 | -0.015000 | 44.177065 | 44.162828 | -0.014237 | +0.000792 | +0.000029 |
| projected_overall_with_aoa0_if_missing | 42.114973 | 42.101996 | -0.012976 | 42.114357 | 42.101974 | -0.012383 | +0.000616 | +0.000023 |
| SuperGLUE/boolq |  |  |  | 68.562691 | 68.562691 | +0.000000 |  |  |
| SuperGLUE/mnli |  |  |  | 59.800326 | 59.800326 | +0.000000 |  |  |
| SuperGLUE/mrpc |  |  |  | 88.737201 | 88.737201 | +0.000000 |  |  |
| SuperGLUE/multirc |  |  |  | 67.367987 | 67.285479 | -0.082508 |  |  |
| SuperGLUE/qqp |  |  |  | 71.519959 | 71.519959 | +0.000000 |  |  |
| SuperGLUE/rte |  |  |  | 63.309353 | 63.309353 | +0.000000 |  |  |
| SuperGLUE/wsc |  |  |  | 69.230769 | 69.230769 | +0.000000 |  |  |

## Prediction-level transitions

| column | n | official-macro A | official-macro B | official-macro B-A | micro A | micro B | micro B-A | B-only | A-only | net item |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BLiMP | 59875 | 68.544710 | 68.419053 | -0.125657 | 68.425887 | 68.302296 | -0.123591 | 264 | 338 | -74 |
| Supplement | 5218 | 63.145914 | 63.780333 | +0.634419 | 73.936374 | 74.089690 | +0.153315 | 22 | 14 | +8 |
| EWoK | 7618 | 49.996114 | 50.262103 | +0.265989 | 50.065634 | 50.210029 | +0.144395 | 96 | 85 | +11 |
| Entity | 6780 | 28.230197 | 28.248605 | +0.018408 | 27.817109 | 27.935103 | +0.117994 | 52 | 44 | +8 |
| COMPS | 91028 | 52.108975 | 52.151595 | +0.042620 | 52.566243 | 52.597003 | +0.030760 | 1049 | 1021 | +28 |
| GlobalPIQA_parallel | 103 | 30.097087 | 29.126214 | -0.970874 | 30.097087 | 29.126214 | -0.970874 | 0 | 1 | -1 |
| GlobalPIQA_nonparallel | 100 | 48.000000 | 47.000000 | -1.000000 | 48.000000 | 47.000000 | -1.000000 | 1 | 2 | -1 |

## SuperGLUE task transitions

| task | metric | n | A primary | B primary | B-A primary | A acc | B acc | B-A acc | B-only | A-only | net item |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| boolq | accuracy | 1635 | 68.562691 | 68.562691 | +0.000000 | 68.562691 | 68.562691 | +0.000000 | 0 | 0 | +0 |
| mnli | accuracy | 4908 | 59.800326 | 59.800326 | +0.000000 | 59.800326 | 59.800326 | +0.000000 | 0 | 0 | +0 |
| mrpc | f1 | 204 | 88.737201 | 88.737201 | +0.000000 | 83.823529 | 83.823529 | +0.000000 | 0 | 0 | +0 |
| multirc | accuracy | 2424 | 67.367987 | 67.285479 | -0.082508 | 67.367987 | 67.285479 | -0.082508 | 123 | 125 | -2 |
| qqp | f1 | 20215 | 71.519959 | 71.519959 | +0.000000 | 77.976750 | 77.976750 | +0.000000 | 0 | 0 | +0 |
| rte | accuracy | 139 | 63.309353 | 63.309353 | +0.000000 | 63.309353 | 63.309353 | +0.000000 | 0 | 0 | +0 |
| wsc | accuracy | 52 | 69.230769 | 69.230769 | +0.000000 | 69.230769 | 69.230769 | +0.000000 | 0 | 0 | +0 |

## Largest subtask movements by official-macro component


### BLiMP

| subtask | n | A acc | B acc | B-A acc | net items |
|---|---:|---:|---:|---:|---:|
| npi_present_2 | 914 | 37.199125 | 35.776805 | -1.422319 | -13 |
| wh_island | 960 | 56.979167 | 55.833333 | -1.145833 | -11 |
| ellipsis_n_bar_1 | 802 | 74.812968 | 73.815461 | -0.997506 | -8 |
| principle_A_c_command | 946 | 50.317125 | 51.268499 | +0.951374 | +9 |
| only_npi_licensor_present | 882 | 80.725624 | 79.818594 | -0.907029 | -8 |
| distractor_agreement_relational_noun | 788 | 49.365482 | 50.253807 | +0.888325 | +7 |
| npi_present_1 | 909 | 31.903190 | 31.023102 | -0.880088 | -8 |
| principle_A_domain_2 | 915 | 53.661202 | 52.896175 | -0.765027 | -7 |
| wh_vs_that_with_gap | 919 | 43.960827 | 44.722524 | +0.761697 | +7 |
| left_branch_island_simple_question | 951 | 51.524711 | 50.788644 | -0.736067 | -7 |
| expletive_it_object_raising | 759 | 68.115942 | 67.457181 | -0.658762 | -5 |
| matrix_question_npi_licensor_present | 929 | 41.227126 | 40.581270 | -0.645856 | -6 |
| tough_vs_raising_1 | 948 | 36.075949 | 36.708861 | +0.632911 | +6 |
| existential_there_object_raising | 812 | 73.645320 | 73.029557 | -0.615764 | -5 |
| complex_NP_island | 846 | 39.479905 | 38.888889 | -0.591017 | -5 |

### Supplement

| subtask | n | A acc | B acc | B-A acc | net items |
|---|---:|---:|---:|---:|---:|
| qa_congruence_easy | 64 | 68.750000 | 70.312500 | +1.562500 | +1 |
| turn_taking | 280 | 67.500000 | 68.214286 | +0.714286 | +2 |
| qa_congruence_tricky | 165 | 49.090909 | 49.696970 | +0.606061 | +1 |
| hypernym | 842 | 49.524941 | 49.762470 | +0.237530 | +2 |
| subject_aux_inversion | 3867 | 80.863719 | 80.915438 | +0.051720 | +2 |

### EWoK

| subtask | n | A acc | B acc | B-A acc | net items |
|---|---:|---:|---:|---:|---:|
| material-dynamics | 770 | 50.909091 | 52.727273 | +1.818182 | +14 |
| quantitative-properties | 314 | 48.089172 | 49.681529 | +1.592357 | +5 |
| spatial-relations | 490 | 46.734694 | 47.959184 | +1.224490 | +6 |
| social-properties | 328 | 47.560976 | 46.646341 | -0.914634 | -3 |
| physical-relations | 818 | 49.755501 | 49.022005 | -0.733496 | -6 |
| social-interactions | 294 | 54.421769 | 53.741497 | -0.680272 | -2 |
| material-properties | 170 | 50.588235 | 51.176471 | +0.588235 | +1 |
| agent-properties | 2210 | 50.407240 | 50.000000 | -0.407240 | -9 |
| social-relations | 1548 | 50.968992 | 51.227390 | +0.258398 | +4 |
| physical-interactions | 556 | 48.021583 | 48.201439 | +0.179856 | +1 |
| physical-dynamics | 120 | 52.500000 | 52.500000 | +0.000000 | +0 |

### Entity

| subtask | n | A acc | B acc | B-A acc | net items |
|---|---:|---:|---:|---:|---:|
| move_contents_0_ops | 516 | 45.155039 | 47.093023 | +1.937984 | +10 |
| ambiref_0_ops | 508 | 25.590551 | 26.574803 | +0.984252 | +5 |
| move_contents_5_ops | 116 | 42.241379 | 41.379310 | -0.862069 | -1 |
| move_contents_4_ops | 353 | 31.161473 | 30.311615 | -0.849858 | -3 |
| move_contents_3_ops | 406 | 26.847291 | 26.108374 | -0.738916 | -3 |
| regular_0_ops | 517 | 40.618956 | 41.199226 | +0.580271 | +3 |
| regular_1_ops | 409 | 14.914425 | 15.403423 | +0.488998 | +2 |
| regular_3_ops | 425 | 30.588235 | 30.117647 | -0.470588 | -2 |
| ambiref_1_ops | 428 | 14.953271 | 15.420561 | +0.467290 | +2 |
| regular_4_ops | 388 | 27.835052 | 27.577320 | -0.257732 | -1 |
| regular_2_ops | 405 | 22.716049 | 22.469136 | -0.246914 | -1 |
| ambiref_2_ops | 413 | 24.939467 | 24.697337 | -0.242131 | -1 |
| ambiref_4_ops | 434 | 34.562212 | 34.331797 | -0.230415 | -1 |
| move_contents_1_ops | 437 | 17.391304 | 17.162471 | -0.228833 | -1 |
| ambiref_3_ops | 409 | 26.894866 | 26.894866 | +0.000000 | +0 |

### COMPS

| subtask | n | A acc | B acc | B-A acc | net items |
|---|---:|---:|---:|---:|---:|
| wugs_dist_before | 13896 | 37.658319 | 37.938975 | +0.280656 | +39 |
| wugs_dist_in_between | 13896 | 65.486471 | 65.371330 | -0.115141 | -16 |
| base | 49340 | 53.283340 | 53.295501 | +0.012161 | +6 |
| wugs | 13896 | 52.007772 | 52.000576 | -0.007196 | -1 |

### GlobalPIQA_parallel

| subtask | n | A acc | B acc | B-A acc | net items |
|---|---:|---:|---:|---:|---:|
| global_piqa_parallel | 103 | 30.097087 | 29.126214 | -0.970874 | -1 |

### GlobalPIQA_nonparallel

| subtask | n | A acc | B acc | B-A acc | net items |
|---|---:|---:|---:|---:|---:|
| global_piqa_nonparallel | 100 | 48.000000 | 47.000000 | -1.000000 | -1 |

## Interpretation note

Official BabyLM columns use macro averages over subtasks and SuperGLUE primary metrics; micro item flips diagnose which decisions changed but do not directly equal leaderboard movement. Treat deltas as measurement evidence, not as a causal mechanism without training provenance and controls.

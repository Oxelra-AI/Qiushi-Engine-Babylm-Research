# earlier analysis official-coordinate transition comparison: coherent86_alpha050 vs coherent86_alpha075

## Score summary

Scores labelled `computed` are recomputed from prediction files using the official macro over subtasks or SuperGLUE primary metrics where applicable. `payload` is the saved evaluator score. Item flips below are micro counts and should not be read as official score movement when subtasks have unequal sizes.

| column | A payload | B payload | payload B-A | A computed | B computed | computed B-A | A payload-computed | B payload-computed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BLiMP | 68.510000 | 68.540000 | +0.030000 | 68.515970 | 68.544710 | +0.028740 | -0.005970 | -0.004710 |
| Supplement | 63.640000 | 63.150000 | -0.490000 | 63.635369 | 63.145914 | -0.489455 | +0.004631 | +0.004086 |
| EWoK | 50.020000 | 50.000000 | -0.020000 | 50.019616 | 49.996114 | -0.023502 | +0.000384 | +0.003886 |
| Entity | 28.320000 | 28.230000 | -0.090000 | 28.322219 | 28.230197 | -0.092022 | -0.002219 | -0.000197 |
| COMPS | 52.050000 | 52.110000 | +0.060000 | 52.047175 | 52.108975 | +0.061800 | +0.002825 | +0.001025 |
| GlobalPIQA_parallel | 29.130000 | 30.100000 | +0.970000 | 29.126214 | 30.097087 | +0.970874 | +0.003786 | +0.002913 |
| GlobalPIQA_nonparallel | 48.000000 | 48.000000 | +0.000000 | 48.000000 | 48.000000 | +0.000000 | +0.000000 | +0.000000 |
| GlobalPIQA | 38.565000 | 39.050000 | +0.485000 | 38.563107 | 39.048544 | +0.485437 | +0.001893 | +0.001456 |
| Reading | 8.165000 | 8.165000 | +0.000000 | 8.165000 | 8.165000 | +0.000000 | +0.000000 | +0.000000 |
| SuperGLUE | 69.819222 | 69.789755 | -0.029467 | 69.819222 | 69.789755 | -0.029467 | +0.000000 | +0.000000 |
| cheap7_mean | 44.181429 | 44.177857 | -0.003571 | 44.181208 | 44.177065 | -0.004143 | +0.000221 | +0.000792 |
| projected_overall_with_aoa0_if_missing | 42.121025 | 42.114973 | -0.006052 | 42.120853 | 42.114357 | -0.006497 | +0.000172 | +0.000616 |
| SuperGLUE/boolq |  |  |  | 68.562691 | 68.562691 | +0.000000 |  |  |
| SuperGLUE/mnli |  |  |  | 59.800326 | 59.800326 | +0.000000 |  |  |
| SuperGLUE/mrpc |  |  |  | 88.737201 | 88.737201 | +0.000000 |  |  |
| SuperGLUE/multirc |  |  |  | 67.574257 | 67.367987 | -0.206271 |  |  |
| SuperGLUE/qqp |  |  |  | 71.519959 | 71.519959 | +0.000000 |  |  |
| SuperGLUE/rte |  |  |  | 63.309353 | 63.309353 | +0.000000 |  |  |
| SuperGLUE/wsc |  |  |  | 69.230769 | 69.230769 | +0.000000 |  |  |

## Prediction-level transitions

| column | n | official-macro A | official-macro B | official-macro B-A | micro A | micro B | micro B-A | B-only | A-only | net item |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BLiMP | 59875 | 68.515970 | 68.544710 | +0.028740 | 68.404175 | 68.425887 | +0.021712 | 194 | 181 | +13 |
| Supplement | 5218 | 63.635369 | 63.145914 | -0.489455 | 73.974703 | 73.936374 | -0.038329 | 10 | 12 | -2 |
| EWoK | 7618 | 50.019616 | 49.996114 | -0.023502 | 49.986873 | 50.065634 | +0.078761 | 40 | 34 | +6 |
| Entity | 6780 | 28.322219 | 28.230197 | -0.092022 | 27.831858 | 27.817109 | -0.014749 | 16 | 17 | -1 |
| COMPS | 91028 | 52.047175 | 52.108975 | +0.061800 | 52.513512 | 52.566243 | +0.052731 | 578 | 530 | +48 |
| GlobalPIQA_parallel | 103 | 29.126214 | 30.097087 | +0.970874 | 29.126214 | 30.097087 | +0.970874 | 1 | 0 | +1 |
| GlobalPIQA_nonparallel | 100 | 48.000000 | 48.000000 | +0.000000 | 48.000000 | 48.000000 | +0.000000 | 0 | 0 | +0 |

## SuperGLUE task transitions

| task | metric | n | A primary | B primary | B-A primary | A acc | B acc | B-A acc | B-only | A-only | net item |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| boolq | accuracy | 1635 | 68.562691 | 68.562691 | +0.000000 | 68.562691 | 68.562691 | +0.000000 | 0 | 0 | +0 |
| mnli | accuracy | 4908 | 59.800326 | 59.800326 | +0.000000 | 59.800326 | 59.800326 | +0.000000 | 0 | 0 | +0 |
| mrpc | f1 | 204 | 88.737201 | 88.737201 | +0.000000 | 83.823529 | 83.823529 | +0.000000 | 0 | 0 | +0 |
| multirc | accuracy | 2424 | 67.574257 | 67.367987 | -0.206271 | 67.574257 | 67.367987 | -0.206271 | 125 | 130 | -5 |
| qqp | f1 | 20215 | 71.519959 | 71.519959 | +0.000000 | 77.976750 | 77.976750 | +0.000000 | 0 | 0 | +0 |
| rte | accuracy | 139 | 63.309353 | 63.309353 | +0.000000 | 63.309353 | 63.309353 | +0.000000 | 0 | 0 | +0 |
| wsc | accuracy | 52 | 69.230769 | 69.230769 | +0.000000 | 69.230769 | 69.230769 | +0.000000 | 0 | 0 | +0 |

## Largest subtask movements by official-macro component


### BLiMP

| subtask | n | A acc | B acc | B-A acc | net items |
|---|---:|---:|---:|---:|---:|
| npi_present_2 | 914 | 34.901532 | 37.199125 | +2.297593 | +21 |
| ellipsis_n_bar_1 | 802 | 73.316708 | 74.812968 | +1.496259 | +12 |
| principle_A_domain_1 | 914 | 77.024070 | 76.039387 | -0.984683 | -9 |
| adjunct_island | 928 | 76.185345 | 75.431034 | -0.754310 | -7 |
| wh_island | 960 | 57.604167 | 56.979167 | -0.625000 | -6 |
| principle_A_reconstruction | 967 | 41.882110 | 41.261634 | -0.620476 | -6 |
| superlative_quantifiers_2 | 986 | 82.251521 | 81.643002 | -0.608519 | -6 |
| tough_vs_raising_1 | 948 | 36.603376 | 36.075949 | -0.527426 | -5 |
| sentential_subject_island | 961 | 32.778356 | 32.258065 | -0.520291 | -5 |
| distractor_agreement_relational_noun | 788 | 49.873096 | 49.365482 | -0.507614 | -4 |
| passive_2 | 903 | 73.754153 | 73.311185 | -0.442968 | -4 |
| npi_present_1 | 909 | 31.463146 | 31.903190 | +0.440044 | +4 |
| existential_there_quantifiers_2 | 911 | 40.944018 | 40.504940 | -0.439078 | -4 |
| principle_A_case_2 | 915 | 83.825137 | 84.262295 | +0.437158 | +4 |
| matrix_question_npi_licensor_present | 929 | 41.657696 | 41.227126 | -0.430571 | -4 |

### Supplement

| subtask | n | A acc | B acc | B-A acc | net items |
|---|---:|---:|---:|---:|---:|
| qa_congruence_easy | 64 | 70.312500 | 68.750000 | -1.562500 | -1 |
| qa_congruence_tricky | 165 | 49.696970 | 49.090909 | -0.606061 | -1 |
| hypernym | 842 | 49.881235 | 49.524941 | -0.356295 | -3 |
| subject_aux_inversion | 3867 | 80.786139 | 80.863719 | +0.077580 | +3 |
| turn_taking | 280 | 67.500000 | 67.500000 | +0.000000 | +0 |

### EWoK

| subtask | n | A acc | B acc | B-A acc | net items |
|---|---:|---:|---:|---:|---:|
| material-properties | 170 | 51.764706 | 50.588235 | -1.176471 | -2 |
| social-properties | 328 | 46.646341 | 47.560976 | +0.914634 | +3 |
| quantitative-properties | 314 | 48.726115 | 48.089172 | -0.636943 | -2 |
| physical-interactions | 556 | 48.561151 | 48.021583 | -0.539568 | -3 |
| material-dynamics | 770 | 50.519481 | 50.909091 | +0.389610 | +3 |
| social-interactions | 294 | 54.081633 | 54.421769 | +0.340136 | +1 |
| social-relations | 1548 | 50.645995 | 50.968992 | +0.322997 | +5 |
| spatial-relations | 490 | 46.530612 | 46.734694 | +0.204082 | +1 |
| physical-relations | 818 | 49.877751 | 49.755501 | -0.122249 | -1 |
| agent-properties | 2210 | 50.361991 | 50.407240 | +0.045249 | +1 |
| physical-dynamics | 120 | 52.500000 | 52.500000 | +0.000000 | +0 |

### Entity

| subtask | n | A acc | B acc | B-A acc | net items |
|---|---:|---:|---:|---:|---:|
| regular_5_ops | 94 | 29.787234 | 28.723404 | -1.063830 | -1 |
| move_contents_5_ops | 116 | 43.103448 | 42.241379 | -0.862069 | -1 |
| regular_2_ops | 405 | 21.975309 | 22.716049 | +0.740741 | +3 |
| regular_1_ops | 409 | 15.647922 | 14.914425 | -0.733496 | -3 |
| regular_3_ops | 425 | 31.058824 | 30.588235 | -0.470588 | -2 |
| ambiref_0_ops | 508 | 25.196850 | 25.590551 | +0.393701 | +2 |
| move_contents_0_ops | 516 | 45.542636 | 45.155039 | -0.387597 | -2 |
| regular_4_ops | 388 | 27.577320 | 27.835052 | +0.257732 | +1 |
| move_contents_2_ops | 399 | 21.553885 | 21.303258 | -0.250627 | -1 |
| move_contents_3_ops | 406 | 26.600985 | 26.847291 | +0.246305 | +1 |
| ambiref_3_ops | 409 | 26.650367 | 26.894866 | +0.244499 | +1 |
| move_contents_1_ops | 437 | 17.162471 | 17.391304 | +0.228833 | +1 |
| regular_0_ops | 517 | 40.618956 | 40.618956 | +0.000000 | +0 |
| ambiref_4_ops | 434 | 34.562212 | 34.562212 | +0.000000 | +0 |
| ambiref_1_ops | 428 | 14.953271 | 14.953271 | +0.000000 | +0 |

### COMPS

| subtask | n | A acc | B acc | B-A acc | net items |
|---|---:|---:|---:|---:|---:|
| wugs_dist_before | 13896 | 37.478411 | 37.658319 | +0.179908 | +25 |
| wugs_dist_in_between | 13896 | 65.443293 | 65.486471 | +0.043178 | +6 |
| base | 49340 | 53.244832 | 53.283340 | +0.038508 | +19 |
| wugs | 13896 | 52.022165 | 52.007772 | -0.014393 | -2 |

### GlobalPIQA_parallel

| subtask | n | A acc | B acc | B-A acc | net items |
|---|---:|---:|---:|---:|---:|
| global_piqa_parallel | 103 | 29.126214 | 30.097087 | +0.970874 | +1 |

### GlobalPIQA_nonparallel

| subtask | n | A acc | B acc | B-A acc | net items |
|---|---:|---:|---:|---:|---:|
| global_piqa_nonparallel | 100 | 48.000000 | 48.000000 | +0.000000 | +0 |

## Interpretation note

Official BabyLM columns use macro averages over subtasks and SuperGLUE primary metrics; micro item flips diagnose which decisions changed but do not directly equal leaderboard movement. Treat deltas as measurement evidence, not as a causal mechanism without training provenance and controls.

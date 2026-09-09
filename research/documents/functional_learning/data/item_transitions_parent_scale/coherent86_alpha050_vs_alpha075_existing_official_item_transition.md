# earlier analysis item-transition comparison: coherent86_alpha050 vs coherent86_alpha075

## Score summary

| column | A | B | B-A |
|---|---:|---:|---:|
| BLiMP | 68.5100 | 68.5400 | +0.0300 |
| Supplement | 63.6400 | 63.1500 | -0.4900 |
| EWoK | 50.0200 | 50.0000 | -0.0200 |
| Entity | 28.3200 | 28.2300 | -0.0900 |
| COMPS | 52.0500 | 52.1100 | +0.0600 |
| GlobalPIQA | 38.5650 | 39.0500 | +0.4850 |
| Reading | 8.1650 | 8.1650 | +0.0000 |
| SuperGLUE | 69.8192 | 69.7898 | -0.0295 |
| cheap7_or_zero_readout_mean | 44.1814 | 44.1779 | -0.0036 |

## Prediction-level transitions

| column | n | A acc | B acc | B-A acc | B-only | A-only | net items |
|---|---:|---:|---:|---:|---:|---:|---:|
| BLiMP | 59875 | 68.4042 | 68.4259 | +0.0217 | 194 | 181 | +13 |
| Supplement | 5218 | 73.9747 | 73.9364 | -0.0383 | 10 | 12 | -2 |
| EWoK | 7618 | 49.9869 | 50.0656 | +0.0788 | 40 | 34 | +6 |
| Entity | 6780 | 27.8319 | 27.8171 | -0.0147 | 16 | 17 | -1 |
| COMPS | 91028 | 52.5135 | 52.5662 | +0.0527 | 578 | 530 | +48 |
| GlobalPIQA_parallel | 103 | 29.1262 | 30.0971 | +0.9709 | 1 | 0 | +1 |
| GlobalPIQA_nonparallel | 100 | 48.0000 | 48.0000 | +0.0000 | 0 | 0 | +0 |
| SuperGLUE/boolq | 1635 | 68.5627 | 68.5627 | +0.0000 | 0 | 0 | +0 |
| SuperGLUE/mnli | 4908 | 59.8003 | 59.8003 | +0.0000 | 0 | 0 | +0 |
| SuperGLUE/mrpc | 204 | 83.8235 | 83.8235 | +0.0000 | 0 | 0 | +0 |
| SuperGLUE/multirc | 2424 | 67.5743 | 67.3680 | -0.2063 | 125 | 130 | -5 |
| SuperGLUE/qqp | 20215 | 77.9767 | 77.9767 | +0.0000 | 0 | 0 | +0 |
| SuperGLUE/rte | 139 | 63.3094 | 63.3094 | +0.0000 | 0 | 0 | +0 |
| SuperGLUE/wsc | 52 | 69.2308 | 69.2308 | +0.0000 | 0 | 0 | +0 |

## Largest subtask movements by absolute accuracy difference


### BLiMP

| subtask | n | A acc | B acc | B-A acc | net items |
|---|---:|---:|---:|---:|---:|
| npi_present_2 | 914 | 34.902 | 37.199 | +2.298 | +21 |
| ellipsis_n_bar_1 | 802 | 73.317 | 74.813 | +1.496 | +12 |
| principle_A_domain_1 | 914 | 77.024 | 76.039 | -0.985 | -9 |
| adjunct_island | 928 | 76.185 | 75.431 | -0.754 | -7 |
| wh_island | 960 | 57.604 | 56.979 | -0.625 | -6 |
| principle_A_reconstruction | 967 | 41.882 | 41.262 | -0.620 | -6 |
| superlative_quantifiers_2 | 986 | 82.252 | 81.643 | -0.609 | -6 |
| tough_vs_raising_1 | 948 | 36.603 | 36.076 | -0.527 | -5 |
| sentential_subject_island | 961 | 32.778 | 32.258 | -0.520 | -5 |
| distractor_agreement_relational_noun | 788 | 49.873 | 49.365 | -0.508 | -4 |
| passive_2 | 903 | 73.754 | 73.311 | -0.443 | -4 |
| npi_present_1 | 909 | 31.463 | 31.903 | +0.440 | +4 |
| existential_there_quantifiers_2 | 911 | 40.944 | 40.505 | -0.439 | -4 |
| principle_A_case_2 | 915 | 83.825 | 84.262 | +0.437 | +4 |
| matrix_question_npi_licensor_present | 929 | 41.658 | 41.227 | -0.431 | -4 |

### Supplement

| subtask | n | A acc | B acc | B-A acc | net items |
|---|---:|---:|---:|---:|---:|
| qa_congruence_easy | 64 | 70.312 | 68.750 | -1.562 | -1 |
| qa_congruence_tricky | 165 | 49.697 | 49.091 | -0.606 | -1 |
| hypernym | 842 | 49.881 | 49.525 | -0.356 | -3 |
| subject_aux_inversion | 3867 | 80.786 | 80.864 | +0.078 | +3 |
| turn_taking | 280 | 67.500 | 67.500 | +0.000 | +0 |

### EWoK

| subtask | n | A acc | B acc | B-A acc | net items |
|---|---:|---:|---:|---:|---:|
| material-properties | 170 | 51.765 | 50.588 | -1.176 | -2 |
| social-properties | 328 | 46.646 | 47.561 | +0.915 | +3 |
| quantitative-properties | 314 | 48.726 | 48.089 | -0.637 | -2 |
| physical-interactions | 556 | 48.561 | 48.022 | -0.540 | -3 |
| material-dynamics | 770 | 50.519 | 50.909 | +0.390 | +3 |
| social-interactions | 294 | 54.082 | 54.422 | +0.340 | +1 |
| social-relations | 1548 | 50.646 | 50.969 | +0.323 | +5 |
| spatial-relations | 490 | 46.531 | 46.735 | +0.204 | +1 |
| physical-relations | 818 | 49.878 | 49.756 | -0.122 | -1 |
| agent-properties | 2210 | 50.362 | 50.407 | +0.045 | +1 |
| physical-dynamics | 120 | 52.500 | 52.500 | +0.000 | +0 |

### Entity

| subtask | n | A acc | B acc | B-A acc | net items |
|---|---:|---:|---:|---:|---:|
| regular_5_ops | 94 | 29.787 | 28.723 | -1.064 | -1 |
| move_contents_5_ops | 116 | 43.103 | 42.241 | -0.862 | -1 |
| regular_2_ops | 405 | 21.975 | 22.716 | +0.741 | +3 |
| regular_1_ops | 409 | 15.648 | 14.914 | -0.733 | -3 |
| regular_3_ops | 425 | 31.059 | 30.588 | -0.471 | -2 |
| ambiref_0_ops | 508 | 25.197 | 25.591 | +0.394 | +2 |
| move_contents_0_ops | 516 | 45.543 | 45.155 | -0.388 | -2 |
| regular_4_ops | 388 | 27.577 | 27.835 | +0.258 | +1 |
| move_contents_2_ops | 399 | 21.554 | 21.303 | -0.251 | -1 |
| move_contents_3_ops | 406 | 26.601 | 26.847 | +0.246 | +1 |
| ambiref_3_ops | 409 | 26.650 | 26.895 | +0.244 | +1 |
| move_contents_1_ops | 437 | 17.162 | 17.391 | +0.229 | +1 |
| regular_0_ops | 517 | 40.619 | 40.619 | +0.000 | +0 |
| ambiref_4_ops | 434 | 34.562 | 34.562 | +0.000 | +0 |
| ambiref_1_ops | 428 | 14.953 | 14.953 | +0.000 | +0 |

### COMPS

| subtask | n | A acc | B acc | B-A acc | net items |
|---|---:|---:|---:|---:|---:|
| wugs_dist_before | 13896 | 37.478 | 37.658 | +0.180 | +25 |
| wugs_dist_in_between | 13896 | 65.443 | 65.486 | +0.043 | +6 |
| base | 49340 | 53.245 | 53.283 | +0.039 | +19 |
| wugs | 13896 | 52.022 | 52.008 | -0.014 | -2 |

### GlobalPIQA_parallel

| subtask | n | A acc | B acc | B-A acc | net items |
|---|---:|---:|---:|---:|---:|
| global_piqa_parallel | 103 | 29.126 | 30.097 | +0.971 | +1 |

### GlobalPIQA_nonparallel

| subtask | n | A acc | B acc | B-A acc | net items |
|---|---:|---:|---:|---:|---:|
| global_piqa_nonparallel | 100 | 48.000 | 48.000 | +0.000 | +0 |

### SuperGLUE/boolq

| subtask | n | A acc | B acc | B-A acc | net items |
|---|---:|---:|---:|---:|---:|
| boolq | 1635 | 68.563 | 68.563 | +0.000 | +0 |

### SuperGLUE/mnli

| subtask | n | A acc | B acc | B-A acc | net items |
|---|---:|---:|---:|---:|---:|
| mnli | 4908 | 59.800 | 59.800 | +0.000 | +0 |

### SuperGLUE/mrpc

| subtask | n | A acc | B acc | B-A acc | net items |
|---|---:|---:|---:|---:|---:|
| mrpc | 204 | 83.824 | 83.824 | +0.000 | +0 |

### SuperGLUE/multirc

| subtask | n | A acc | B acc | B-A acc | net items |
|---|---:|---:|---:|---:|---:|
| multirc | 2424 | 67.574 | 67.368 | -0.206 | -5 |

### SuperGLUE/qqp

| subtask | n | A acc | B acc | B-A acc | net items |
|---|---:|---:|---:|---:|---:|
| qqp | 20215 | 77.977 | 77.977 | +0.000 | +0 |

### SuperGLUE/rte

| subtask | n | A acc | B acc | B-A acc | net items |
|---|---:|---:|---:|---:|---:|
| rte | 139 | 63.309 | 63.309 | +0.000 | +0 |

### SuperGLUE/wsc

| subtask | n | A acc | B acc | B-A acc | net items |
|---|---:|---:|---:|---:|---:|
| wsc | 52 | 69.231 | 69.231 | +0.000 | +0 |

## Interpretation note

Positive B-A item counts show changed decisions on this evaluation coordinate, not by themselves a causal mechanism; compare with training provenance, scale-only controls, and full-score movement before attributing gains to new experience.

# earlier analysis official-coordinate transition comparison: dense_seed62065_zero_reading_current vs dense_seed62064_zero_reading_current

## Score summary

Scores labelled `computed` are recomputed from prediction files using the official macro over subtasks or SuperGLUE primary metrics where applicable. `payload` is the saved evaluator score. Item flips below are micro counts and should not be read as official score movement when subtasks have unequal sizes.

| column | A payload | B payload | payload B-A | A computed | B computed | computed B-A | A payload-computed | B payload-computed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BLiMP | 68.060000 | 68.020000 | -0.040000 | 68.065520 | 68.028300 | -0.037219 | -0.005520 | -0.008300 |
| Supplement | 63.040000 | 63.090000 | +0.050000 | 63.037493 | 63.093405 | +0.055913 | +0.002507 | -0.003405 |
| EWoK | 49.920000 | 49.770000 | -0.150000 | 49.922478 | 49.769931 | -0.152547 | -0.002478 | +0.000069 |
| Entity | 29.370000 | 29.420000 | +0.050000 | 29.366952 | 29.415692 | +0.048740 | +0.003048 | +0.004308 |
| COMPS | 52.150000 | 52.150000 | +0.000000 | 52.153718 | 52.145457 | -0.008261 | -0.003718 | +0.004543 |
| GlobalPIQA_parallel | 30.100000 | 30.100000 | +0.000000 | 30.097087 | 30.097087 | +0.000000 | +0.002913 | +0.002913 |
| GlobalPIQA_nonparallel | 50.000000 | 50.000000 | +0.000000 | 50.000000 | 50.000000 | +0.000000 | +0.000000 | +0.000000 |
| GlobalPIQA | 40.050000 | 40.050000 | +0.000000 | 40.048544 | 40.048544 | +0.000000 | +0.001456 | +0.001456 |
| Reading | 8.220000 | 8.210000 | -0.010000 | 8.220000 | 8.210000 | -0.010000 | +0.000000 | +0.000000 |
| SuperGLUE | 68.562691 | 68.562691 | +0.000000 |  |  |  |  |  |
| cheap7_mean | 44.401429 | 44.387143 | -0.014286 | 44.402101 | 44.387333 | -0.014768 | -0.000672 | -0.000190 |
| projected_overall_with_aoa0_if_missing | 42.152521 | 42.141410 | -0.011111 |  |  |  |  |  |

## Prediction-level transitions

| column | n | official-macro A | official-macro B | official-macro B-A | micro A | micro B | micro B-A | B-only | A-only | net item |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BLiMP | 59875 | 68.065520 | 68.028300 | -0.037219 | 67.949896 | 67.911482 | -0.038413 | 31 | 54 | -23 |
| Supplement | 5218 | 63.037493 | 63.093405 | +0.055913 | 73.878881 | 73.840552 | -0.038329 | 2 | 4 | -2 |
| EWoK | 7618 | 49.922478 | 49.769931 | -0.152547 | 50.157522 | 50.039380 | -0.118141 | 9 | 18 | -9 |
| Entity | 6780 | 29.366952 | 29.415692 | +0.048740 | 28.451327 | 28.466077 | +0.014749 | 5 | 4 | +1 |
| COMPS | 91028 | 52.153718 | 52.145457 | -0.008261 | 52.646438 | 52.645340 | -0.001099 | 125 | 126 | -1 |
| GlobalPIQA_parallel | 103 | 30.097087 | 30.097087 | +0.000000 | 30.097087 | 30.097087 | +0.000000 | 0 | 0 | +0 |
| GlobalPIQA_nonparallel | 100 | 50.000000 | 50.000000 | +0.000000 | 50.000000 | 50.000000 | +0.000000 | 0 | 0 | +0 |

## Largest subtask movements by official-macro component


### BLiMP

| subtask | n | A acc | B acc | B-A acc | net items |
|---|---:|---:|---:|---:|---:|
| wh_vs_that_with_gap | 919 | 45.048966 | 44.613711 | -0.435256 | -4 |
| wh_island | 960 | 58.437500 | 58.020833 | -0.416667 | -4 |
| wh_questions_object_gap | 859 | 56.344587 | 55.995343 | -0.349243 | -3 |
| wh_questions_subject_gap | 898 | 76.614699 | 76.948775 | +0.334076 | +3 |
| passive_2 | 903 | 73.532669 | 73.200443 | -0.332226 | -3 |
| existential_there_quantifiers_2 | 911 | 44.017563 | 43.688255 | -0.329308 | -3 |
| principle_A_c_command | 946 | 50.000000 | 49.682875 | -0.317125 | -3 |
| complex_NP_island | 846 | 39.361702 | 39.598109 | +0.236407 | +2 |
| intransitive | 868 | 57.373272 | 57.142857 | -0.230415 | -2 |
| npi_present_1 | 909 | 29.812981 | 30.033003 | +0.220022 | +2 |
| wh_vs_that_with_gap_long_distance | 910 | 14.725275 | 14.505495 | -0.219780 | -2 |
| principle_A_domain_1 | 914 | 73.522976 | 73.741794 | +0.218818 | +2 |
| coordinate_structure_constraint_object_extraction | 949 | 77.133825 | 76.923077 | -0.210748 | -2 |
| left_branch_island_simple_question | 951 | 49.316509 | 49.106204 | -0.210305 | -2 |
| sentential_subject_island | 961 | 34.339230 | 34.547347 | +0.208117 | +2 |

### Supplement

| subtask | n | A acc | B acc | B-A acc | net items |
|---|---:|---:|---:|---:|---:|
| turn_taking | 280 | 67.500000 | 67.857143 | +0.357143 | +1 |
| subject_aux_inversion | 3867 | 80.915438 | 80.837859 | -0.077580 | -3 |
| hypernym | 842 | 48.931116 | 48.931116 | +0.000000 | +0 |
| qa_congruence_tricky | 165 | 49.090909 | 49.090909 | +0.000000 | +0 |
| qa_congruence_easy | 64 | 68.750000 | 68.750000 | +0.000000 | +0 |

### EWoK

| subtask | n | A acc | B acc | B-A acc | net items |
|---|---:|---:|---:|---:|---:|
| social-interactions | 294 | 54.761905 | 53.401361 | -1.360544 | -4 |
| social-relations | 1548 | 50.968992 | 50.516796 | -0.452196 | -7 |
| social-properties | 328 | 45.121951 | 44.817073 | -0.304878 | -1 |
| material-dynamics | 770 | 54.675325 | 54.935065 | +0.259740 | +2 |
| physical-interactions | 556 | 46.942446 | 47.122302 | +0.179856 | +1 |
| agent-properties | 2210 | 50.135747 | 50.135747 | +0.000000 | +0 |
| physical-relations | 818 | 49.388753 | 49.388753 | +0.000000 | +0 |
| spatial-relations | 490 | 46.734694 | 46.734694 | +0.000000 | +0 |
| quantitative-properties | 314 | 48.407643 | 48.407643 | +0.000000 | +0 |
| material-properties | 170 | 51.176471 | 51.176471 | +0.000000 | +0 |
| physical-dynamics | 120 | 50.833333 | 50.833333 | +0.000000 | +0 |

### Entity

| subtask | n | A acc | B acc | B-A acc | net items |
|---|---:|---:|---:|---:|---:|
| ambiref_5_ops | 123 | 36.585366 | 37.398374 | +0.813008 | +1 |
| move_contents_4_ops | 353 | 34.844193 | 35.410765 | +0.566572 | +2 |
| regular_4_ops | 388 | 30.412371 | 30.154639 | -0.257732 | -1 |
| regular_2_ops | 405 | 24.444444 | 24.691358 | +0.246914 | +1 |
| ambiref_3_ops | 409 | 28.850856 | 28.606357 | -0.244499 | -1 |
| ambiref_2_ops | 413 | 24.455206 | 24.213075 | -0.242131 | -1 |
| ambiref_1_ops | 428 | 14.018692 | 13.785047 | -0.233645 | -1 |
| move_contents_1_ops | 437 | 16.704805 | 16.933638 | +0.228833 | +1 |
| regular_0_ops | 517 | 35.783366 | 35.783366 | +0.000000 | +0 |
| move_contents_0_ops | 516 | 42.441860 | 42.441860 | +0.000000 | +0 |
| ambiref_0_ops | 508 | 23.818898 | 23.818898 | +0.000000 | +0 |
| ambiref_4_ops | 434 | 37.557604 | 37.557604 | +0.000000 | +0 |
| regular_3_ops | 425 | 31.764706 | 31.764706 | +0.000000 | +0 |
| regular_1_ops | 409 | 17.359413 | 17.359413 | +0.000000 | +0 |
| move_contents_3_ops | 406 | 28.571429 | 28.571429 | +0.000000 | +0 |

### COMPS

| subtask | n | A acc | B acc | B-A acc | net items |
|---|---:|---:|---:|---:|---:|
| wugs_dist_in_between | 13896 | 64.968336 | 64.910766 | -0.057571 | -8 |
| wugs_dist_before | 13896 | 38.536269 | 38.565055 | +0.028785 | +4 |
| wugs | 13896 | 51.691134 | 51.676742 | -0.014393 | -2 |
| base | 49340 | 53.419133 | 53.429266 | +0.010134 | +5 |

### GlobalPIQA_parallel

| subtask | n | A acc | B acc | B-A acc | net items |
|---|---:|---:|---:|---:|---:|
| global_piqa_parallel | 103 | 30.097087 | 30.097087 | +0.000000 | +0 |

### GlobalPIQA_nonparallel

| subtask | n | A acc | B acc | B-A acc | net items |
|---|---:|---:|---:|---:|---:|
| global_piqa_nonparallel | 100 | 50.000000 | 50.000000 | +0.000000 | +0 |

## Interpretation note

Official BabyLM columns use macro averages over subtasks and SuperGLUE primary metrics; micro item flips diagnose which decisions changed but do not directly equal leaderboard movement. Treat deltas as measurement evidence, not as a causal mechanism without training provenance and controls.

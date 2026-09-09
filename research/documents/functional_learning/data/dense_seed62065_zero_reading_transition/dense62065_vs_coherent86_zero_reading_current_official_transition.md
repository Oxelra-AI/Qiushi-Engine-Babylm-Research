# earlier analysis official-coordinate transition comparison: dense_seed62065_zero_reading_current vs coherent86_official_zero_reading

## Score summary

Scores labelled `computed` are recomputed from prediction files using the official macro over subtasks or SuperGLUE primary metrics where applicable. `payload` is the saved evaluator score. Item flips below are micro counts and should not be read as official score movement when subtasks have unequal sizes.

| column | A payload | B payload | payload B-A | A computed | B computed | computed B-A | A payload-computed | B payload-computed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BLiMP | 68.510000 | 68.020000 | -0.490000 | 68.515970 | 68.028300 | -0.487670 | -0.005970 | -0.008300 |
| Supplement | 63.640000 | 63.090000 | -0.550000 | 63.635369 | 63.093405 | -0.541963 | +0.004631 | -0.003405 |
| EWoK | 50.020000 | 49.770000 | -0.250000 | 50.019616 | 49.769931 | -0.249685 | +0.000384 | +0.000069 |
| Entity | 28.320000 | 29.420000 | +1.100000 | 28.322219 | 29.415692 | +1.093473 | -0.002219 | +0.004308 |
| COMPS | 52.050000 | 52.150000 | +0.100000 | 52.047175 | 52.145457 | +0.098282 | +0.002825 | +0.004543 |
| GlobalPIQA_parallel | 29.130000 | 30.100000 | +0.970000 | 29.126214 | 30.097087 | +0.970874 | +0.003786 | +0.002913 |
| GlobalPIQA_nonparallel | 48.000000 | 50.000000 | +2.000000 | 48.000000 | 50.000000 | +2.000000 | +0.000000 | +0.000000 |
| GlobalPIQA | 38.565000 | 40.050000 | +1.485000 | 38.563107 | 40.048544 | +1.485437 | +0.001893 | +0.001456 |
| Reading | 8.165000 | 8.210000 | +0.045000 | 8.165000 | 8.210000 | +0.045000 | +0.000000 | +0.000000 |
| SuperGLUE |  | 68.562691 |  |  |  |  |  |  |
| cheap7_mean | 44.181429 | 44.387143 | +0.205714 | 44.181208 | 44.387333 | +0.206125 | +0.000221 | -0.000190 |
| projected_overall_with_aoa0_if_missing |  | 42.141410 |  |  |  |  |  |  |

## Prediction-level transitions

| column | n | official-macro A | official-macro B | official-macro B-A | micro A | micro B | micro B-A | B-only | A-only | net item |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BLiMP | 59875 | 68.515970 | 68.028300 | -0.487670 | 68.404175 | 67.911482 | -0.492693 | 785 | 1080 | -295 |
| Supplement | 5218 | 63.635369 | 63.093405 | -0.541963 | 73.974703 | 73.840552 | -0.134151 | 52 | 59 | -7 |
| EWoK | 7618 | 50.019616 | 49.769931 | -0.249685 | 49.986873 | 50.039380 | +0.052507 | 235 | 231 | +4 |
| Entity | 6780 | 28.322219 | 29.415692 | +1.093473 | 27.831858 | 28.466077 | +0.634218 | 222 | 179 | +43 |
| COMPS | 91028 | 52.047175 | 52.145457 | +0.098282 | 52.513512 | 52.645340 | +0.131828 | 3588 | 3468 | +120 |
| GlobalPIQA_parallel | 103 | 29.126214 | 30.097087 | +0.970874 | 29.126214 | 30.097087 | +0.970874 | 2 | 1 | +1 |
| GlobalPIQA_nonparallel | 100 | 48.000000 | 50.000000 | +2.000000 | 48.000000 | 50.000000 | +2.000000 | 2 | 0 | +2 |

## Largest subtask movements by official-macro component


### BLiMP

| subtask | n | A acc | B acc | B-A acc | net items |
|---|---:|---:|---:|---:|---:|
| matrix_question_npi_licensor_present | 929 | 41.657696 | 45.640474 | +3.982777 | +37 |
| principle_A_domain_1 | 914 | 77.024070 | 73.741794 | -3.282276 | -30 |
| distractor_agreement_relational_noun | 788 | 49.873096 | 46.700508 | -3.172589 | -25 |
| only_npi_licensor_present | 882 | 80.385488 | 77.437642 | -2.947846 | -26 |
| superlative_quantifiers_2 | 986 | 82.251521 | 79.310345 | -2.941176 | -29 |
| existential_there_quantifiers_2 | 911 | 40.944018 | 43.688255 | +2.744237 | +25 |
| determiner_noun_agreement_1 | 929 | 88.912809 | 86.544672 | -2.368138 | -22 |
| left_branch_island_simple_question | 951 | 51.209253 | 49.106204 | -2.103049 | -20 |
| wh_vs_that_no_gap | 861 | 88.734030 | 86.643438 | -2.090592 | -18 |
| left_branch_island_echo_question | 947 | 38.542767 | 36.536431 | -2.006336 | -19 |
| sentential_negation_npi_licensor_present | 919 | 99.455930 | 97.606094 | -1.849837 | -17 |
| coordinate_structure_constraint_object_extraction | 949 | 78.714436 | 76.923077 | -1.791359 | -17 |
| sentential_subject_island | 961 | 32.778356 | 34.547347 | +1.768991 | +17 |
| ellipsis_n_bar_2 | 828 | 96.376812 | 97.946860 | +1.570048 | +13 |
| drop_argument | 920 | 64.891304 | 63.369565 | -1.521739 | -14 |

### Supplement

| subtask | n | A acc | B acc | B-A acc | net items |
|---|---:|---:|---:|---:|---:|
| qa_congruence_easy | 64 | 70.312500 | 68.750000 | -1.562500 | -1 |
| hypernym | 842 | 49.881235 | 48.931116 | -0.950119 | -8 |
| qa_congruence_tricky | 165 | 49.696970 | 49.090909 | -0.606061 | -1 |
| turn_taking | 280 | 67.500000 | 67.857143 | +0.357143 | +1 |
| subject_aux_inversion | 3867 | 80.786139 | 80.837859 | +0.051720 | +2 |

### EWoK

| subtask | n | A acc | B acc | B-A acc | net items |
|---|---:|---:|---:|---:|---:|
| material-dynamics | 770 | 50.519481 | 54.935065 | +4.415584 | +34 |
| social-properties | 328 | 46.646341 | 44.817073 | -1.829268 | -6 |
| physical-dynamics | 120 | 52.500000 | 50.833333 | -1.666667 | -2 |
| physical-interactions | 556 | 48.561151 | 47.122302 | -1.438849 | -8 |
| social-interactions | 294 | 54.081633 | 53.401361 | -0.680272 | -2 |
| material-properties | 170 | 51.764706 | 51.176471 | -0.588235 | -1 |
| physical-relations | 818 | 49.877751 | 49.388753 | -0.488998 | -4 |
| quantitative-properties | 314 | 48.726115 | 48.407643 | -0.318471 | -1 |
| agent-properties | 2210 | 50.361991 | 50.135747 | -0.226244 | -5 |
| spatial-relations | 490 | 46.530612 | 46.734694 | +0.204082 | +1 |
| social-relations | 1548 | 50.645995 | 50.516796 | -0.129199 | -2 |

### Entity

| subtask | n | A acc | B acc | B-A acc | net items |
|---|---:|---:|---:|---:|---:|
| ambiref_5_ops | 123 | 31.707317 | 37.398374 | +5.691057 | +7 |
| regular_0_ops | 517 | 40.618956 | 35.783366 | -4.835590 | -25 |
| move_contents_4_ops | 353 | 31.161473 | 35.410765 | +4.249292 | +15 |
| move_contents_2_ops | 399 | 21.553885 | 25.313283 | +3.759398 | +15 |
| move_contents_0_ops | 516 | 45.542636 | 42.441860 | -3.100775 | -16 |
| ambiref_4_ops | 434 | 34.562212 | 37.557604 | +2.995392 | +13 |
| regular_2_ops | 405 | 21.975309 | 24.691358 | +2.716049 | +11 |
| regular_4_ops | 388 | 27.577320 | 30.154639 | +2.577320 | +10 |
| move_contents_3_ops | 406 | 26.600985 | 28.571429 | +1.970443 | +8 |
| ambiref_3_ops | 409 | 26.650367 | 28.606357 | +1.955990 | +8 |
| move_contents_5_ops | 116 | 43.103448 | 44.827586 | +1.724138 | +2 |
| regular_1_ops | 409 | 15.647922 | 17.359413 | +1.711491 | +7 |
| ambiref_0_ops | 508 | 25.196850 | 23.818898 | -1.377953 | -7 |
| ambiref_1_ops | 428 | 14.953271 | 13.785047 | -1.168224 | -5 |
| regular_5_ops | 94 | 29.787234 | 30.851064 | +1.063830 | +1 |

### COMPS

| subtask | n | A acc | B acc | B-A acc | net items |
|---|---:|---:|---:|---:|---:|
| wugs_dist_before | 13896 | 37.478411 | 38.565055 | +1.086644 | +151 |
| wugs_dist_in_between | 13896 | 65.443293 | 64.910766 | -0.532527 | -74 |
| wugs | 13896 | 52.022165 | 51.676742 | -0.345423 | -48 |
| base | 49340 | 53.244832 | 53.429266 | +0.184435 | +91 |

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

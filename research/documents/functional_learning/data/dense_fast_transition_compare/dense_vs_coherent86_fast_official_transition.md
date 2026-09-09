# earlier analysis official-coordinate transition comparison: dense_focus_seed62064_u0080_fast vs coherent86_alpha075_fast

## Score summary

Scores labelled `computed` are recomputed from prediction files using the official macro over subtasks or SuperGLUE primary metrics where applicable. `payload` is the saved evaluator score. Item flips below are micro counts and should not be read as official score movement when subtasks have unequal sizes.

| column | A payload | B payload | payload B-A | A computed | B computed | computed B-A | A payload-computed | B payload-computed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BLiMP | 69.170000 | 68.620000 | -0.550000 | 69.186567 | 68.634328 | -0.552239 | -0.016567 | -0.014328 |
| Supplement | 66.400000 | 66.000000 | -0.400000 | 66.400000 | 66.000000 | -0.400000 | +0.000000 | +0.000000 |
| EWoK | 49.820000 | 50.180000 | +0.360000 | 49.818182 | 50.181818 | +0.363636 | +0.001818 | -0.001818 |
| Entity | 27.780000 | 28.440000 | +0.660000 | 27.777594 | 28.435894 | +0.658300 | +0.002406 | +0.004106 |
| COMPS | 52.050000 | 52.150000 | +0.100000 | 52.047175 | 52.153718 | +0.106543 | +0.002825 | -0.003718 |
| GlobalPIQA_parallel | 29.130000 | 30.100000 | +0.970000 | 29.126214 | 30.097087 | +0.970874 | +0.003786 | +0.002913 |
| GlobalPIQA_nonparallel | 48.000000 | 50.000000 | +2.000000 | 48.000000 | 50.000000 | +2.000000 | +0.000000 | +0.000000 |
| GlobalPIQA | 38.565000 | 40.050000 | +1.485000 | 38.563107 | 40.048544 | +1.485437 | +0.001893 | +0.001456 |
| Reading | 8.165000 | 8.220000 | +0.055000 | 8.165000 | 8.220000 | +0.055000 | +0.000000 | +0.000000 |
| cheap7_mean | 44.564286 | 44.808571 | +0.244286 | 44.565375 | 44.810615 | +0.245240 | -0.001089 | -0.002043 |

## Prediction-level transitions

| column | n | official-macro A | official-macro B | official-macro B-A | micro A | micro B | micro B-A | B-only | A-only | net item |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BLiMP | 13400 | 69.186567 | 68.634328 | -0.552239 | 69.186567 | 68.634328 | -0.552239 | 175 | 249 | -74 |
| Supplement | 250 | 66.400000 | 66.000000 | -0.400000 | 66.400000 | 66.000000 | -0.400000 | 2 | 3 | -1 |
| EWoK | 1100 | 49.818182 | 50.181818 | +0.363636 | 49.818182 | 50.181818 | +0.363636 | 32 | 28 | +4 |
| Entity | 2238 | 27.777594 | 28.435894 | +0.658300 | 28.150134 | 28.462913 | +0.312779 | 76 | 69 | +7 |
| COMPS | 91028 | 52.047175 | 52.153718 | +0.106543 | 52.513512 | 52.646438 | +0.132926 | 3530 | 3409 | +121 |
| GlobalPIQA_parallel | 103 | 29.126214 | 30.097087 | +0.970874 | 29.126214 | 30.097087 | +0.970874 | 2 | 1 | +1 |
| GlobalPIQA_nonparallel | 100 | 48.000000 | 50.000000 | +2.000000 | 48.000000 | 50.000000 | +2.000000 | 2 | 0 | +2 |

## Largest subtask movements by official-macro component


### BLiMP

| subtask | n | A acc | B acc | B-A acc | net items |
|---|---:|---:|---:|---:|---:|
| wh_questions_object_gap | 200 | 52.500000 | 58.000000 | +5.500000 | +11 |
| existential_there_quantifiers_2 | 200 | 46.000000 | 50.500000 | +4.500000 | +9 |
| superlative_quantifiers_2 | 200 | 82.500000 | 78.000000 | -4.500000 | -9 |
| left_branch_island_echo_question | 200 | 39.500000 | 35.500000 | -4.000000 | -8 |
| only_npi_licensor_present | 200 | 79.500000 | 75.500000 | -4.000000 | -8 |
| principle_A_domain_1 | 200 | 75.000000 | 71.000000 | -4.000000 | -8 |
| distractor_agreement_relational_noun | 200 | 48.500000 | 45.000000 | -3.500000 | -7 |
| transitive | 200 | 74.000000 | 71.000000 | -3.000000 | -6 |
| animate_subject_trans | 200 | 63.500000 | 61.000000 | -2.500000 | -5 |
| coordinate_structure_constraint_complex_left_branch | 200 | 36.500000 | 34.000000 | -2.500000 | -5 |
| determiner_noun_agreement_1 | 200 | 90.000000 | 87.500000 | -2.500000 | -5 |
| matrix_question_npi_licensor_present | 200 | 45.000000 | 47.500000 | +2.500000 | +5 |
| passive_1 | 200 | 75.000000 | 72.500000 | -2.500000 | -5 |
| inchoative | 200 | 43.000000 | 41.000000 | -2.000000 | -4 |
| left_branch_island_simple_question | 200 | 53.500000 | 51.500000 | -2.000000 | -4 |

### Supplement

| subtask | n | A acc | B acc | B-A acc | net items |
|---|---:|---:|---:|---:|---:|
| qa_congruence_easy | 50 | 76.000000 | 74.000000 | -2.000000 | -1 |
| qa_congruence_tricky | 50 | 52.000000 | 54.000000 | +2.000000 | +1 |
| subject_aux_inversion | 50 | 82.000000 | 80.000000 | -2.000000 | -1 |
| hypernym | 50 | 52.000000 | 52.000000 | +0.000000 | +0 |
| turn_taking | 50 | 70.000000 | 70.000000 | +0.000000 | +0 |

### EWoK

| subtask | n | A acc | B acc | B-A acc | net items |
|---|---:|---:|---:|---:|---:|
| material-dynamics | 100 | 58.000000 | 62.000000 | +4.000000 | +4 |
| spatial-relations | 100 | 40.000000 | 44.000000 | +4.000000 | +4 |
| agent-properties | 100 | 54.000000 | 52.000000 | -2.000000 | -2 |
| physical-dynamics | 100 | 51.000000 | 49.000000 | -2.000000 | -2 |
| social-relations | 100 | 51.000000 | 49.000000 | -2.000000 | -2 |
| material-properties | 100 | 49.000000 | 50.000000 | +1.000000 | +1 |
| physical-interactions | 100 | 52.000000 | 51.000000 | -1.000000 | -1 |
| physical-relations | 100 | 56.000000 | 57.000000 | +1.000000 | +1 |
| quantitative-properties | 100 | 46.000000 | 47.000000 | +1.000000 | +1 |
| social-interactions | 100 | 49.000000 | 49.000000 | +0.000000 | +0 |
| social-properties | 100 | 42.000000 | 42.000000 | +0.000000 | +0 |

### Entity

| subtask | n | A acc | B acc | B-A acc | net items |
|---|---:|---:|---:|---:|---:|
| regular_0_ops | 517 | 40.618956 | 35.783366 | -4.835590 | -25 |
| regular_4_ops | 388 | 27.577320 | 30.412371 | +2.835052 | +11 |
| regular_2_ops | 405 | 21.975309 | 24.444444 | +2.469136 | +10 |
| regular_1_ops | 409 | 15.647922 | 17.359413 | +1.711491 | +7 |
| regular_5_ops | 94 | 29.787234 | 30.851064 | +1.063830 | +1 |
| regular_3_ops | 425 | 31.058824 | 31.764706 | +0.705882 | +3 |

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

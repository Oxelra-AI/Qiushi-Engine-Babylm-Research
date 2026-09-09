# earlier analysis official-coordinate transition comparison: dense_focus_seed62065_u0080_fast vs dense_focus_seed62064_u0080_fast

## Score summary

Scores labelled `computed` are recomputed from prediction files using the official macro over subtasks or SuperGLUE primary metrics where applicable. `payload` is the saved evaluator score. Item flips below are micro counts and should not be read as official score movement when subtasks have unequal sizes.

| column | A payload | B payload | payload B-A | A computed | B computed | computed B-A | A payload-computed | B payload-computed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BLiMP | 68.620000 | 68.600000 | -0.020000 | 68.634328 | 68.611940 | -0.022388 | -0.014328 | -0.011940 |
| Supplement | 66.000000 | 66.000000 | +0.000000 | 66.000000 | 66.000000 | +0.000000 | +0.000000 | +0.000000 |
| EWoK | 50.180000 | 50.000000 | -0.180000 | 50.181818 | 50.000000 | -0.181818 | -0.001818 | +0.000000 |
| Entity | 28.440000 | 28.430000 | -0.010000 | 28.435894 | 28.434091 | -0.001803 | +0.004106 | -0.004091 |
| COMPS | 52.150000 | 52.150000 | +0.000000 | 52.153718 | 52.145457 | -0.008261 | -0.003718 | +0.004543 |
| GlobalPIQA_parallel | 30.100000 | 30.100000 | +0.000000 | 30.097087 | 30.097087 | +0.000000 | +0.002913 | +0.002913 |
| GlobalPIQA_nonparallel | 50.000000 | 50.000000 | +0.000000 | 50.000000 | 50.000000 | +0.000000 | +0.000000 | +0.000000 |
| GlobalPIQA | 40.050000 | 40.050000 | +0.000000 | 40.048544 | 40.048544 | +0.000000 | +0.001456 | +0.001456 |
| Reading | 8.220000 | 8.210000 | -0.010000 | 8.220000 | 8.210000 | -0.010000 | +0.000000 | +0.000000 |
| cheap7_mean | 44.808571 | 44.777143 | -0.031429 | 44.810615 | 44.778576 | -0.032039 | -0.002043 | -0.001433 |

## Prediction-level transitions

| column | n | official-macro A | official-macro B | official-macro B-A | micro A | micro B | micro B-A | B-only | A-only | net item |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BLiMP | 13400 | 68.634328 | 68.611940 | -0.022388 | 68.634328 | 68.611940 | -0.022388 | 8 | 11 | -3 |
| Supplement | 250 | 66.000000 | 66.000000 | +0.000000 | 66.000000 | 66.000000 | +0.000000 | 0 | 0 | +0 |
| EWoK | 1100 | 50.181818 | 50.000000 | -0.181818 | 50.181818 | 50.000000 | -0.181818 | 2 | 4 | -2 |
| Entity | 2238 | 28.435894 | 28.434091 | -0.001803 | 28.462913 | 28.462913 | +0.000000 | 1 | 1 | +0 |
| COMPS | 91028 | 52.153718 | 52.145457 | -0.008261 | 52.646438 | 52.645340 | -0.001099 | 125 | 126 | -1 |
| GlobalPIQA_parallel | 103 | 30.097087 | 30.097087 | +0.000000 | 30.097087 | 30.097087 | +0.000000 | 0 | 0 | +0 |
| GlobalPIQA_nonparallel | 100 | 50.000000 | 50.000000 | +0.000000 | 50.000000 | 50.000000 | +0.000000 | 0 | 0 | +0 |

## Largest subtask movements by official-macro component


### BLiMP

| subtask | n | A acc | B acc | B-A acc | net items |
|---|---:|---:|---:|---:|---:|
| coordinate_structure_constraint_object_extraction | 200 | 78.000000 | 77.000000 | -1.000000 | -2 |
| principle_A_domain_3 | 200 | 56.500000 | 55.500000 | -1.000000 | -2 |
| wh_questions_object_gap | 200 | 58.000000 | 57.000000 | -1.000000 | -2 |
| coordinate_structure_constraint_complex_left_branch | 200 | 34.000000 | 34.500000 | +0.500000 | +1 |
| determiner_noun_agreement_irregular_1 | 200 | 77.000000 | 76.500000 | -0.500000 | -1 |
| existential_there_subject_raising | 200 | 79.000000 | 78.500000 | -0.500000 | -1 |
| expletive_it_object_raising | 200 | 67.500000 | 67.000000 | -0.500000 | -1 |
| npi_present_1 | 200 | 30.500000 | 31.000000 | +0.500000 | +1 |
| only_npi_scope | 200 | 71.500000 | 72.000000 | +0.500000 | +1 |
| passive_1 | 200 | 72.500000 | 73.000000 | +0.500000 | +1 |
| passive_2 | 200 | 75.500000 | 75.000000 | -0.500000 | -1 |
| principle_A_case_2 | 200 | 82.500000 | 83.000000 | +0.500000 | +1 |
| principle_A_domain_1 | 200 | 71.000000 | 71.500000 | +0.500000 | +1 |
| wh_questions_subject_gap | 200 | 76.500000 | 77.000000 | +0.500000 | +1 |
| wh_questions_subject_gap_long_distance | 200 | 93.500000 | 94.000000 | +0.500000 | +1 |

### Supplement

| subtask | n | A acc | B acc | B-A acc | net items |
|---|---:|---:|---:|---:|---:|
| hypernym | 50 | 52.000000 | 52.000000 | +0.000000 | +0 |
| qa_congruence_easy | 50 | 74.000000 | 74.000000 | +0.000000 | +0 |
| qa_congruence_tricky | 50 | 54.000000 | 54.000000 | +0.000000 | +0 |
| subject_aux_inversion | 50 | 80.000000 | 80.000000 | +0.000000 | +0 |
| turn_taking | 50 | 70.000000 | 70.000000 | +0.000000 | +0 |

### EWoK

| subtask | n | A acc | B acc | B-A acc | net items |
|---|---:|---:|---:|---:|---:|
| physical-interactions | 100 | 51.000000 | 52.000000 | +1.000000 | +1 |
| physical-relations | 100 | 57.000000 | 56.000000 | -1.000000 | -1 |
| social-interactions | 100 | 49.000000 | 48.000000 | -1.000000 | -1 |
| social-relations | 100 | 49.000000 | 48.000000 | -1.000000 | -1 |
| agent-properties | 100 | 52.000000 | 52.000000 | +0.000000 | +0 |
| material-dynamics | 100 | 62.000000 | 62.000000 | +0.000000 | +0 |
| material-properties | 100 | 50.000000 | 50.000000 | +0.000000 | +0 |
| physical-dynamics | 100 | 49.000000 | 49.000000 | +0.000000 | +0 |
| quantitative-properties | 100 | 47.000000 | 47.000000 | +0.000000 | +0 |
| social-properties | 100 | 42.000000 | 42.000000 | +0.000000 | +0 |
| spatial-relations | 100 | 44.000000 | 44.000000 | +0.000000 | +0 |

### Entity

| subtask | n | A acc | B acc | B-A acc | net items |
|---|---:|---:|---:|---:|---:|
| regular_4_ops | 388 | 30.412371 | 30.154639 | -0.257732 | -1 |
| regular_2_ops | 405 | 24.444444 | 24.691358 | +0.246914 | +1 |
| regular_0_ops | 517 | 35.783366 | 35.783366 | +0.000000 | +0 |
| regular_3_ops | 425 | 31.764706 | 31.764706 | +0.000000 | +0 |
| regular_1_ops | 409 | 17.359413 | 17.359413 | +0.000000 | +0 |
| regular_5_ops | 94 | 30.851064 | 30.851064 | +0.000000 | +0 |

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

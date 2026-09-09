# earlier analysis official-coordinate transition comparison: coherent86_official_ref_self vs coherent86_official_ref

## Score summary

Scores labelled `computed` are recomputed from prediction files using the official macro over subtasks or SuperGLUE primary metrics where applicable. `payload` is the saved evaluator score. Item flips below are micro counts and should not be read as official score movement when subtasks have unequal sizes.

| column | A payload | B payload | payload B-A | A computed | B computed | computed B-A | A payload-computed | B payload-computed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BLiMP | 68.510000 | 68.510000 | +0.000000 | 68.515970 | 68.515970 | +0.000000 | -0.005970 | -0.005970 |
| Supplement | 63.640000 | 63.640000 | +0.000000 | 63.635369 | 63.635369 | +0.000000 | +0.004631 | +0.004631 |
| EWoK | 50.020000 | 50.020000 | +0.000000 | 50.019616 | 50.019616 | +0.000000 | +0.000384 | +0.000384 |
| Entity | 28.320000 | 28.320000 | +0.000000 | 28.322219 | 28.322219 | +0.000000 | -0.002219 | -0.002219 |
| COMPS | 52.050000 | 52.050000 | +0.000000 | 52.047175 | 52.047175 | +0.000000 | +0.002825 | +0.002825 |
| GlobalPIQA_parallel | 29.130000 | 29.130000 | +0.000000 | 29.126214 | 29.126214 | +0.000000 | +0.003786 | +0.003786 |
| GlobalPIQA_nonparallel | 48.000000 | 48.000000 | +0.000000 | 48.000000 | 48.000000 | +0.000000 | +0.000000 | +0.000000 |
| GlobalPIQA | 38.565000 | 38.565000 | +0.000000 | 38.563107 | 38.563107 | +0.000000 | +0.001893 | +0.001893 |
| Reading | 8.165000 | 8.165000 | +0.000000 | 8.165000 | 8.165000 | +0.000000 | +0.000000 | +0.000000 |
| SuperGLUE | 69.819222 | 69.819222 | +0.000000 | 69.819222 | 69.819222 | +0.000000 | +0.000000 | +0.000000 |
| cheap7_mean | 44.181429 | 44.181429 | +0.000000 | 44.181208 | 44.181208 | +0.000000 | +0.000221 | +0.000221 |
| projected_overall_with_aoa0_if_missing | 42.121025 | 42.121025 | +0.000000 | 42.120853 | 42.120853 | +0.000000 | +0.000172 | +0.000172 |
| SuperGLUE/boolq |  |  |  | 68.562691 | 68.562691 | +0.000000 |  |  |
| SuperGLUE/mnli |  |  |  | 59.800326 | 59.800326 | +0.000000 |  |  |
| SuperGLUE/mrpc |  |  |  | 88.737201 | 88.737201 | +0.000000 |  |  |
| SuperGLUE/multirc |  |  |  | 67.574257 | 67.574257 | +0.000000 |  |  |
| SuperGLUE/qqp |  |  |  | 71.519959 | 71.519959 | +0.000000 |  |  |
| SuperGLUE/rte |  |  |  | 63.309353 | 63.309353 | +0.000000 |  |  |
| SuperGLUE/wsc |  |  |  | 69.230769 | 69.230769 | +0.000000 |  |  |

## Prediction-level transitions

| column | n | official-macro A | official-macro B | official-macro B-A | micro A | micro B | micro B-A | B-only | A-only | net item |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BLiMP | 59875 | 68.515970 | 68.515970 | +0.000000 | 68.404175 | 68.404175 | +0.000000 | 0 | 0 | +0 |
| Supplement | 5218 | 63.635369 | 63.635369 | +0.000000 | 73.974703 | 73.974703 | +0.000000 | 0 | 0 | +0 |
| EWoK | 7618 | 50.019616 | 50.019616 | +0.000000 | 49.986873 | 49.986873 | +0.000000 | 0 | 0 | +0 |
| Entity | 6780 | 28.322219 | 28.322219 | +0.000000 | 27.831858 | 27.831858 | +0.000000 | 0 | 0 | +0 |
| COMPS | 91028 | 52.047175 | 52.047175 | +0.000000 | 52.513512 | 52.513512 | +0.000000 | 0 | 0 | +0 |
| GlobalPIQA_parallel | 103 | 29.126214 | 29.126214 | +0.000000 | 29.126214 | 29.126214 | +0.000000 | 0 | 0 | +0 |
| GlobalPIQA_nonparallel | 100 | 48.000000 | 48.000000 | +0.000000 | 48.000000 | 48.000000 | +0.000000 | 0 | 0 | +0 |

## SuperGLUE task transitions

| task | metric | n | A primary | B primary | B-A primary | A acc | B acc | B-A acc | B-only | A-only | net item |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| boolq | accuracy | 1635 | 68.562691 | 68.562691 | +0.000000 | 68.562691 | 68.562691 | +0.000000 | 0 | 0 | +0 |
| mnli | accuracy | 4908 | 59.800326 | 59.800326 | +0.000000 | 59.800326 | 59.800326 | +0.000000 | 0 | 0 | +0 |
| mrpc | f1 | 204 | 88.737201 | 88.737201 | +0.000000 | 83.823529 | 83.823529 | +0.000000 | 0 | 0 | +0 |
| multirc | accuracy | 2424 | 67.574257 | 67.574257 | +0.000000 | 67.574257 | 67.574257 | +0.000000 | 0 | 0 | +0 |
| qqp | f1 | 20215 | 71.519959 | 71.519959 | +0.000000 | 77.976750 | 77.976750 | +0.000000 | 0 | 0 | +0 |
| rte | accuracy | 139 | 63.309353 | 63.309353 | +0.000000 | 63.309353 | 63.309353 | +0.000000 | 0 | 0 | +0 |
| wsc | accuracy | 52 | 69.230769 | 69.230769 | +0.000000 | 69.230769 | 69.230769 | +0.000000 | 0 | 0 | +0 |

## Largest subtask movements by official-macro component


### BLiMP

| subtask | n | A acc | B acc | B-A acc | net items |
|---|---:|---:|---:|---:|---:|
| superlative_quantifiers_2 | 986 | 82.251521 | 82.251521 | +0.000000 | +0 |
| superlative_quantifiers_1 | 979 | 96.731359 | 96.731359 | +0.000000 | +0 |
| anaphor_gender_agreement | 971 | 82.801236 | 82.801236 | +0.000000 | +0 |
| principle_A_reconstruction | 967 | 41.882110 | 41.882110 | +0.000000 | +0 |
| irregular_past_participle_adjectives | 961 | 91.571280 | 91.571280 | +0.000000 | +0 |
| sentential_subject_island | 961 | 32.778356 | 32.778356 | +0.000000 | +0 |
| wh_island | 960 | 57.604167 | 57.604167 | +0.000000 | +0 |
| left_branch_island_simple_question | 951 | 51.209253 | 51.209253 | +0.000000 | +0 |
| coordinate_structure_constraint_object_extraction | 949 | 78.714436 | 78.714436 | +0.000000 | +0 |
| tough_vs_raising_1 | 948 | 36.603376 | 36.603376 | +0.000000 | +0 |
| left_branch_island_echo_question | 947 | 38.542767 | 38.542767 | +0.000000 | +0 |
| principle_A_c_command | 946 | 50.739958 | 50.739958 | +0.000000 | +0 |
| regular_plural_subject_verb_agreement_2 | 945 | 72.804233 | 72.804233 | +0.000000 | +0 |
| irregular_past_participle_verbs | 942 | 79.830149 | 79.830149 | +0.000000 | +0 |
| determiner_noun_agreement_with_adj_2 | 941 | 94.686504 | 94.686504 | +0.000000 | +0 |

### Supplement

| subtask | n | A acc | B acc | B-A acc | net items |
|---|---:|---:|---:|---:|---:|
| subject_aux_inversion | 3867 | 80.786139 | 80.786139 | +0.000000 | +0 |
| hypernym | 842 | 49.881235 | 49.881235 | +0.000000 | +0 |
| turn_taking | 280 | 67.500000 | 67.500000 | +0.000000 | +0 |
| qa_congruence_tricky | 165 | 49.696970 | 49.696970 | +0.000000 | +0 |
| qa_congruence_easy | 64 | 70.312500 | 70.312500 | +0.000000 | +0 |

### EWoK

| subtask | n | A acc | B acc | B-A acc | net items |
|---|---:|---:|---:|---:|---:|
| agent-properties | 2210 | 50.361991 | 50.361991 | +0.000000 | +0 |
| social-relations | 1548 | 50.645995 | 50.645995 | +0.000000 | +0 |
| physical-relations | 818 | 49.877751 | 49.877751 | +0.000000 | +0 |
| material-dynamics | 770 | 50.519481 | 50.519481 | +0.000000 | +0 |
| physical-interactions | 556 | 48.561151 | 48.561151 | +0.000000 | +0 |
| spatial-relations | 490 | 46.530612 | 46.530612 | +0.000000 | +0 |
| social-properties | 328 | 46.646341 | 46.646341 | +0.000000 | +0 |
| quantitative-properties | 314 | 48.726115 | 48.726115 | +0.000000 | +0 |
| social-interactions | 294 | 54.081633 | 54.081633 | +0.000000 | +0 |
| material-properties | 170 | 51.764706 | 51.764706 | +0.000000 | +0 |
| physical-dynamics | 120 | 52.500000 | 52.500000 | +0.000000 | +0 |

### Entity

| subtask | n | A acc | B acc | B-A acc | net items |
|---|---:|---:|---:|---:|---:|
| regular_0_ops | 517 | 40.618956 | 40.618956 | +0.000000 | +0 |
| move_contents_0_ops | 516 | 45.542636 | 45.542636 | +0.000000 | +0 |
| ambiref_0_ops | 508 | 25.196850 | 25.196850 | +0.000000 | +0 |
| move_contents_1_ops | 437 | 17.162471 | 17.162471 | +0.000000 | +0 |
| ambiref_4_ops | 434 | 34.562212 | 34.562212 | +0.000000 | +0 |
| ambiref_1_ops | 428 | 14.953271 | 14.953271 | +0.000000 | +0 |
| regular_3_ops | 425 | 31.058824 | 31.058824 | +0.000000 | +0 |
| ambiref_2_ops | 413 | 24.939467 | 24.939467 | +0.000000 | +0 |
| ambiref_3_ops | 409 | 26.650367 | 26.650367 | +0.000000 | +0 |
| regular_1_ops | 409 | 15.647922 | 15.647922 | +0.000000 | +0 |
| move_contents_3_ops | 406 | 26.600985 | 26.600985 | +0.000000 | +0 |
| regular_2_ops | 405 | 21.975309 | 21.975309 | +0.000000 | +0 |
| move_contents_2_ops | 399 | 21.553885 | 21.553885 | +0.000000 | +0 |
| regular_4_ops | 388 | 27.577320 | 27.577320 | +0.000000 | +0 |
| move_contents_4_ops | 353 | 31.161473 | 31.161473 | +0.000000 | +0 |

### COMPS

| subtask | n | A acc | B acc | B-A acc | net items |
|---|---:|---:|---:|---:|---:|
| base | 49340 | 53.244832 | 53.244832 | +0.000000 | +0 |
| wugs | 13896 | 52.022165 | 52.022165 | +0.000000 | +0 |
| wugs_dist_before | 13896 | 37.478411 | 37.478411 | +0.000000 | +0 |
| wugs_dist_in_between | 13896 | 65.443293 | 65.443293 | +0.000000 | +0 |

### GlobalPIQA_parallel

| subtask | n | A acc | B acc | B-A acc | net items |
|---|---:|---:|---:|---:|---:|
| global_piqa_parallel | 103 | 29.126214 | 29.126214 | +0.000000 | +0 |

### GlobalPIQA_nonparallel

| subtask | n | A acc | B acc | B-A acc | net items |
|---|---:|---:|---:|---:|---:|
| global_piqa_nonparallel | 100 | 48.000000 | 48.000000 | +0.000000 | +0 |

## Interpretation note

Official BabyLM columns use macro averages over subtasks and SuperGLUE primary metrics; micro item flips diagnose which decisions changed but do not directly equal leaderboard movement. Treat deltas as measurement evidence, not as a causal mechanism without training provenance and controls.

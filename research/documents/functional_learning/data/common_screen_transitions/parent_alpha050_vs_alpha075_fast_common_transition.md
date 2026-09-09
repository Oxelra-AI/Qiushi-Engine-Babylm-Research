# earlier analysis common-screen transition: coherent86_alpha050 vs coherent86_alpha075

## Common-screen score summary

Computed scores are recomputed from saved predictions with the causal interface trajectory data coordinate; Reading is copied from the eval JSON. Payload scores are from the saved evaluator JSON.

| column | A payload | B payload | payload B-A | A computed | B computed | computed B-A |
|---|---:|---:|---:|---:|---:|---:|
| BLiMP | 69.170000 | 69.310000 | +0.140000 | 69.186567 | 69.320896 | +0.134328 |
| Supplement | 66.400000 | 66.000000 | -0.400000 | 66.400000 | 66.000000 | -0.400000 |
| EWoK | 49.820000 | 50.180000 | +0.360000 | 49.818182 | 50.181818 | +0.363636 |
| Entity | 27.780000 | 27.570000 | -0.210000 | 27.777594 | 27.566020 | -0.211574 |
| COMPS | 52.050000 | 52.110000 | +0.060000 | 52.047175 | 52.108975 | +0.061800 |
| GlobalPIQA_parallel | 29.130000 | 30.100000 | +0.970000 | 29.126214 | 30.097087 | +0.970874 |
| GlobalPIQA_nonparallel | 48.000000 | 48.000000 | +0.000000 | 48.000000 | 48.000000 | +0.000000 |
| GlobalPIQA_mean | 38.565000 | 39.050000 | +0.485000 | 38.563107 | 39.048544 | +0.485437 |
| Reading | 8.165000 | 8.165000 | +0.000000 | 8.165000 | 8.165000 | +0.000000 |
| equal_valid_mean | 44.564286 | 44.626429 | +0.062143 | 44.565375 | 44.627322 | +0.061947 |

## Prediction-level transitions

| column | n | A macro | B macro | B-A macro | A micro | B micro | B-A micro | B-only | A-only | net item |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BLiMP | 13400 | 69.186567 | 69.320896 | +0.134328 | 69.186567 | 69.320896 | +0.134328 | 61 | 43 | +18 |
| Supplement | 250 | 66.400000 | 66.000000 | -0.400000 | 66.400000 | 66.000000 | -0.400000 | 0 | 1 | -1 |
| EWoK | 1100 | 49.818182 | 50.181818 | +0.363636 | 49.818182 | 50.181818 | +0.363636 | 7 | 3 | +4 |
| Entity | 2238 | 27.777594 | 27.566020 | -0.211574 | 28.150134 | 28.060769 | -0.089366 | 5 | 7 | -2 |
| COMPS | 91028 | 52.047175 | 52.108975 | +0.061800 | 52.513512 | 52.566243 | +0.052731 | 578 | 530 | +48 |
| GlobalPIQA_parallel | 103 | 29.126214 | 30.097087 | +0.970874 | 29.126214 | 30.097087 | +0.970874 | 1 | 0 | +1 |
| GlobalPIQA_nonparallel | 100 | 48.000000 | 48.000000 | +0.000000 | 48.000000 | 48.000000 | +0.000000 | 0 | 0 | +0 |

## Largest subtask movements


### BLiMP

| subtask | n | A acc | B acc | B-A | net items |
|---|---:|---:|---:|---:|---:|
| npi_present_2 | 200 | 34.000000 | 37.500000 | +3.500000 | +7 |
| adjunct_island | 200 | 79.500000 | 78.000000 | -1.500000 | -3 |
| only_npi_licensor_present | 200 | 79.500000 | 81.000000 | +1.500000 | +3 |
| sentential_negation_npi_scope | 200 | 70.500000 | 72.000000 | +1.500000 | +3 |
| wh_vs_that_with_gap | 200 | 41.500000 | 43.000000 | +1.500000 | +3 |
| determiner_noun_agreement_with_adj_irregular_1 | 200 | 90.000000 | 91.000000 | +1.000000 | +2 |
| determiner_noun_agreement_with_adj_irregular_2 | 200 | 92.000000 | 91.000000 | -1.000000 | -2 |
| ellipsis_n_bar_1 | 200 | 77.000000 | 78.000000 | +1.000000 | +2 |
| only_npi_scope | 200 | 71.500000 | 72.500000 | +1.000000 | +2 |
| principle_A_case_2 | 200 | 80.500000 | 81.500000 | +1.000000 | +2 |
| principle_A_domain_3 | 200 | 56.000000 | 55.000000 | -1.000000 | -2 |
| sentential_subject_island | 200 | 37.000000 | 36.000000 | -1.000000 | -2 |
| superlative_quantifiers_2 | 200 | 82.500000 | 81.500000 | -1.000000 | -2 |
| tough_vs_raising_1 | 200 | 34.500000 | 33.500000 | -1.000000 | -2 |
| anaphor_gender_agreement | 200 | 84.000000 | 83.500000 | -0.500000 | -1 |

### Supplement

| subtask | n | A acc | B acc | B-A | net items |
|---|---:|---:|---:|---:|---:|
| qa_congruence_easy | 50 | 76.000000 | 74.000000 | -2.000000 | -1 |
| hypernym | 50 | 52.000000 | 52.000000 | +0.000000 | +0 |
| qa_congruence_tricky | 50 | 52.000000 | 52.000000 | +0.000000 | +0 |
| subject_aux_inversion | 50 | 82.000000 | 82.000000 | +0.000000 | +0 |
| turn_taking | 50 | 70.000000 | 70.000000 | +0.000000 | +0 |

### EWoK

| subtask | n | A acc | B acc | B-A | net items |
|---|---:|---:|---:|---:|---:|
| material-dynamics | 100 | 58.000000 | 59.000000 | +1.000000 | +1 |
| material-properties | 100 | 49.000000 | 48.000000 | -1.000000 | -1 |
| physical-interactions | 100 | 52.000000 | 51.000000 | -1.000000 | -1 |
| physical-relations | 100 | 56.000000 | 57.000000 | +1.000000 | +1 |
| quantitative-properties | 100 | 46.000000 | 47.000000 | +1.000000 | +1 |
| social-interactions | 100 | 49.000000 | 50.000000 | +1.000000 | +1 |
| social-properties | 100 | 42.000000 | 43.000000 | +1.000000 | +1 |
| social-relations | 100 | 51.000000 | 52.000000 | +1.000000 | +1 |
| agent-properties | 100 | 54.000000 | 54.000000 | +0.000000 | +0 |
| physical-dynamics | 100 | 51.000000 | 51.000000 | +0.000000 | +0 |
| spatial-relations | 100 | 40.000000 | 40.000000 | +0.000000 | +0 |

### Entity

| subtask | n | A acc | B acc | B-A | net items |
|---|---:|---:|---:|---:|---:|
| regular_5_ops | 94 | 29.787234 | 28.723404 | -1.063830 | -1 |
| regular_2_ops | 405 | 21.975309 | 22.716049 | +0.740741 | +3 |
| regular_1_ops | 409 | 15.647922 | 14.914425 | -0.733496 | -3 |
| regular_3_ops | 425 | 31.058824 | 30.588235 | -0.470588 | -2 |
| regular_4_ops | 388 | 27.577320 | 27.835052 | +0.257732 | +1 |
| regular_0_ops | 517 | 40.618956 | 40.618956 | +0.000000 | +0 |

### COMPS

| subtask | n | A acc | B acc | B-A | net items |
|---|---:|---:|---:|---:|---:|
| wugs_dist_before | 13896 | 37.478411 | 37.658319 | +0.179908 | +25 |
| wugs_dist_in_between | 13896 | 65.443293 | 65.486471 | +0.043178 | +6 |
| base | 49340 | 53.244832 | 53.283340 | +0.038508 | +19 |
| wugs | 13896 | 52.022165 | 52.007772 | -0.014393 | -2 |

### GlobalPIQA_parallel

| subtask | n | A acc | B acc | B-A | net items |
|---|---:|---:|---:|---:|---:|
| global_piqa_parallel | 103 | 29.126214 | 30.097087 | +0.970874 | +1 |

### GlobalPIQA_nonparallel

| subtask | n | A acc | B acc | B-A | net items |
|---|---:|---:|---:|---:|---:|
| global_piqa_nonparallel | 100 | 48.000000 | 48.000000 | +0.000000 | +0 |

## Interpretation note

This is a fast common-screen analysis. The result separates readout-scale effects from same-scale child training effects only on the causal interface trajectory screen; complete official-style evaluation and training provenance are still needed before treating any movement as a BabyLM advance or a general learning principle.

# earlier analysis common-screen transition: pa87_alpha050 vs coherent86_alpha050

## Common-screen score summary

Computed scores are recomputed from saved predictions with the causal interface trajectory data coordinate; Reading is copied from the eval JSON. Payload scores are from the saved evaluator JSON.

| column | A payload | B payload | payload B-A | A computed | B computed | computed B-A |
|---|---:|---:|---:|---:|---:|---:|
| BLiMP | 69.310000 | 69.150000 | -0.160000 | 69.320896 | 69.156716 | -0.164179 |
| Supplement | 66.000000 | 67.200000 | +1.200000 | 66.000000 | 67.200000 | +1.200000 |
| EWoK | 50.180000 | 50.000000 | -0.180000 | 50.181818 | 50.000000 | -0.181818 |
| Entity | 27.570000 | 27.580000 | +0.010000 | 27.566020 | 27.581693 | +0.015672 |
| COMPS | 52.110000 | 52.150000 | +0.040000 | 52.108975 | 52.151595 | +0.042620 |
| GlobalPIQA_parallel | 30.100000 | 29.130000 | -0.970000 | 30.097087 | 29.126214 | -0.970874 |
| GlobalPIQA_nonparallel | 48.000000 | 47.000000 | -1.000000 | 48.000000 | 47.000000 | -1.000000 |
| GlobalPIQA_mean | 39.050000 | 38.065000 | -0.985000 | 39.048544 | 38.063107 | -0.985437 |
| Reading | 8.165000 | 8.215000 | +0.050000 | 8.165000 | 8.215000 | +0.050000 |
| equal_valid_mean | 44.626429 | 44.622857 | -0.003571 | 44.627322 | 44.624016 | -0.003306 |

## Prediction-level transitions

| column | n | A macro | B macro | B-A macro | A micro | B micro | B-A micro | B-only | A-only | net item |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BLiMP | 13400 | 69.320896 | 69.156716 | -0.164179 | 69.320896 | 69.156716 | -0.164179 | 59 | 81 | -22 |
| Supplement | 250 | 66.000000 | 67.200000 | +1.200000 | 66.000000 | 67.200000 | +1.200000 | 3 | 0 | +3 |
| EWoK | 1100 | 50.181818 | 50.000000 | -0.181818 | 50.181818 | 50.000000 | -0.181818 | 14 | 16 | -2 |
| Entity | 2238 | 27.566020 | 27.581693 | +0.015672 | 28.060769 | 28.105451 | +0.044683 | 18 | 17 | +1 |
| COMPS | 91028 | 52.108975 | 52.151595 | +0.042620 | 52.566243 | 52.597003 | +0.030760 | 1049 | 1021 | +28 |
| GlobalPIQA_parallel | 103 | 30.097087 | 29.126214 | -0.970874 | 30.097087 | 29.126214 | -0.970874 | 0 | 1 | -1 |
| GlobalPIQA_nonparallel | 100 | 48.000000 | 47.000000 | -1.000000 | 48.000000 | 47.000000 | -1.000000 | 1 | 2 | -1 |

## Largest subtask movements


### BLiMP

| subtask | n | A acc | B acc | B-A | net items |
|---|---:|---:|---:|---:|---:|
| only_npi_licensor_present | 200 | 81.000000 | 79.000000 | -2.000000 | -4 |
| adjunct_island | 200 | 78.000000 | 79.500000 | +1.500000 | +3 |
| distractor_agreement_relative_clause | 200 | 33.000000 | 31.500000 | -1.500000 | -3 |
| only_npi_scope | 200 | 72.500000 | 71.000000 | -1.500000 | -3 |
| passive_1 | 200 | 75.500000 | 74.000000 | -1.500000 | -3 |
| tough_vs_raising_1 | 200 | 33.500000 | 35.000000 | +1.500000 | +3 |
| wh_island | 200 | 61.000000 | 59.500000 | -1.500000 | -3 |
| causative | 200 | 54.000000 | 53.000000 | -1.000000 | -2 |
| determiner_noun_agreement_with_adj_irregular_2 | 200 | 91.000000 | 92.000000 | +1.000000 | +2 |
| distractor_agreement_relational_noun | 200 | 48.500000 | 49.500000 | +1.000000 | +2 |
| existential_there_object_raising | 200 | 75.500000 | 74.500000 | -1.000000 | -2 |
| existential_there_quantifiers_2 | 200 | 45.500000 | 46.500000 | +1.000000 | +2 |
| irregular_past_participle_verbs | 200 | 79.500000 | 78.500000 | -1.000000 | -2 |
| passive_2 | 200 | 75.500000 | 74.500000 | -1.000000 | -2 |
| principle_A_c_command | 200 | 48.000000 | 49.000000 | +1.000000 | +2 |

### Supplement

| subtask | n | A acc | B acc | B-A | net items |
|---|---:|---:|---:|---:|---:|
| hypernym | 50 | 52.000000 | 54.000000 | +2.000000 | +1 |
| qa_congruence_easy | 50 | 74.000000 | 76.000000 | +2.000000 | +1 |
| qa_congruence_tricky | 50 | 52.000000 | 54.000000 | +2.000000 | +1 |
| subject_aux_inversion | 50 | 82.000000 | 82.000000 | +0.000000 | +0 |
| turn_taking | 50 | 70.000000 | 70.000000 | +0.000000 | +0 |

### EWoK

| subtask | n | A acc | B acc | B-A | net items |
|---|---:|---:|---:|---:|---:|
| social-relations | 100 | 52.000000 | 48.000000 | -4.000000 | -4 |
| quantitative-properties | 100 | 47.000000 | 50.000000 | +3.000000 | +3 |
| material-properties | 100 | 48.000000 | 50.000000 | +2.000000 | +2 |
| physical-relations | 100 | 57.000000 | 55.000000 | -2.000000 | -2 |
| social-interactions | 100 | 50.000000 | 48.000000 | -2.000000 | -2 |
| material-dynamics | 100 | 59.000000 | 60.000000 | +1.000000 | +1 |
| social-properties | 100 | 43.000000 | 42.000000 | -1.000000 | -1 |
| spatial-relations | 100 | 40.000000 | 41.000000 | +1.000000 | +1 |
| agent-properties | 100 | 54.000000 | 54.000000 | +0.000000 | +0 |
| physical-dynamics | 100 | 51.000000 | 51.000000 | +0.000000 | +0 |
| physical-interactions | 100 | 51.000000 | 51.000000 | +0.000000 | +0 |

### Entity

| subtask | n | A acc | B acc | B-A | net items |
|---|---:|---:|---:|---:|---:|
| regular_0_ops | 517 | 40.618956 | 41.199226 | +0.580271 | +3 |
| regular_1_ops | 409 | 14.914425 | 15.403423 | +0.488998 | +2 |
| regular_3_ops | 425 | 30.588235 | 30.117647 | -0.470588 | -2 |
| regular_4_ops | 388 | 27.835052 | 27.577320 | -0.257732 | -1 |
| regular_2_ops | 405 | 22.716049 | 22.469136 | -0.246914 | -1 |
| regular_5_ops | 94 | 28.723404 | 28.723404 | +0.000000 | +0 |

### COMPS

| subtask | n | A acc | B acc | B-A | net items |
|---|---:|---:|---:|---:|---:|
| wugs_dist_before | 13896 | 37.658319 | 37.938975 | +0.280656 | +39 |
| wugs_dist_in_between | 13896 | 65.486471 | 65.371330 | -0.115141 | -16 |
| base | 49340 | 53.283340 | 53.295501 | +0.012161 | +6 |
| wugs | 13896 | 52.007772 | 52.000576 | -0.007196 | -1 |

### GlobalPIQA_parallel

| subtask | n | A acc | B acc | B-A | net items |
|---|---:|---:|---:|---:|---:|
| global_piqa_parallel | 103 | 30.097087 | 29.126214 | -0.970874 | -1 |

### GlobalPIQA_nonparallel

| subtask | n | A acc | B acc | B-A | net items |
|---|---:|---:|---:|---:|---:|
| global_piqa_nonparallel | 100 | 48.000000 | 47.000000 | -1.000000 | -1 |

## Interpretation note

This is a fast common-screen analysis. The result separates readout-scale effects from same-scale child training effects only on the causal interface trajectory screen; complete official-style evaluation and training provenance are still needed before treating any movement as a BabyLM advance or a general learning principle.

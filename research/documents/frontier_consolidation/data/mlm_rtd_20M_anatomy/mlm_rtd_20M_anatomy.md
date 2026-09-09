# mlm rtd gdes 20m screen plan MLM+RTD-GDES 20M score anatomy

## Score movement
| column | baseline20M | MLM+RTD20M | delta |
|---|---:|---:|---:|
| BLiMP | 59.6900 | 60.1600 | +0.4700 |
| Supplement | 55.4500 | 56.9400 | +1.4900 |
| EWoK | 50.7300 | 50.0700 | -0.6600 |
| Entity | 18.6500 | 19.3500 | +0.7000 |
| COMPS | 50.2600 | 50.2400 | -0.0200 |
| GlobalPIQA_parallel | 20.3900 | 18.4500 | -1.9400 |
| GlobalPIQA_nonparallel | 48.0000 | 48.0000 | +0.0000 |
| GlobalPIQA | 34.1950 | 33.2250 | -0.9700 |
| Reading | 8.6700 | 8.4350 | -0.2350 |
| cheap7 | 39.6636 | 39.7743 | +0.1107 |

## MLM loss comparison against spatial repair route status legal baseline
- Matched steps: 506
- Mean MLM-loss delta all steps: -0.001946
- Mean delta first 50 steps: -0.006739
- Mean delta last 50 steps: +0.002963
- Selected steps:
  - earlier analysis: baseline 9.83754, MLM+RTD MLM 9.83754, delta -0.00000
  - legal tokenizer clean control trajectory design: baseline 6.81939, MLM+RTD MLM 6.79674, delta -0.02265
  - lamb update allocation closure: baseline 6.30556, MLM+RTD MLM 6.30900, delta +0.00344
  - frozen anchor fastpath disruption design: baseline 5.18163, MLM+RTD MLM 5.15869, delta -0.02294
  - seed43122 tail completion rationale: baseline 4.52951, MLM+RTD MLM 4.52596, delta -0.00355
  - commoncopy architecture interaction result: baseline 4.18824, MLM+RTD MLM 4.18938, delta +0.00114
  - fixed budget learning principle after review: baseline 4.06204, MLM+RTD MLM 4.05632, delta -0.00572
  - earlier analysis: baseline 3.88374, MLM+RTD MLM 3.87600, delta -0.00774
  - earlier analysis: baseline 3.96890, MLM+RTD MLM 3.96098, delta -0.00792
  - earlier analysis: baseline 3.89574, MLM+RTD MLM 3.88960, delta -0.00614
  - earlier analysis: baseline 3.74062, MLM+RTD MLM 3.74464, delta +0.00402
  - earlier analysis: baseline 3.75558, MLM+RTD MLM 3.81046, delta +0.05488

## Largest fine-grained movements
### Supplement
#### UID ACCURACY
| key | baseline | MLM+RTD | delta |
|---|---:|---:|---:|
| subject_aux_inversion | 77.06 | 73.96 | -3.10 |
| turn_taking | 62.14 | 63.21 | +1.07 |
| hypernym | 49.17 | 50.95 | +1.78 |
| qa_congruence_tricky | 35.76 | 38.79 | +3.03 |
| qa_congruence_easy | 53.12 | 57.81 | +4.69 |

#### FIELD ACCURACY
| key | baseline | MLM+RTD | delta |
|---|---:|---:|---:|
| supplement | 70.16 | 68.36 | -1.80 |

#### LINGUISTICS_TERM ACCURACY
| key | baseline | MLM+RTD | delta |
|---|---:|---:|---:|
| supplement | 70.16 | 68.36 | -1.80 |

### EWoK
#### UID ACCURACY
| key | baseline | MLM+RTD | delta |
|---|---:|---:|---:|
| social-properties | 48.78 | 40.85 | -7.93 |
| material-dynamics | 55.58 | 48.31 | -7.27 |
| physical-dynamics | 56.67 | 51.67 | -5.00 |
| social-interactions | 60.20 | 55.78 | -4.42 |
| physical-relations | 51.10 | 49.39 | -1.71 |
| agent-properties | 49.77 | 49.28 | -0.49 |
| physical-interactions | 48.92 | 49.28 | +0.36 |
| spatial-relations | 45.10 | 47.14 | +2.04 |
| social-relations | 48.26 | 50.58 | +2.32 |
| quantitative-properties | 51.27 | 57.32 | +6.05 |
| material-properties | 42.35 | 51.18 | +8.83 |

#### CONTEXT_CONTRAST ACCURACY
| key | baseline | MLM+RTD | delta |
|---|---:|---:|---:|
| negation | 54.21 | 43.68 | -10.53 |
| material | 54.29 | 48.57 | -5.72 |
| antonym | 49.67 | 49.50 | -0.17 |
| game | 55.00 | 55.00 | +0.00 |
| variable swap | 49.73 | 50.00 | +0.27 |
| other | 47.55 | 50.41 | +2.86 |
| number | 53.75 | 60.00 | +6.25 |
| variable_swap | 46.67 | 53.33 | +6.66 |
| active-passive | 43.33 | 50.00 | +6.67 |

#### CONTEXT_TYPE ACCURACY
| key | baseline | MLM+RTD | delta |
|---|---:|---:|---:|
| indirect | 49.92 | 49.31 | -0.61 |
| direct | 50.66 | 50.14 | -0.52 |

#### TARGET_CONTRAST ACCURACY
| key | baseline | MLM+RTD | delta |
|---|---:|---:|---:|
| concept swap | 50.55 | 49.47 | -1.08 |
| variable swap | 49.62 | 49.86 | +0.24 |

### Entity
#### UID ACCURACY
| key | baseline | MLM+RTD | delta |
|---|---:|---:|---:|
| ambiref_5_ops | 22.76 | 19.51 | -3.25 |
| regular_2_ops | 18.27 | 16.05 | -2.22 |
| move_contents_5_ops | 19.83 | 18.97 | -0.86 |
| ambiref_3_ops | 18.83 | 18.09 | -0.74 |
| ambiref | 18.81 | 18.49 | -0.32 |
| move_contents | 16.87 | 18.31 | +1.44 |
| regular_1_ops | 14.43 | 15.89 | +1.46 |
| regular_5_ops | 35.11 | 39.36 | +4.25 |
| move_contents_1_ops | 18.99 | 23.34 | +4.35 |
| move_contents_0_ops | 15.12 | 19.96 | +4.84 |

### BLiMP
#### UID ACCURACY
| key | baseline | MLM+RTD | delta |
|---|---:|---:|---:|
| only_npi_licensor_present | 90.70 | 67.35 | -23.35 |
| only_npi_scope | 56.63 | 38.47 | -18.16 |
| existential_there_quantifiers_2 | 40.07 | 30.41 | -9.66 |
| irregular_past_participle_verbs | 85.99 | 81.85 | -4.14 |
| matrix_question_npi_licensor_present | 10.55 | 6.46 | -4.09 |
| principle_A_domain_2 | 49.07 | 56.50 | +7.43 |
| principle_A_reconstruction | 27.30 | 36.30 | +9.00 |
| sentential_negation_npi_scope | 51.66 | 60.73 | +9.07 |
| coordinate_structure_constraint_object_extraction | 58.69 | 75.45 | +16.76 |
| principle_A_domain_1 | 55.36 | 72.21 | +16.85 |

#### FIELD ACCURACY
| key | baseline | MLM+RTD | delta |
|---|---:|---:|---:|
| semantics | 57.87 | 54.22 | -3.65 |
| morphology | 69.78 | 70.37 | +0.59 |
| syntax | 53.90 | 54.87 | +0.97 |
| syntax/semantics | 58.22 | 60.65 | +2.43 |

#### LINGUISTICS_TERM ACCURACY
| key | baseline | MLM+RTD | delta |
|---|---:|---:|---:|
| npi_licensing | 49.21 | 44.63 | -4.58 |
| quantifiers | 70.42 | 68.02 | -2.40 |
| determiner_noun_agreement | 80.86 | 79.48 | -1.38 |
| s-selection | 67.11 | 66.61 | -0.50 |
| filler_gap_dependency | 57.97 | 58.23 | +0.26 |
| irregular_forms | 82.76 | 83.87 | +1.11 |
| anaphor_agreement | 65.19 | 66.88 | +1.69 |
| island_effects | 41.88 | 44.36 | +2.48 |
| subject_verb_agreement | 52.20 | 54.78 | +2.58 |
| binding | 58.59 | 64.21 | +5.62 |


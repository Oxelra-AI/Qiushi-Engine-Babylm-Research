# shuffle control audit conclusion dual-view sentinel comparison — validation_existing_20m_routes

spatial repair route status 20M reference cheap7: `39.663571`

## live_adapter vs reference_20M

| Column | base | candidate | delta |
|---|---:|---:|---:|
| BLiMP | 59.6900 | 60.3800 | +0.6900 |
| Supplement | 55.4500 | 56.2300 | +0.7800 |
| EWoK | 50.7300 | 49.4900 | -1.2400 |
| Entity | 18.6500 | 18.6500 | +0.0000 |
| COMPS | 50.2600 | 50.4400 | +0.1800 |
| GlobalPIQA | 34.1950 | 32.2250 | -1.9700 |
| Reading | 8.6700 | 8.3100 | -0.3600 |
| **cheap7** | **39.663571** | **39.389286** | **-0.274286** |

### Training/accounting
- base: `{'run_dir': 'experiments/archive/frontier_consolidation/training/runs/complianttok_reinvest_seed43022_r2', 'scientific_metrics': 'experiments/archive/frontier_consolidation/training/runs/complianttok_reinvest_seed43022_r2/scientific_metrics.json', 'mode': None, 'updates': 2529, 'total_charged_words': 100000000, 'total_main_word_exposure': None, 'total_aux_word_exposure': None, 'schedule_total': None, 'final_loss': 2.5525617599487305, 'mean_aux_loss': None, 'aux_loss_batches': None}`
- candidate: `{'run_dir': 'experiments/archive/frontier_consolidation/training/runs/adapter128_live_h100M20M_seed43022', 'scientific_metrics': 'experiments/archive/frontier_consolidation/training/runs/adapter128_live_h100M20M_seed43022/scientific_metrics.json', 'mode': None, 'updates': 506, 'total_charged_words': 20008711, 'total_main_word_exposure': None, 'total_aux_word_exposure': None, 'schedule_total': None, 'final_loss': 3.756544828414917, 'mean_aux_loss': None, 'aux_loss_batches': None}`

### Fragile groups
#### Supplement
| group | n | base item % | cand item % | net gain-loss | gain | loss |
|---|---:|---:|---:|---:|---:|---:|
| subject_aux_inversion | 3867 | 77.06 | 77.99 | +36 | 229 | 193 |
| qa_congruence_tricky | 165 | 35.76 | 35.15 | -1 | 6 | 7 |
| qa_congruence_easy | 64 | 53.12 | 54.69 | +1 | 4 | 3 |

#### EWoK
| group | n | base item % | cand item % | net gain-loss | gain | loss |
|---|---:|---:|---:|---:|---:|---:|
| material-dynamics | 770 | 55.58 | 50.65 | -38 | 98 | 136 |
| material-properties | 170 | 42.35 | 45.29 | +5 | 35 | 30 |
| spatial-relations | 490 | 45.10 | 47.35 | +11 | 102 | 91 |
| quantitative-properties | 314 | 51.27 | 50.96 | -1 | 55 | 56 |
| physical-dynamics | 120 | 56.67 | 43.33 | -16 | 11 | 27 |
| physical-interactions | 556 | 48.92 | 51.80 | +16 | 68 | 52 |
| social-relations | 1548 | 48.26 | 50.78 | +39 | 278 | 239 |

#### Entity
| group | n | base item % | cand item % | net gain-loss | gain | loss |
|---|---:|---:|---:|---:|---:|---:|
| regular_5_ops | 94 | 35.11 | 34.04 | -1 | 7 | 8 |
| regular_4_ops | 388 | 17.01 | 19.85 | +11 | 24 | 13 |
| regular_3_ops | 425 | 20.71 | 20.94 | +1 | 16 | 15 |
| ambiref_5_ops | 123 | 22.76 | 19.51 | -4 | 5 | 9 |
| ambiref_4_ops | 434 | 19.12 | 17.28 | -8 | 11 | 19 |
| move_contents_5_ops | 116 | 19.83 | 17.24 | -3 | 3 | 6 |
| move_contents_4_ops | 353 | 12.75 | 12.75 | +0 | 18 | 18 |

#### COMPS
| group | n | base item % | cand item % | net gain-loss | gain | loss |
|---|---:|---:|---:|---:|---:|---:|
| base | 49340 | 51.17 | 51.56 | +195 | 6750 | 6555 |
| wugs | 13896 | 49.63 | 50.61 | +136 | 2224 | 2088 |
| wugs_dist_before | 13896 | 49.83 | 49.77 | -8 | 2087 | 2095 |
| wugs_dist_in_between | 13896 | 50.41 | 49.83 | -80 | 2043 | 2123 |

#### GlobalPIQA
| group | n | base item % | cand item % | net gain-loss | gain | loss |
|---|---:|---:|---:|---:|---:|---:|
| GlobalPIQA_parallel | 103 | 20.39 | 18.45 | -2 | 2 | 4 |
| GlobalPIQA_nonparallel | 100 | 48.00 | 46.00 | -2 | 5 | 7 |

### Best/worst item-net groups
- BLiMP: best principle_A_domain_1 +133/914, sentential_subject_island +106/961, only_npi_scope +89/837, sentential_negation_npi_scope +82/871; worst wh_island -200/960, existential_there_quantifiers_2 -95/911, complex_NP_island -53/846, wh_questions_object_gap -47/859
- Supplement: best qa_congruence_easy +1/64, hypernym +11/842, subject_aux_inversion +36/3867, turn_taking +2/280; worst qa_congruence_tricky -1/165, turn_taking +2/280, subject_aux_inversion +36/3867, hypernym +11/842
- EWoK: best material-properties +5/170, physical-interactions +16/556, social-relations +39/1548, spatial-relations +11/490; worst physical-dynamics -16/120, social-interactions -22/294, material-dynamics -38/770, quantitative-properties -1/314
- Entity: best regular_4_ops +11/388, move_contents_1_ops +10/437, move_contents_2_ops +8/399, move_contents_0_ops +8/516; worst ambiref_5_ops -4/123, move_contents_5_ops -3/116, ambiref_4_ops -8/434, regular_5_ops -1/94
- COMPS: best wugs +136/13896, base +195/49340, wugs_dist_before -8/13896, wugs_dist_in_between -80/13896; worst wugs_dist_in_between -80/13896, wugs_dist_before -8/13896, base +195/49340, wugs +136/13896
- GlobalPIQA: best GlobalPIQA_parallel -2/103, GlobalPIQA_nonparallel -2/100; worst GlobalPIQA_nonparallel -2/100, GlobalPIQA_parallel -2/103

## u256_20M vs reference_20M

| Column | base | candidate | delta |
|---|---:|---:|---:|
| BLiMP | 59.6900 | 60.5900 | +0.9000 |
| Supplement | 55.4500 | 57.8900 | +2.4400 |
| EWoK | 50.7300 | 50.2500 | -0.4800 |
| Entity | 18.6500 | 18.3700 | -0.2800 |
| COMPS | 50.2600 | 50.7900 | +0.5300 |
| GlobalPIQA | 34.1950 | 38.6500 | +4.4550 |
| Reading | 8.6700 | 7.5300 | -1.1400 |
| **cheap7** | **39.663571** | **40.581429** | **+0.917857** |

### Training/accounting
- base: `{'run_dir': 'experiments/archive/frontier_consolidation/training/runs/complianttok_reinvest_seed43022_r2', 'scientific_metrics': 'experiments/archive/frontier_consolidation/training/runs/complianttok_reinvest_seed43022_r2/scientific_metrics.json', 'mode': None, 'updates': 2529, 'total_charged_words': 100000000, 'total_main_word_exposure': None, 'total_aux_word_exposure': None, 'schedule_total': None, 'final_loss': 2.5525617599487305, 'mean_aux_loss': None, 'aux_loss_batches': None}`
- candidate: `{'run_dir': 'experiments/archive/frontier_consolidation/training/runs/eu_U256_legal16k_seed43022_20M', 'scientific_metrics': 'experiments/archive/frontier_consolidation/training/runs/eu_U256_legal16k_seed43022_20M/scientific_metrics.json', 'mode': None, 'updates': None, 'total_charged_words': 20000000, 'total_main_word_exposure': None, 'total_aux_word_exposure': None, 'schedule_total': None, 'final_loss': 3.718591442220565, 'mean_aux_loss': None, 'aux_loss_batches': None}`

### Fragile groups
#### Supplement
| group | n | base item % | cand item % | net gain-loss | gain | loss |
|---|---:|---:|---:|---:|---:|---:|
| subject_aux_inversion | 3867 | 77.06 | 79.91 | +110 | 346 | 236 |
| qa_congruence_tricky | 165 | 35.76 | 38.18 | +4 | 9 | 5 |
| qa_congruence_easy | 64 | 53.12 | 56.25 | +2 | 6 | 4 |

#### EWoK
| group | n | base item % | cand item % | net gain-loss | gain | loss |
|---|---:|---:|---:|---:|---:|---:|
| material-dynamics | 770 | 55.58 | 44.81 | -83 | 146 | 229 |
| material-properties | 170 | 42.35 | 52.35 | +17 | 44 | 27 |
| spatial-relations | 490 | 45.10 | 49.18 | +20 | 115 | 95 |
| quantitative-properties | 314 | 51.27 | 50.32 | -3 | 55 | 58 |
| physical-dynamics | 120 | 56.67 | 55.83 | -1 | 22 | 23 |
| physical-interactions | 556 | 48.92 | 50.90 | +11 | 100 | 89 |
| social-relations | 1548 | 48.26 | 49.68 | +22 | 306 | 284 |

#### Entity
| group | n | base item % | cand item % | net gain-loss | gain | loss |
|---|---:|---:|---:|---:|---:|---:|
| regular_5_ops | 94 | 35.11 | 26.60 | -8 | 4 | 12 |
| regular_4_ops | 388 | 17.01 | 17.01 | +0 | 32 | 32 |
| regular_3_ops | 425 | 20.71 | 19.06 | -7 | 30 | 37 |
| ambiref_5_ops | 123 | 22.76 | 20.33 | -3 | 10 | 13 |
| ambiref_4_ops | 434 | 19.12 | 18.66 | -2 | 37 | 39 |
| move_contents_5_ops | 116 | 19.83 | 20.69 | +1 | 11 | 10 |
| move_contents_4_ops | 353 | 12.75 | 15.58 | +10 | 27 | 17 |

#### COMPS
| group | n | base item % | cand item % | net gain-loss | gain | loss |
|---|---:|---:|---:|---:|---:|---:|
| base | 49340 | 51.17 | 51.69 | +258 | 8436 | 8178 |
| wugs | 13896 | 49.63 | 50.73 | +152 | 2755 | 2603 |
| wugs_dist_before | 13896 | 49.83 | 50.30 | +66 | 3222 | 3156 |
| wugs_dist_in_between | 13896 | 50.41 | 50.43 | +3 | 3194 | 3191 |

#### GlobalPIQA
| group | n | base item % | cand item % | net gain-loss | gain | loss |
|---|---:|---:|---:|---:|---:|---:|
| GlobalPIQA_parallel | 103 | 20.39 | 23.30 | +3 | 9 | 6 |
| GlobalPIQA_nonparallel | 100 | 48.00 | 54.00 | +6 | 12 | 6 |

### Best/worst item-net groups
- BLiMP: best principle_A_domain_1 +156/914, npi_present_1 +143/909, wh_island +149/960, adjunct_island +123/928; worst existential_there_quantifiers_2 -250/911, irregular_plural_subject_verb_agreement_2 -71/892, tough_vs_raising_2 -61/920, left_branch_island_simple_question -62/951
- Supplement: best turn_taking +9/280, qa_congruence_easy +2/64, subject_aux_inversion +110/3867, qa_congruence_tricky +4/165; worst hypernym +5/842, qa_congruence_tricky +4/165, subject_aux_inversion +110/3867, qa_congruence_easy +2/64
- EWoK: best material-properties +17/170, spatial-relations +20/490, physical-interactions +11/556, social-relations +22/1548; worst material-dynamics -83/770, social-interactions -12/294, social-properties -13/328, physical-relations -23/818
- Entity: best move_contents_4_ops +10/353, move_contents_0_ops +13/516, ambiref_1_ops +10/428, regular_1_ops +5/409; worst regular_5_ops -8/94, move_contents_2_ops -10/399, ambiref_5_ops -3/123, regular_3_ops -7/425
- COMPS: best wugs +152/13896, base +258/49340, wugs_dist_before +66/13896, wugs_dist_in_between +3/13896; worst wugs_dist_in_between +3/13896, wugs_dist_before +66/13896, base +258/49340, wugs +152/13896
- GlobalPIQA: best GlobalPIQA_nonparallel +6/100, GlobalPIQA_parallel +3/103; worst GlobalPIQA_parallel +3/103, GlobalPIQA_nonparallel +6/100

## u256_20M vs live_adapter

| Column | base | candidate | delta |
|---|---:|---:|---:|
| BLiMP | 60.3800 | 60.5900 | +0.2100 |
| Supplement | 56.2300 | 57.8900 | +1.6600 |
| EWoK | 49.4900 | 50.2500 | +0.7600 |
| Entity | 18.6500 | 18.3700 | -0.2800 |
| COMPS | 50.4400 | 50.7900 | +0.3500 |
| GlobalPIQA | 32.2250 | 38.6500 | +6.4250 |
| Reading | 8.3100 | 7.5300 | -0.7800 |
| **cheap7** | **39.389286** | **40.581429** | **+1.192143** |

### Training/accounting
- base: `{'run_dir': 'experiments/archive/frontier_consolidation/training/runs/adapter128_live_h100M20M_seed43022', 'scientific_metrics': 'experiments/archive/frontier_consolidation/training/runs/adapter128_live_h100M20M_seed43022/scientific_metrics.json', 'mode': None, 'updates': 506, 'total_charged_words': 20008711, 'total_main_word_exposure': None, 'total_aux_word_exposure': None, 'schedule_total': None, 'final_loss': 3.756544828414917, 'mean_aux_loss': None, 'aux_loss_batches': None}`
- candidate: `{'run_dir': 'experiments/archive/frontier_consolidation/training/runs/eu_U256_legal16k_seed43022_20M', 'scientific_metrics': 'experiments/archive/frontier_consolidation/training/runs/eu_U256_legal16k_seed43022_20M/scientific_metrics.json', 'mode': None, 'updates': None, 'total_charged_words': 20000000, 'total_main_word_exposure': None, 'total_aux_word_exposure': None, 'schedule_total': None, 'final_loss': 3.718591442220565, 'mean_aux_loss': None, 'aux_loss_batches': None}`

### Fragile groups
#### Supplement
| group | n | base item % | cand item % | net gain-loss | gain | loss |
|---|---:|---:|---:|---:|---:|---:|
| subject_aux_inversion | 3867 | 77.99 | 79.91 | +74 | 300 | 226 |
| qa_congruence_tricky | 165 | 35.15 | 38.18 | +5 | 11 | 6 |
| qa_congruence_easy | 64 | 54.69 | 56.25 | +1 | 5 | 4 |

#### EWoK
| group | n | base item % | cand item % | net gain-loss | gain | loss |
|---|---:|---:|---:|---:|---:|---:|
| material-dynamics | 770 | 50.65 | 44.81 | -45 | 121 | 166 |
| material-properties | 170 | 45.29 | 52.35 | +12 | 46 | 34 |
| spatial-relations | 490 | 47.35 | 49.18 | +9 | 81 | 72 |
| quantitative-properties | 314 | 50.96 | 50.32 | -2 | 71 | 73 |
| physical-dynamics | 120 | 43.33 | 55.83 | +15 | 29 | 14 |
| physical-interactions | 556 | 51.80 | 50.90 | -5 | 77 | 82 |
| social-relations | 1548 | 50.78 | 49.68 | -17 | 312 | 329 |

#### Entity
| group | n | base item % | cand item % | net gain-loss | gain | loss |
|---|---:|---:|---:|---:|---:|---:|
| regular_5_ops | 94 | 34.04 | 26.60 | -7 | 3 | 10 |
| regular_4_ops | 388 | 19.85 | 17.01 | -11 | 24 | 35 |
| regular_3_ops | 425 | 20.94 | 19.06 | -8 | 31 | 39 |
| ambiref_5_ops | 123 | 19.51 | 20.33 | +1 | 12 | 11 |
| ambiref_4_ops | 434 | 17.28 | 18.66 | +6 | 37 | 31 |
| move_contents_5_ops | 116 | 17.24 | 20.69 | +4 | 13 | 9 |
| move_contents_4_ops | 353 | 12.75 | 15.58 | +10 | 28 | 18 |

#### COMPS
| group | n | base item % | cand item % | net gain-loss | gain | loss |
|---|---:|---:|---:|---:|---:|---:|
| base | 49340 | 51.56 | 51.69 | +63 | 8641 | 8578 |
| wugs | 13896 | 50.61 | 50.73 | +16 | 2707 | 2691 |
| wugs_dist_before | 13896 | 49.77 | 50.30 | +74 | 2545 | 2471 |
| wugs_dist_in_between | 13896 | 49.83 | 50.43 | +83 | 2572 | 2489 |

#### GlobalPIQA
| group | n | base item % | cand item % | net gain-loss | gain | loss |
|---|---:|---:|---:|---:|---:|---:|
| GlobalPIQA_parallel | 103 | 18.45 | 23.30 | +5 | 8 | 3 |
| GlobalPIQA_nonparallel | 100 | 46.00 | 54.00 | +8 | 15 | 7 |

### Best/worst item-net groups
- BLiMP: best wh_island +349/960, adjunct_island +138/928, npi_present_2 +130/914, npi_present_1 +122/909; worst existential_there_quantifiers_2 -155/911, sentential_subject_island -156/961, principle_A_c_command -138/946, only_npi_licensor_present -115/882
- Supplement: best qa_congruence_tricky +5/165, turn_taking +7/280, subject_aux_inversion +74/3867, qa_congruence_easy +1/64; worst hypernym -6/842, qa_congruence_easy +1/64, subject_aux_inversion +74/3867, turn_taking +7/280
- EWoK: best physical-dynamics +15/120, material-properties +12/170, social-interactions +10/294, spatial-relations +9/490; worst material-dynamics -45/770, social-properties -15/328, physical-relations -32/818, social-relations -17/1548
- Entity: best move_contents_5_ops +4/116, move_contents_4_ops +10/353, ambiref_1_ops +9/428, regular_2_ops +7/405; worst regular_5_ops -7/94, move_contents_2_ops -18/399, regular_4_ops -11/388, move_contents_1_ops -12/437
- COMPS: best wugs_dist_in_between +83/13896, wugs_dist_before +74/13896, base +63/49340, wugs +16/13896; worst wugs +16/13896, base +63/49340, wugs_dist_before +74/13896, wugs_dist_in_between +83/13896
- GlobalPIQA: best GlobalPIQA_nonparallel +8/100, GlobalPIQA_parallel +5/103; worst GlobalPIQA_parallel +5/103, GlobalPIQA_nonparallel +8/100

JSON: `experiments/archive/frontier_consolidation/data/dualview_sentinel_compare/validation_existing_20m_routes.json`

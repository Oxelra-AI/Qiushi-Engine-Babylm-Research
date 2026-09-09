# shuffle control audit conclusion dual-view sentinel comparison — dualview_panel_complete

spatial repair route status 20M reference cheap7: `39.663571`

## aligned vs reference_20M

| Column | base | candidate | delta |
|---|---:|---:|---:|
| BLiMP | 59.6900 | 56.7300 | -2.9600 |
| Supplement | 55.4500 | 51.7600 | -3.6900 |
| EWoK | 50.7300 | 51.1100 | +0.3800 |
| Entity | 18.6500 | 17.9100 | -0.7400 |
| COMPS | 50.2600 | 50.2000 | -0.0600 |
| GlobalPIQA | 34.1950 | 36.1050 | +1.9100 |
| Reading | 8.6700 | 7.5250 | -1.1450 |
| **cheap7** | **39.663571** | **38.762857** | **-0.900714** |

### Training/accounting
- base: `{'run_dir': 'experiments/archive/frontier_consolidation/training/runs/complianttok_reinvest_seed43022_r2', 'scientific_metrics': 'experiments/archive/frontier_consolidation/training/runs/complianttok_reinvest_seed43022_r2/scientific_metrics.json', 'mode': None, 'updates': 2529, 'total_charged_words': 100000000, 'total_main_word_exposure': None, 'total_aux_word_exposure': None, 'schedule_total': None, 'final_loss': 2.5525617599487305, 'mean_aux_loss': None, 'aux_loss_batches': None}`
- candidate: `{'run_dir': 'experiments/archive/frontier_consolidation/training/runs/dualview_aligned_20M_seed43022', 'scientific_metrics': 'experiments/archive/frontier_consolidation/training/runs/dualview_aligned_20M_seed43022/scientific_metrics.json', 'mode': 'aligned', 'updates': 481, 'total_charged_words': 19988304, 'total_main_word_exposure': 19021584, 'total_aux_word_exposure': 966720, 'schedule_total': 2529, 'final_loss': 4.379467487335205, 'mean_aux_loss': 7.470588489233876, 'aux_loss_batches': 481}`

### Fragile groups
#### Supplement
| group | n | base item % | cand item % | net gain-loss | gain | loss |
|---|---:|---:|---:|---:|---:|---:|
| subject_aux_inversion | 3867 | 77.06 | 76.75 | -12 | 346 | 358 |
| qa_congruence_tricky | 165 | 35.76 | 32.73 | -5 | 9 | 14 |
| qa_congruence_easy | 64 | 53.12 | 43.75 | -6 | 0 | 6 |

#### EWoK
| group | n | base item % | cand item % | net gain-loss | gain | loss |
|---|---:|---:|---:|---:|---:|---:|
| material-dynamics | 770 | 55.58 | 51.43 | -32 | 182 | 214 |
| material-properties | 170 | 42.35 | 56.47 | +24 | 49 | 25 |
| spatial-relations | 490 | 45.10 | 50.00 | +24 | 154 | 130 |
| quantitative-properties | 314 | 51.27 | 52.87 | +5 | 67 | 62 |
| physical-dynamics | 120 | 56.67 | 45.00 | -14 | 23 | 37 |
| physical-interactions | 556 | 48.92 | 52.16 | +18 | 150 | 132 |
| social-relations | 1548 | 48.26 | 50.00 | +27 | 327 | 300 |

#### Entity
| group | n | base item % | cand item % | net gain-loss | gain | loss |
|---|---:|---:|---:|---:|---:|---:|
| regular_5_ops | 94 | 35.11 | 18.09 | -16 | 6 | 22 |
| regular_4_ops | 388 | 17.01 | 18.30 | +5 | 59 | 54 |
| regular_3_ops | 425 | 20.71 | 15.76 | -21 | 46 | 67 |
| ambiref_5_ops | 123 | 22.76 | 21.95 | -1 | 17 | 18 |
| ambiref_4_ops | 434 | 19.12 | 16.59 | -11 | 44 | 55 |
| move_contents_5_ops | 116 | 19.83 | 27.59 | +9 | 23 | 14 |
| move_contents_4_ops | 353 | 12.75 | 15.30 | +9 | 43 | 34 |

#### COMPS
| group | n | base item % | cand item % | net gain-loss | gain | loss |
|---|---:|---:|---:|---:|---:|---:|
| base | 49340 | 51.17 | 50.82 | -173 | 10234 | 10407 |
| wugs | 13896 | 49.63 | 49.43 | -28 | 3404 | 3432 |
| wugs_dist_before | 13896 | 49.83 | 50.21 | +53 | 2066 | 2013 |
| wugs_dist_in_between | 13896 | 50.41 | 50.35 | -9 | 2039 | 2048 |

#### GlobalPIQA
| group | n | base item % | cand item % | net gain-loss | gain | loss |
|---|---:|---:|---:|---:|---:|---:|
| GlobalPIQA_parallel | 103 | 20.39 | 26.21 | +6 | 11 | 5 |
| GlobalPIQA_nonparallel | 100 | 48.00 | 46.00 | -2 | 6 | 8 |

### Best/worst item-net groups
- BLiMP: best principle_A_domain_1 +245/914, left_branch_island_echo_question +225/947, only_npi_scope +146/837, wh_island +155/960; worst existential_there_quantifiers_2 -278/911, determiner_noun_agreement_with_adj_2 -280/941, determiner_noun_agreement_2 -222/931, determiner_noun_agreement_with_adj_irregular_1 -169/718
- Supplement: best hypernym +21/842, subject_aux_inversion -12/3867, qa_congruence_tricky -5/165, turn_taking -23/280; worst qa_congruence_easy -6/64, turn_taking -23/280, qa_congruence_tricky -5/165, subject_aux_inversion -12/3867
- EWoK: best material-properties +24/170, spatial-relations +24/490, physical-interactions +18/556, social-properties +10/328; worst physical-dynamics -14/120, social-interactions -24/294, material-dynamics -32/770, physical-relations -4/818
- Entity: best move_contents_5_ops +9/116, regular_1_ops +12/409, move_contents_4_ops +9/353, move_contents_1_ops +7/437; worst regular_5_ops -16/94, regular_3_ops -21/425, ambiref_0_ops -17/508, ambiref_4_ops -11/434
- COMPS: best wugs_dist_before +53/13896, wugs_dist_in_between -9/13896, wugs -28/13896, base -173/49340; worst base -173/49340, wugs -28/13896, wugs_dist_in_between -9/13896, wugs_dist_before +53/13896
- GlobalPIQA: best GlobalPIQA_parallel +6/103, GlobalPIQA_nonparallel -2/100; worst GlobalPIQA_nonparallel -2/100, GlobalPIQA_parallel +6/103

## shuffled vs reference_20M

| Column | base | candidate | delta |
|---|---:|---:|---:|
| BLiMP | 59.6900 | 56.5400 | -3.1500 |
| Supplement | 55.4500 | 52.8200 | -2.6300 |
| EWoK | 50.7300 | 51.3100 | +0.5800 |
| Entity | 18.6500 | 17.3500 | -1.3000 |
| COMPS | 50.2600 | 49.8000 | -0.4600 |
| GlobalPIQA | 34.1950 | 34.6200 | +0.4250 |
| Reading | 8.6700 | 7.3500 | -1.3200 |
| **cheap7** | **39.663571** | **38.541429** | **-1.122143** |

### Training/accounting
- base: `{'run_dir': 'experiments/archive/frontier_consolidation/training/runs/complianttok_reinvest_seed43022_r2', 'scientific_metrics': 'experiments/archive/frontier_consolidation/training/runs/complianttok_reinvest_seed43022_r2/scientific_metrics.json', 'mode': None, 'updates': 2529, 'total_charged_words': 100000000, 'total_main_word_exposure': None, 'total_aux_word_exposure': None, 'schedule_total': None, 'final_loss': 2.5525617599487305, 'mean_aux_loss': None, 'aux_loss_batches': None}`
- candidate: `{'run_dir': 'experiments/archive/frontier_consolidation/training/runs/dualview_shuffled_20M_seed43022', 'scientific_metrics': 'experiments/archive/frontier_consolidation/training/runs/dualview_shuffled_20M_seed43022/scientific_metrics.json', 'mode': 'shuffled', 'updates': 481, 'total_charged_words': 19988304, 'total_main_word_exposure': 19021584, 'total_aux_word_exposure': 966720, 'schedule_total': 2529, 'final_loss': 4.439597129821777, 'mean_aux_loss': 7.532271432331365, 'aux_loss_batches': 481}`

### Fragile groups
#### Supplement
| group | n | base item % | cand item % | net gain-loss | gain | loss |
|---|---:|---:|---:|---:|---:|---:|
| subject_aux_inversion | 3867 | 77.06 | 78.61 | +60 | 393 | 333 |
| qa_congruence_tricky | 165 | 35.76 | 36.97 | +2 | 12 | 10 |
| qa_congruence_easy | 64 | 53.12 | 43.75 | -6 | 2 | 8 |

#### EWoK
| group | n | base item % | cand item % | net gain-loss | gain | loss |
|---|---:|---:|---:|---:|---:|---:|
| material-dynamics | 770 | 55.58 | 53.64 | -15 | 229 | 244 |
| material-properties | 170 | 42.35 | 52.94 | +18 | 46 | 28 |
| spatial-relations | 490 | 45.10 | 46.94 | +9 | 129 | 120 |
| quantitative-properties | 314 | 51.27 | 52.55 | +4 | 63 | 59 |
| physical-dynamics | 120 | 56.67 | 57.50 | +1 | 34 | 33 |
| physical-interactions | 556 | 48.92 | 51.44 | +14 | 136 | 122 |
| social-relations | 1548 | 48.26 | 49.48 | +19 | 359 | 340 |

#### Entity
| group | n | base item % | cand item % | net gain-loss | gain | loss |
|---|---:|---:|---:|---:|---:|---:|
| regular_5_ops | 94 | 35.11 | 15.96 | -18 | 4 | 22 |
| regular_4_ops | 388 | 17.01 | 17.01 | +0 | 52 | 52 |
| regular_3_ops | 425 | 20.71 | 16.47 | -18 | 47 | 65 |
| ambiref_5_ops | 123 | 22.76 | 19.51 | -4 | 14 | 18 |
| ambiref_4_ops | 434 | 19.12 | 16.59 | -11 | 35 | 46 |
| move_contents_5_ops | 116 | 19.83 | 20.69 | +1 | 15 | 14 |
| move_contents_4_ops | 353 | 12.75 | 17.85 | +18 | 43 | 25 |

#### COMPS
| group | n | base item % | cand item % | net gain-loss | gain | loss |
|---|---:|---:|---:|---:|---:|---:|
| base | 49340 | 51.17 | 50.31 | -423 | 10807 | 11230 |
| wugs | 13896 | 49.63 | 49.09 | -76 | 3347 | 3423 |
| wugs_dist_before | 13896 | 49.83 | 49.96 | +19 | 1950 | 1931 |
| wugs_dist_in_between | 13896 | 50.41 | 49.86 | -77 | 1907 | 1984 |

#### GlobalPIQA
| group | n | base item % | cand item % | net gain-loss | gain | loss |
|---|---:|---:|---:|---:|---:|---:|
| GlobalPIQA_parallel | 103 | 20.39 | 25.24 | +5 | 10 | 5 |
| GlobalPIQA_nonparallel | 100 | 48.00 | 44.00 | -4 | 7 | 11 |

### Best/worst item-net groups
- BLiMP: best principle_A_domain_1 +402/914, wh_island +363/960, superlative_quantifiers_2 +307/986, wh_questions_object_gap +166/859; worst superlative_quantifiers_1 -324/979, determiner_noun_agreement_with_adj_2 -306/941, ellipsis_n_bar_2 -222/828, anaphor_number_agreement -232/931
- Supplement: best subject_aux_inversion +60/3867, qa_congruence_tricky +2/165, hypernym +8/842, turn_taking -21/280; worst qa_congruence_easy -6/64, turn_taking -21/280, hypernym +8/842, qa_congruence_tricky +2/165
- EWoK: best material-properties +18/170, social-properties +17/328, physical-interactions +14/556, spatial-relations +9/490; worst social-interactions -33/294, physical-relations -27/818, material-dynamics -15/770, agent-properties -14/2210
- Entity: best move_contents_4_ops +18/353, regular_1_ops +12/409, regular_0_ops +11/517, ambiref_1_ops +8/428; worst regular_5_ops -18/94, regular_3_ops -18/425, ambiref_2_ops -16/413, ambiref_5_ops -4/123
- COMPS: best wugs_dist_before +19/13896, wugs -76/13896, wugs_dist_in_between -77/13896, base -423/49340; worst base -423/49340, wugs_dist_in_between -77/13896, wugs -76/13896, wugs_dist_before +19/13896
- GlobalPIQA: best GlobalPIQA_parallel +5/103, GlobalPIQA_nonparallel -4/100; worst GlobalPIQA_nonparallel -4/100, GlobalPIQA_parallel +5/103

## mlm_only vs reference_20M

| Column | base | candidate | delta |
|---|---:|---:|---:|
| BLiMP | 59.6900 | 59.5400 | -0.1500 |
| Supplement | 55.4500 | 58.4300 | +2.9800 |
| EWoK | 50.7300 | 50.1000 | -0.6300 |
| Entity | 18.6500 | 18.3900 | -0.2600 |
| COMPS | 50.2600 | 50.7000 | +0.4400 |
| GlobalPIQA | 34.1950 | 32.6650 | -1.5300 |
| Reading | 8.6700 | 8.6800 | +0.0100 |
| **cheap7** | **39.663571** | **39.786429** | **+0.122857** |

### Training/accounting
- base: `{'run_dir': 'experiments/archive/frontier_consolidation/training/runs/complianttok_reinvest_seed43022_r2', 'scientific_metrics': 'experiments/archive/frontier_consolidation/training/runs/complianttok_reinvest_seed43022_r2/scientific_metrics.json', 'mode': None, 'updates': 2529, 'total_charged_words': 100000000, 'total_main_word_exposure': None, 'total_aux_word_exposure': None, 'schedule_total': None, 'final_loss': 2.5525617599487305, 'mean_aux_loss': None, 'aux_loss_batches': None}`
- candidate: `{'run_dir': 'experiments/archive/frontier_consolidation/training/runs/dualview_mlm_only_20M_seed43022', 'scientific_metrics': 'experiments/archive/frontier_consolidation/training/runs/dualview_mlm_only_20M_seed43022/scientific_metrics.json', 'mode': 'mlm_only', 'updates': 506, 'total_charged_words': 20000000, 'total_main_word_exposure': 20000000, 'total_aux_word_exposure': 0, 'schedule_total': 2529, 'final_loss': 3.870621681213379, 'mean_aux_loss': 0.0, 'aux_loss_batches': 0}`

### Fragile groups
#### Supplement
| group | n | base item % | cand item % | net gain-loss | gain | loss |
|---|---:|---:|---:|---:|---:|---:|
| subject_aux_inversion | 3867 | 77.06 | 76.26 | -31 | 215 | 246 |
| qa_congruence_tricky | 165 | 35.76 | 40.61 | +8 | 12 | 4 |
| qa_congruence_easy | 64 | 53.12 | 56.25 | +2 | 6 | 4 |

#### EWoK
| group | n | base item % | cand item % | net gain-loss | gain | loss |
|---|---:|---:|---:|---:|---:|---:|
| material-dynamics | 770 | 55.58 | 46.36 | -71 | 97 | 168 |
| material-properties | 170 | 42.35 | 49.41 | +12 | 43 | 31 |
| spatial-relations | 490 | 45.10 | 46.53 | +7 | 101 | 94 |
| quantitative-properties | 314 | 51.27 | 50.32 | -3 | 67 | 70 |
| physical-dynamics | 120 | 56.67 | 54.17 | -3 | 23 | 26 |
| physical-interactions | 556 | 48.92 | 50.54 | +9 | 102 | 93 |
| social-relations | 1548 | 48.26 | 49.48 | +19 | 283 | 264 |

#### Entity
| group | n | base item % | cand item % | net gain-loss | gain | loss |
|---|---:|---:|---:|---:|---:|---:|
| regular_5_ops | 94 | 35.11 | 31.91 | -3 | 7 | 10 |
| regular_4_ops | 388 | 17.01 | 16.24 | -3 | 14 | 17 |
| regular_3_ops | 425 | 20.71 | 20.71 | +0 | 19 | 19 |
| ambiref_5_ops | 123 | 22.76 | 17.89 | -6 | 7 | 13 |
| ambiref_4_ops | 434 | 19.12 | 14.75 | -19 | 11 | 30 |
| move_contents_5_ops | 116 | 19.83 | 19.83 | +0 | 6 | 6 |
| move_contents_4_ops | 353 | 12.75 | 13.31 | +2 | 20 | 18 |

#### COMPS
| group | n | base item % | cand item % | net gain-loss | gain | loss |
|---|---:|---:|---:|---:|---:|---:|
| base | 49340 | 51.17 | 51.91 | +368 | 7738 | 7370 |
| wugs | 13896 | 49.63 | 49.96 | +46 | 2443 | 2397 |
| wugs_dist_before | 13896 | 49.83 | 50.77 | +131 | 2356 | 2225 |
| wugs_dist_in_between | 13896 | 50.41 | 50.15 | -36 | 2277 | 2313 |

#### GlobalPIQA
| group | n | base item % | cand item % | net gain-loss | gain | loss |
|---|---:|---:|---:|---:|---:|---:|
| GlobalPIQA_parallel | 103 | 20.39 | 22.33 | +2 | 3 | 1 |
| GlobalPIQA_nonparallel | 100 | 48.00 | 43.00 | -5 | 6 | 11 |

### Best/worst item-net groups
- BLiMP: best npi_present_1 +124/909, npi_present_2 +116/914, left_branch_island_echo_question +97/947, principle_A_domain_2 +77/915; worst ellipsis_n_bar_1 -133/802, principle_A_case_2 -105/915, only_npi_licensor_present -101/882, determiner_noun_agreement_with_adj_2 -95/941
- Supplement: best turn_taking +14/280, qa_congruence_tricky +8/165, qa_congruence_easy +2/64, hypernym +23/842; worst subject_aux_inversion -31/3867, hypernym +23/842, qa_congruence_easy +2/64, qa_congruence_tricky +8/165
- EWoK: best material-properties +12/170, social-properties +6/328, physical-interactions +9/556, spatial-relations +7/490; worst material-dynamics -71/770, social-interactions -25/294, physical-dynamics -3/120, quantitative-properties -3/314
- Entity: best regular_1_ops +17/409, ambiref_3_ops +15/409, ambiref_1_ops +7/428, move_contents_1_ops +6/437; worst ambiref_5_ops -6/123, ambiref_4_ops -19/434, regular_5_ops -3/94, ambiref_0_ops -11/508
- COMPS: best wugs_dist_before +131/13896, base +368/49340, wugs +46/13896, wugs_dist_in_between -36/13896; worst wugs_dist_in_between -36/13896, wugs +46/13896, base +368/49340, wugs_dist_before +131/13896
- GlobalPIQA: best GlobalPIQA_parallel +2/103, GlobalPIQA_nonparallel -5/100; worst GlobalPIQA_nonparallel -5/100, GlobalPIQA_parallel +2/103

## shuffled vs aligned

| Column | base | candidate | delta |
|---|---:|---:|---:|
| BLiMP | 56.7300 | 56.5400 | -0.1900 |
| Supplement | 51.7600 | 52.8200 | +1.0600 |
| EWoK | 51.1100 | 51.3100 | +0.2000 |
| Entity | 17.9100 | 17.3500 | -0.5600 |
| COMPS | 50.2000 | 49.8000 | -0.4000 |
| GlobalPIQA | 36.1050 | 34.6200 | -1.4850 |
| Reading | 7.5250 | 7.3500 | -0.1750 |
| **cheap7** | **38.762857** | **38.541429** | **-0.221429** |

### Training/accounting
- base: `{'run_dir': 'experiments/archive/frontier_consolidation/training/runs/dualview_aligned_20M_seed43022', 'scientific_metrics': 'experiments/archive/frontier_consolidation/training/runs/dualview_aligned_20M_seed43022/scientific_metrics.json', 'mode': 'aligned', 'updates': 481, 'total_charged_words': 19988304, 'total_main_word_exposure': 19021584, 'total_aux_word_exposure': 966720, 'schedule_total': 2529, 'final_loss': 4.379467487335205, 'mean_aux_loss': 7.470588489233876, 'aux_loss_batches': 481}`
- candidate: `{'run_dir': 'experiments/archive/frontier_consolidation/training/runs/dualview_shuffled_20M_seed43022', 'scientific_metrics': 'experiments/archive/frontier_consolidation/training/runs/dualview_shuffled_20M_seed43022/scientific_metrics.json', 'mode': 'shuffled', 'updates': 481, 'total_charged_words': 19988304, 'total_main_word_exposure': 19021584, 'total_aux_word_exposure': 966720, 'schedule_total': 2529, 'final_loss': 4.439597129821777, 'mean_aux_loss': 7.532271432331365, 'aux_loss_batches': 481}`

### Fragile groups
#### Supplement
| group | n | base item % | cand item % | net gain-loss | gain | loss |
|---|---:|---:|---:|---:|---:|---:|
| subject_aux_inversion | 3867 | 76.75 | 78.61 | +72 | 216 | 144 |
| qa_congruence_tricky | 165 | 32.73 | 36.97 | +7 | 11 | 4 |
| qa_congruence_easy | 64 | 43.75 | 43.75 | +0 | 3 | 3 |

#### EWoK
| group | n | base item % | cand item % | net gain-loss | gain | loss |
|---|---:|---:|---:|---:|---:|---:|
| material-dynamics | 770 | 51.43 | 53.64 | +17 | 172 | 155 |
| material-properties | 170 | 56.47 | 52.94 | -6 | 40 | 46 |
| spatial-relations | 490 | 50.00 | 46.94 | -15 | 81 | 96 |
| quantitative-properties | 314 | 52.87 | 52.55 | -1 | 69 | 70 |
| physical-dynamics | 120 | 45.00 | 57.50 | +15 | 37 | 22 |
| physical-interactions | 556 | 52.16 | 51.44 | -4 | 133 | 137 |
| social-relations | 1548 | 50.00 | 49.48 | -8 | 294 | 302 |

#### Entity
| group | n | base item % | cand item % | net gain-loss | gain | loss |
|---|---:|---:|---:|---:|---:|---:|
| regular_5_ops | 94 | 18.09 | 15.96 | -2 | 1 | 3 |
| regular_4_ops | 388 | 18.30 | 17.01 | -5 | 27 | 32 |
| regular_3_ops | 425 | 15.76 | 16.47 | +3 | 27 | 24 |
| ambiref_5_ops | 123 | 21.95 | 19.51 | -3 | 10 | 13 |
| ambiref_4_ops | 434 | 16.59 | 16.59 | +0 | 36 | 36 |
| move_contents_5_ops | 116 | 27.59 | 20.69 | -8 | 10 | 18 |
| move_contents_4_ops | 353 | 15.30 | 17.85 | +9 | 30 | 21 |

#### COMPS
| group | n | base item % | cand item % | net gain-loss | gain | loss |
|---|---:|---:|---:|---:|---:|---:|
| base | 49340 | 50.82 | 50.31 | -250 | 8731 | 8981 |
| wugs | 13896 | 49.43 | 49.09 | -48 | 2420 | 2468 |
| wugs_dist_before | 13896 | 50.21 | 49.96 | -34 | 1350 | 1384 |
| wugs_dist_in_between | 13896 | 50.35 | 49.86 | -68 | 1334 | 1402 |

#### GlobalPIQA
| group | n | base item % | cand item % | net gain-loss | gain | loss |
|---|---:|---:|---:|---:|---:|---:|
| GlobalPIQA_parallel | 103 | 26.21 | 25.24 | -1 | 7 | 8 |
| GlobalPIQA_nonparallel | 100 | 46.00 | 44.00 | -2 | 4 | 6 |

### Best/worst item-net groups
- BLiMP: best wh_island +208/960, superlative_quantifiers_2 +180/986, principle_A_domain_1 +157/914, principle_A_c_command +112/946; worst superlative_quantifiers_1 -288/979, only_npi_scope -143/837, left_branch_island_echo_question -112/947, anaphor_number_agreement -109/931
- Supplement: best qa_congruence_tricky +7/165, subject_aux_inversion +72/3867, turn_taking +2/280, qa_congruence_easy +0/64; worst hypernym -13/842, qa_congruence_easy +0/64, turn_taking +2/280, subject_aux_inversion +72/3867
- EWoK: best physical-dynamics +15/120, material-dynamics +17/770, social-properties +7/328, quantitative-properties -1/314; worst material-properties -6/170, spatial-relations -15/490, social-interactions -9/294, physical-relations -23/818
- Entity: best ambiref_3_ops +12/409, ambiref_1_ops +12/428, move_contents_4_ops +9/353, regular_2_ops +5/405; worst move_contents_5_ops -8/116, ambiref_2_ops -15/413, move_contents_0_ops -14/516, ambiref_5_ops -3/123
- COMPS: best wugs_dist_before -34/13896, wugs -48/13896, wugs_dist_in_between -68/13896, base -250/49340; worst base -250/49340, wugs_dist_in_between -68/13896, wugs -48/13896, wugs_dist_before -34/13896
- GlobalPIQA: best GlobalPIQA_parallel -1/103, GlobalPIQA_nonparallel -2/100; worst GlobalPIQA_nonparallel -2/100, GlobalPIQA_parallel -1/103

## mlm_only vs aligned

| Column | base | candidate | delta |
|---|---:|---:|---:|
| BLiMP | 56.7300 | 59.5400 | +2.8100 |
| Supplement | 51.7600 | 58.4300 | +6.6700 |
| EWoK | 51.1100 | 50.1000 | -1.0100 |
| Entity | 17.9100 | 18.3900 | +0.4800 |
| COMPS | 50.2000 | 50.7000 | +0.5000 |
| GlobalPIQA | 36.1050 | 32.6650 | -3.4400 |
| Reading | 7.5250 | 8.6800 | +1.1550 |
| **cheap7** | **38.762857** | **39.786429** | **+1.023571** |

### Training/accounting
- base: `{'run_dir': 'experiments/archive/frontier_consolidation/training/runs/dualview_aligned_20M_seed43022', 'scientific_metrics': 'experiments/archive/frontier_consolidation/training/runs/dualview_aligned_20M_seed43022/scientific_metrics.json', 'mode': 'aligned', 'updates': 481, 'total_charged_words': 19988304, 'total_main_word_exposure': 19021584, 'total_aux_word_exposure': 966720, 'schedule_total': 2529, 'final_loss': 4.379467487335205, 'mean_aux_loss': 7.470588489233876, 'aux_loss_batches': 481}`
- candidate: `{'run_dir': 'experiments/archive/frontier_consolidation/training/runs/dualview_mlm_only_20M_seed43022', 'scientific_metrics': 'experiments/archive/frontier_consolidation/training/runs/dualview_mlm_only_20M_seed43022/scientific_metrics.json', 'mode': 'mlm_only', 'updates': 506, 'total_charged_words': 20000000, 'total_main_word_exposure': 20000000, 'total_aux_word_exposure': 0, 'schedule_total': 2529, 'final_loss': 3.870621681213379, 'mean_aux_loss': 0.0, 'aux_loss_batches': 0}`

### Fragile groups
#### Supplement
| group | n | base item % | cand item % | net gain-loss | gain | loss |
|---|---:|---:|---:|---:|---:|---:|
| subject_aux_inversion | 3867 | 76.75 | 76.26 | -19 | 292 | 311 |
| qa_congruence_tricky | 165 | 32.73 | 40.61 | +13 | 19 | 6 |
| qa_congruence_easy | 64 | 43.75 | 56.25 | +8 | 9 | 1 |

#### EWoK
| group | n | base item % | cand item % | net gain-loss | gain | loss |
|---|---:|---:|---:|---:|---:|---:|
| material-dynamics | 770 | 51.43 | 46.36 | -39 | 167 | 206 |
| material-properties | 170 | 56.47 | 49.41 | -12 | 39 | 51 |
| spatial-relations | 490 | 50.00 | 46.53 | -17 | 103 | 120 |
| quantitative-properties | 314 | 52.87 | 50.32 | -8 | 80 | 88 |
| physical-dynamics | 120 | 45.00 | 54.17 | +11 | 31 | 20 |
| physical-interactions | 556 | 52.16 | 50.54 | -9 | 133 | 142 |
| social-relations | 1548 | 50.00 | 49.48 | -8 | 349 | 357 |

#### Entity
| group | n | base item % | cand item % | net gain-loss | gain | loss |
|---|---:|---:|---:|---:|---:|---:|
| regular_5_ops | 94 | 18.09 | 31.91 | +13 | 21 | 8 |
| regular_4_ops | 388 | 18.30 | 16.24 | -8 | 47 | 55 |
| regular_3_ops | 425 | 15.76 | 20.71 | +21 | 61 | 40 |
| ambiref_5_ops | 123 | 21.95 | 17.89 | -5 | 14 | 19 |
| ambiref_4_ops | 434 | 16.59 | 14.75 | -8 | 40 | 48 |
| move_contents_5_ops | 116 | 27.59 | 19.83 | -9 | 11 | 20 |
| move_contents_4_ops | 353 | 15.30 | 13.31 | -7 | 30 | 37 |

#### COMPS
| group | n | base item % | cand item % | net gain-loss | gain | loss |
|---|---:|---:|---:|---:|---:|---:|
| base | 49340 | 50.82 | 51.91 | +541 | 10277 | 9736 |
| wugs | 13896 | 49.43 | 49.96 | +74 | 3222 | 3148 |
| wugs_dist_before | 13896 | 50.21 | 50.77 | +78 | 2552 | 2474 |
| wugs_dist_in_between | 13896 | 50.35 | 50.15 | -27 | 2505 | 2532 |

#### GlobalPIQA
| group | n | base item % | cand item % | net gain-loss | gain | loss |
|---|---:|---:|---:|---:|---:|---:|
| GlobalPIQA_parallel | 103 | 26.21 | 22.33 | -4 | 6 | 10 |
| GlobalPIQA_nonparallel | 100 | 46.00 | 43.00 | -3 | 8 | 11 |

### Best/worst item-net groups
- BLiMP: best ellipsis_n_bar_2 +218/828, existential_there_quantifiers_2 +216/911, determiner_noun_agreement_with_adj_irregular_1 +165/718, determiner_noun_agreement_2 +208/931; worst only_npi_scope -213/837, principle_A_domain_1 -215/914, superlative_quantifiers_2 -158/986, wh_questions_object_gap -126/859
- Supplement: best turn_taking +37/280, qa_congruence_easy +8/64, qa_congruence_tricky +13/165, hypernym +2/842; worst subject_aux_inversion -19/3867, hypernym +2/842, qa_congruence_tricky +13/165, qa_congruence_easy +8/64
- EWoK: best physical-dynamics +11/120, agent-properties +30/2210, physical-relations +2/818, social-interactions -1/294; worst material-properties -12/170, material-dynamics -39/770, spatial-relations -17/490, quantitative-properties -8/314
- Entity: best regular_5_ops +13/94, regular_3_ops +21/425, ambiref_3_ops +20/409, ambiref_1_ops +11/428; worst move_contents_5_ops -9/116, ambiref_5_ops -5/123, regular_4_ops -8/388, move_contents_4_ops -7/353
- COMPS: best base +541/49340, wugs_dist_before +78/13896, wugs +74/13896, wugs_dist_in_between -27/13896; worst wugs_dist_in_between -27/13896, wugs +74/13896, wugs_dist_before +78/13896, base +541/49340
- GlobalPIQA: best GlobalPIQA_nonparallel -3/100, GlobalPIQA_parallel -4/103; worst GlobalPIQA_parallel -4/103, GlobalPIQA_nonparallel -3/100

JSON: `experiments/archive/frontier_consolidation/data/dualview_sentinel_compare/dualview_panel_complete.json`

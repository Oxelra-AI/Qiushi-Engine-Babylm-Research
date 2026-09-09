# fastpath private localization private-pathway pocket summary

Status: **COMPLETE**

## Aggregate and column partitions

Aggregate: `{'n_common': 170722, 'anchor_correct': 98478, 'coherent_correct': 98363, 'ordinary_correct': 98596, 'shuffled_correct': 99476, 'private_gain': 3116, 'private_loss': 3231, 'private_retained_anchor_correct': 95247, 'private_remains_wrong': 69128, 'private_gain_also_ordinary': 1791, 'private_gain_not_ordinary': 1325, 'private_gain_also_shuffled': 1485, 'private_gain_not_shuffled': 1631, 'private_gain_unique_vs_ordinary_and_shuffled': 673, 'private_gain_shared_by_both_comparators': 833, 'private_loss_also_ordinary_wrong': 1777, 'private_loss_also_shuffled_wrong': 1427, 'private_loss_unique_coherent_wrong_vs_comparators': 812, 'retained_anchor_correct_ordinary_wrong': 3471, 'retained_anchor_correct_shuffled_wrong': 12920, 'retained_anchor_correct_both_comparators_wrong': 1325, 'coherent_correct_ordinary_wrong': 4796, 'coherent_correct_shuffled_wrong': 14551, 'coherent_correct_both_comparators_wrong': 1998, 'ordinary_correct_coherent_wrong': 5029, 'shuffled_correct_coherent_wrong': 15664, 'anchor_correct_pct': 57.683251133421585, 'coherent_correct_pct': 57.61589016061199, 'ordinary_correct_pct': 57.752369349000126, 'shuffled_correct_pct': 58.267827227890955, 'private_gain_pct': 1.8251894893452514, 'private_loss_pct': 1.8925504621548481, 'private_retained_anchor_correct_pct': 55.79070067126674, 'private_remains_wrong_pct': 40.491559377233166, 'retained_anchor_correct_ordinary_wrong_pct': 2.0331298836705285, 'retained_anchor_correct_shuffled_wrong_pct': 7.567858858260799, 'coherent_correct_ordinary_wrong_pct': 2.8092454399550144, 'ordinary_correct_coherent_wrong_pct': 2.945724628343154, 'coherent_correct_shuffled_wrong_pct': 8.523213176977777, 'shuffled_correct_coherent_wrong_pct': 9.175150244256745, 'coherent_minus_anchor_item_net': -115, 'coherent_minus_ordinary_item_net': -233, 'coherent_minus_shuffled_item_net': -1113}`

| column | n | coherent-anchor net | coherent-ordinary net | coherent-shuffled net | private gains | private losses | unique private gains | retained anchor correct while ordinary wrong |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BLiMP | 59875 | 30 | 42 | -421 | 761 | 731 | 169 | 784 |
| Supplement | 5218 | -14 | -21 | 64 | 37 | 51 | 13 | 47 |
| EWoK | 7618 | -2 | -27 | -34 | 147 | 149 | 35 | 159 |
| Entity | 6780 | 7 | -14 | 80 | 73 | 66 | 20 | 96 |
| COMPS | 91028 | -137 | -217 | -799 | 2095 | 2232 | 435 | 2377 |
| GlobalPIQA | 203 | 1 | 4 | -3 | 3 | 2 | 1 | 8 |

## Top coherent-specific private gains

| column | group | n | coherent-anchor net | coherent-ordinary net | coherent-shuffled net | private gains | private losses | unique private gains | retained anchor correct while ordinary wrong |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| COMPS | base | 49340 | -38 | -71 | -603 | 1117 | 1155 | 225 | 1314 |
| COMPS | wugs_dist_in_between | 13896 | 12 | -119 | 1455 | 362 | 350 | 85 | 353 |
| COMPS | wugs | 13896 | -26 | -44 | -126 | 270 | 296 | 69 | 304 |
| COMPS | wugs_dist_before | 13896 | -85 | 17 | -1525 | 346 | 431 | 56 | 406 |
| EWoK | social-relations | 1548 | 11 | -15 | -27 | 40 | 29 | 13 | 30 |
| BLiMP | wh_island | 960 | 28 | 43 | 65 | 35 | 7 | 13 | 26 |
| Supplement | subject_aux_inversion | 3867 | -17 | -27 | 27 | 19 | 36 | 10 | 25 |
| EWoK | agent-properties | 2210 | -8 | 5 | 18 | 34 | 42 | 9 | 51 |
| BLiMP | existential_there_quantifiers_2 | 911 | 19 | -2 | 90 | 29 | 10 | 9 | 14 |
| BLiMP | distractor_agreement_relative_clause | 871 | 10 | -3 | 8 | 20 | 10 | 8 | 12 |
| BLiMP | matrix_question_npi_licensor_present | 929 | 7 | -3 | 94 | 20 | 13 | 8 | 14 |
| BLiMP | adjunct_island | 928 | 14 | 7 | 173 | 21 | 7 | 7 | 15 |

## Top retained anchor decisions where ordinary86 is wrong

| column | group | n | coherent-anchor net | coherent-ordinary net | coherent-shuffled net | private gains | private losses | unique private gains | retained anchor correct while ordinary wrong |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| COMPS | base | 49340 | -38 | -71 | -603 | 1117 | 1155 | 225 | 1314 |
| COMPS | wugs_dist_before | 13896 | -85 | 17 | -1525 | 346 | 431 | 56 | 406 |
| COMPS | wugs_dist_in_between | 13896 | 12 | -119 | 1455 | 362 | 350 | 85 | 353 |
| COMPS | wugs | 13896 | -26 | -44 | -126 | 270 | 296 | 69 | 304 |
| EWoK | agent-properties | 2210 | -8 | 5 | 18 | 34 | 42 | 9 | 51 |
| BLiMP | principle_A_c_command | 946 | 15 | 27 | -36 | 27 | 12 | 6 | 34 |
| EWoK | social-relations | 1548 | 11 | -15 | -27 | 40 | 29 | 13 | 30 |
| BLiMP | tough_vs_raising_1 | 948 | 3 | 28 | 45 | 12 | 9 | 6 | 29 |
| BLiMP | wh_island | 960 | 28 | 43 | 65 | 35 | 7 | 13 | 26 |
| BLiMP | only_npi_licensor_present | 882 | 4 | 21 | -28 | 14 | 10 | 5 | 26 |
| Supplement | subject_aux_inversion | 3867 | -17 | -27 | 27 | 19 | 36 | 10 | 25 |
| BLiMP | sentential_negation_npi_scope | 871 | -10 | 10 | 13 | 9 | 19 | 3 | 25 |

## Focused Entity/EWoK/Supplement/GlobalPIQA coherent-specific gains

| column | group | n | coherent-anchor net | coherent-ordinary net | coherent-shuffled net | private gains | private losses | unique private gains | retained anchor correct while ordinary wrong |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EWoK | social-relations | 1548 | 11 | -15 | -27 | 40 | 29 | 13 | 30 |
| Supplement | subject_aux_inversion | 3867 | -17 | -27 | 27 | 19 | 36 | 10 | 25 |
| EWoK | agent-properties | 2210 | -8 | 5 | 18 | 34 | 42 | 9 | 51 |
| EWoK | physical-interactions | 556 | 4 | 5 | -16 | 9 | 5 | 3 | 17 |
| EWoK | social-interactions | 294 | 2 | 4 | 3 | 10 | 8 | 3 | 8 |
| Entity | regular_4_ops | 388 | 2 | -9 | 0 | 4 | 2 | 3 | 3 |
| Supplement | hypernym | 842 | 0 | 0 | 18 | 13 | 13 | 3 | 12 |
| EWoK | spatial-relations | 490 | -3 | -8 | -11 | 6 | 9 | 3 | 8 |
| Entity | ambiref_2_ops | 413 | 4 | 2 | 19 | 7 | 3 | 2 | 5 |
| Entity | regular_3_ops | 425 | 3 | 5 | 2 | 5 | 2 | 2 | 6 |
| EWoK | quantitative-properties | 314 | -3 | -3 | -7 | 9 | 12 | 2 | 6 |
| Entity | move_contents_0_ops | 516 | 5 | -3 | -1 | 7 | 2 | 1 | 10 |

## Largest private-loss families

| column | group | n | coherent-anchor net | coherent-ordinary net | coherent-shuffled net | private gains | private losses | unique private gains | retained anchor correct while ordinary wrong |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| COMPS | base | 49340 | -38 | -71 | -603 | 1117 | 1155 | 225 | 1314 |
| COMPS | wugs_dist_before | 13896 | -85 | 17 | -1525 | 346 | 431 | 56 | 406 |
| COMPS | wugs_dist_in_between | 13896 | 12 | -119 | 1455 | 362 | 350 | 85 | 353 |
| COMPS | wugs | 13896 | -26 | -44 | -126 | 270 | 296 | 69 | 304 |
| BLiMP | npi_present_2 | 914 | -57 | -15 | -137 | 4 | 61 | 0 | 17 |
| EWoK | agent-properties | 2210 | -8 | 5 | 18 | 34 | 42 | 9 | 51 |
| Supplement | subject_aux_inversion | 3867 | -17 | -27 | 27 | 19 | 36 | 10 | 25 |
| BLiMP | npi_present_1 | 909 | -31 | 4 | -114 | 4 | 35 | 2 | 21 |
| EWoK | social-relations | 1548 | 11 | -15 | -27 | 40 | 29 | 13 | 30 |
| BLiMP | left_branch_island_simple_question | 951 | -15 | 12 | -97 | 12 | 27 | 0 | 22 |
| BLiMP | ellipsis_n_bar_1 | 802 | -13 | -9 | -62 | 12 | 25 | 3 | 12 |
| EWoK | material-dynamics | 770 | 1 | -10 | 7 | 23 | 22 | 0 | 10 |

## Interpretation

Private path has real but small coherent-specific pockets. Most count-level gains come from BLiMP/COMPS due to their size; official-score gains are macro-weighted and strongest in Supplement and GlobalPIQA. Focused Entity/EWoK pockets are mixed and not yet a broad relation-tracking repair.

JSON: `experiments/archive/representation_and_objectives/data/fastpath_private_localization/fastpath_private_pocket_summary.json`

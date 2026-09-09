# coherent86 multiarm mechanism reading private-scale and fast-path panel analysis

Status: **COMPLETE**

This analysis uses saved official-compatible prediction payloads only; it runs no model inference.

## Arms

| arm | payload | cheap7 | BLiMP | Supplement | EWoK | Entity | COMPS | GlobalPIQA | Reading |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| chck82_anchor | `experiments/archive/representation_and_objectives/data/scale1p75_chck82_full_eval_reproduction/staged_full_eval/per_target/scale1p75_chck82_independent.json` | 43.95944987645173 | 68.49128403651986 | 62.9378112562002 | 50.05545332553276 | 28.314041930298774 | 52.19117509443596 | 37.57766990291262 | 8.148713589261902 |
| ordinary86_backbone | `experiments/archive/representation_and_objectives/data/scale1p75_chck86_cheap7_eval/per_target/scale1p75_chck86_cheap7.json` | 43.770714285714284 | 68.47 | 62.68 | 50.14 | 28.58 | 52.29 | 36.135 | 8.1 |
| shuffled86_private | `experiments/archive/frontier_consolidation/data/frozen82_tail4M_shuffled_eval/per_target/frozen82_tail4M_shuffled.json` | 44.01285714285714 | 69.23 | 59.81 | 51.44 | 26.77 | 52.65 | 39.565 | 8.625 |
| coherent86_private_alpha1 | `experiments/archive/frontier_consolidation/data/fastpath4M_coherent_eval/per_target/fastpath4M_coherent.json` | 44.10642857142857 | 68.52 | 63.65 | 49.91 | 28.44 | 51.99 | 38.065 | 8.17 |
| spanbreak86_private | `experiments/archive/frontier_consolidation/data/fastpath4M_spanbreak_eval/per_target/fastpath4M_spanbreak.json` | 43.121428571428574 | 68.18 | 62.36 | 49.37 | 24.24 | 52.04 | 37.565 | 8.095 |
| coherent86_private_alpha0p5 | `experiments/archive/frontier_consolidation/data/private_scale_sentinel_eval/coherent86_private_scale_0p5/per_target/coherent86_private_scale_0p5.json` | 44.17785714285714 | 68.54 | 63.15 | 50.0 | 28.23 | 52.11 | 39.05 | 8.165 |
| coherent86_private_alpha0p75 | `experiments/archive/frontier_consolidation/data/private_scale_sentinel_eval/coherent86_private_scale_0p75/per_target/coherent86_private_scale_0p75.json` | 44.18142857142857 | 68.51 | 63.64 | 50.02 | 28.32 | 52.05 | 38.565 | 8.165 |

## Anchor comparison surface

| arm | Δcheap7 vs anchor | net discrete items | changed items | retention fraction | gain/loss | EWoK focus net | Entity focus net | Supplement focus net |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| coherent86_private_alpha0p75 | 0.22197869497684053 | -86 | 4750 | 0.9754462925729604 | 0.9644334160463193 | 5 | -3 | 4 |
| coherent86_private_alpha0p5 | 0.2184072664054142 | -21 | 3141 | 0.9839456528361664 | 0.9867172675521821 | 10 | -4 | -1 |
| coherent86_private_alpha1 | 0.1469786949768448 | -115 | 6347 | 0.9671906415646134 | 0.9644073042401733 | 6 | 1 | 3 |
| shuffled86_private | 0.053407266405415044 | 998 | 29692 | 0.8543126383557749 | 1.0695615808182894 | 58 | -45 | -34 |
| ordinary86_backbone | -0.18873559073744417 | 118 | 10614 | 0.9467089096041755 | 1.022484756097561 | 38 | 11 | -3 |
| spanbreak86_private | -0.8380213050231546 | -491 | 17707 | 0.9076037287515993 | 0.9460380261567205 | 0 | -265 | -16 |

## Direct compact comparisons

### ordinary86_backbone_minus_chck82_anchor

Aggregate: net 118 from 5366 gains / 5248 losses over 170722 common discrete items; retention fraction 0.9467089096041755; mean payload Δ -0.2121.

| column | payload Δ | gains | losses | net | retention | best groups | worst groups |
|---|---:|---:|---:|---:|---:|---|---|
| BLiMP | -0.021284036519858773 | 1172 | 1184 | -12 | 0.9710775093436256 | tough_vs_raising_2 +30/920, principle_A_domain_1 +28/914, intransitive +25/868 | npi_present_2 -42/914, npi_present_1 -35/909, left_branch_island_simple_question -27/951 |
| Supplement | -0.25781125620019907 | 71 | 64 | 7 | 0.9834625322997416 | subject_aux_inversion +10/3867, hypernym +0/842, qa_congruence_easy +0/64 | qa_congruence_tricky -2/165, turn_taking -1/280, hypernym +0/842 |
| EWoK | 0.08454667446724073 | 275 | 250 | 25 | 0.9342797055730809 | social-relations +26/1548, material-dynamics +11/770, spatial-relations +5/490 | social-properties -5/328, physical-dynamics -1/120, social-interactions -2/294 |
| Entity | 0.2659580697012238 | 145 | 124 | 21 | 0.9344262295081968 | regular_4_ops +11/388, move_contents_4_ops +8/353, move_contents_0_ops +8/516 | move_contents_5_ops -2/116, move_contents_3_ops -6/406, ambiref_3_ops -5/409 |
| COMPS | 0.09882490556403667 | 3697 | 3617 | 80 | 0.9244885177453027 | wugs_dist_in_between +131/13896, wugs +18/13896, base +33/49340 | wugs_dist_before -102/13896, base +33/49340, wugs +18/13896 |
| GlobalPIQA | -1.4426699029126198 | 6 | 9 | -3 | 0.881578947368421 | GlobalPIQA_nonparallel +1/100, GlobalPIQA_parallel -4/103 | GlobalPIQA_parallel -4/103, GlobalPIQA_nonparallel +1/100 |

Focus nets:
- EWoK_relation_like: net 38 from gains 211 / losses 173 over n=5408
- Entity_high_operation: net 11 from gains 64 / losses 53 over n=2748
- Supplement_selected: net -3 from gains 24 / losses 27 over n=1351

### shuffled86_private_minus_chck82_anchor

Aggregate: net 998 from 15345 gains / 14347 losses over 170722 common discrete items; retention fraction 0.8543126383557749; mean payload Δ -0.0171.

| column | payload Δ | gains | losses | net | retention | best groups | worst groups |
|---|---:|---:|---:|---:|---:|---|---|
| BLiMP | 0.7387159634801463 | 3488 | 3037 | 451 | 0.9258128343552288 | only_npi_scope +124/837, wh_questions_object_gap +124/859, npi_present_1 +83/909 | adjunct_island -159/928, matrix_question_npi_licensor_present -87/929, existential_there_quantifiers_2 -71/911 |
| Supplement | -3.1278112562001965 | 154 | 232 | -78 | 0.9400516795865633 | turn_taking +3/280, subject_aux_inversion -44/3867, hypernym -18/842 | qa_congruence_tricky -17/165, qa_congruence_easy -2/64, hypernym -18/842 |
| EWoK | 1.384546674467238 | 709 | 677 | 32 | 0.8220294426919033 | physical-dynamics +12/120, physical-interactions +20/556, social-relations +38/1548 | physical-relations -19/818, agent-properties -26/2210, material-dynamics -6/770 |
| Entity | -1.544041930298775 | 218 | 291 | -73 | 0.8461131676361714 | move_contents_0_ops +6/516, ambiref_5_ops +1/123, regular_1_ops +3/409 | regular_5_ops -7/94, move_contents_5_ops -8/116, ambiref_2_ops -15/413 |
| COMPS | 0.4588249055640361 | 10762 | 10100 | 662 | 0.7891440501043842 | wugs_dist_before +1440/13896, base +565/49340, wugs +100/13896 | wugs_dist_in_between -1443/13896, wugs +100/13896, base +565/49340 |
| GlobalPIQA | 1.98733009708738 | 14 | 10 | 4 | 0.868421052631579 | GlobalPIQA_nonparallel +3/100, GlobalPIQA_parallel +1/103 | GlobalPIQA_parallel +1/103, GlobalPIQA_nonparallel +3/100 |

Focus nets:
- EWoK_relation_like: net 58 from gains 551 / losses 493 over n=5408
- Entity_high_operation: net -45 from gains 76 / losses 121 over n=2748
- Supplement_selected: net -34 from gains 66 / losses 100 over n=1351

### coherent86_private_alpha1_minus_chck82_anchor

Aggregate: net -115 from 3116 gains / 3231 losses over 170722 common discrete items; retention fraction 0.9671906415646134; mean payload Δ +0.1679.

| column | payload Δ | gains | losses | net | retention | best groups | worst groups |
|---|---:|---:|---:|---:|---:|---|---|
| BLiMP | 0.028715963480138385 | 761 | 731 | 30 | 0.9821432933532013 | superlative_quantifiers_2 +30/986, wh_island +28/960, principle_A_domain_1 +23/914 | npi_present_2 -57/914, npi_present_1 -31/909, ellipsis_n_bar_1 -13/802 |
| Supplement | 0.7121887437997998 | 37 | 51 | -14 | 0.9868217054263566 | qa_congruence_easy +2/64, qa_congruence_tricky +2/165, hypernym +0/842 | subject_aux_inversion -17/3867, turn_taking -1/280, hypernym +0/842 |
| EWoK | -0.14545332553276324 | 147 | 149 | -2 | 0.9608307045215563 | material-properties +2/170, physical-interactions +4/556, social-relations +11/1548 | physical-dynamics -2/120, social-properties -4/328, quantitative-properties -3/314 |
| Entity | 0.1259580697012268 | 73 | 66 | 7 | 0.965097831835008 | move_contents_5_ops +3/116, move_contents_0_ops +5/516, ambiref_2_ops +4/413 | move_contents_3_ops -5/406, regular_5_ops -1/94, ambiref_5_ops -1/123 |
| COMPS | -0.2011750944359605 | 2095 | 2232 | -137 | 0.9534029227557411 | wugs_dist_in_between +12/13896, base -38/49340, wugs -26/13896 | wugs_dist_before -85/13896, wugs -26/13896, base -38/49340 |
| GlobalPIQA | 0.48733009708737995 | 3 | 2 | 1 | 0.9736842105263158 | GlobalPIQA_parallel +1/103, GlobalPIQA_nonparallel +0/100 | GlobalPIQA_nonparallel +0/100, GlobalPIQA_parallel +1/103 |

Focus nets:
- EWoK_relation_like: net 6 from gains 113 / losses 107 over n=5408
- Entity_high_operation: net 1 from gains 29 / losses 28 over n=2748
- Supplement_selected: net 3 from gains 18 / losses 15 over n=1351

### spanbreak86_private_minus_chck82_anchor

Aggregate: net -491 from 8608 gains / 9099 losses over 170722 common discrete items; retention fraction 0.9076037287515993; mean payload Δ -0.9687.

| column | payload Δ | gains | losses | net | retention | best groups | worst groups |
|---|---:|---:|---:|---:|---:|---|---|
| BLiMP | -0.3112840365198508 | 1771 | 1953 | -182 | 0.9522925470845446 | existential_there_quantifiers_2 +70/911, only_npi_licensor_present +67/882, wh_island +56/960 | matrix_question_npi_licensor_present -68/929, npi_present_2 -61/914, only_npi_scope -35/837 |
| Supplement | -0.5778112562001994 | 101 | 144 | -43 | 0.9627906976744186 | qa_congruence_easy +1/64, turn_taking +1/280, subject_aux_inversion -27/3867 | qa_congruence_tricky -4/165, hypernym -14/842, subject_aux_inversion -27/3867 |
| EWoK | -0.6854533255327624 | 427 | 439 | -12 | 0.8845951629863302 | social-properties +5/328, physical-interactions +4/556, material-dynamics +5/770 | physical-dynamics -9/120, material-properties -4/170, social-interactions -3/294 |
| Entity | -4.074041930298776 | 486 | 635 | -149 | 0.6641988365943945 | regular_0_ops +72/517, move_contents_0_ops +62/516, ambiref_0_ops +48/508 | move_contents_5_ops -28/116, move_contents_4_ops -45/353, ambiref_4_ops -47/434 |
| COMPS | -0.15117509443596333 | 5818 | 5923 | -105 | 0.8763465553235908 | wugs_dist_before +427/13896, base -33/49340, wugs -13/13896 | wugs_dist_in_between -486/13896, wugs -13/13896, base -33/49340 |
| GlobalPIQA | -0.012669902912620046 | 5 | 5 | 0 | 0.9342105263157895 | GlobalPIQA_parallel +1/103, GlobalPIQA_nonparallel -1/100 | GlobalPIQA_nonparallel -1/100, GlobalPIQA_parallel +1/103 |

Focus nets:
- EWoK_relation_like: net 0 from gains 324 / losses 324 over n=5408
- Entity_high_operation: net -265 from gains 104 / losses 369 over n=2748
- Supplement_selected: net -16 from gains 28 / losses 44 over n=1351

### coherent86_private_alpha0p5_minus_chck82_anchor

Aggregate: net -21 from 1560 gains / 1581 losses over 170722 common discrete items; retention fraction 0.9839456528361664; mean payload Δ +0.2521.

| column | payload Δ | gains | losses | net | retention | best groups | worst groups |
|---|---:|---:|---:|---:|---:|---|---|
| BLiMP | 0.048715963480148616 | 385 | 352 | 33 | 0.9914014216967535 | wh_island +17/960, superlative_quantifiers_2 +14/986, principle_A_domain_1 +11/914 | npi_present_2 -21/914, npi_present_1 -16/909, animate_subject_trans -8/923 |
| Supplement | 0.2121887437997998 | 15 | 27 | -12 | 0.9930232558139535 | qa_congruence_easy +1/64, qa_congruence_tricky +0/165, turn_taking +0/280 | subject_aux_inversion -11/3867, hypernym -2/842, qa_congruence_tricky +0/165 |
| EWoK | -0.055453325532759834 | 83 | 73 | 10 | 0.9808096740273397 | social-interactions +3/294, social-relations +14/1548, material-dynamics +2/770 | quantitative-properties -3/314, physical-dynamics -1/120, physical-interactions -3/556 |
| Entity | -0.08404193029877405 | 33 | 38 | -5 | 0.9799048122686409 | ambiref_2_ops +4/413, move_contents_4_ops +3/353, move_contents_1_ops +2/437 | move_contents_3_ops -4/406, ambiref_5_ops -1/123, ambiref_0_ops -3/508 |
| COMPS | -0.08117509443596305 | 1041 | 1091 | -50 | 0.9772233820459291 | wugs_dist_in_between +29/13896, base -6/49340, wugs -24/13896 | wugs_dist_before -49/13896, wugs -24/13896, base -6/49340 |
| GlobalPIQA | 1.4723300970873794 | 3 | 0 | 3 | 1.0 | GlobalPIQA_parallel +2/103, GlobalPIQA_nonparallel +1/100 | GlobalPIQA_nonparallel +1/100, GlobalPIQA_parallel +2/103 |

Focus nets:
- EWoK_relation_like: net 10 from gains 63 / losses 53 over n=5408
- Entity_high_operation: net -4 from gains 14 / losses 18 over n=2748
- Supplement_selected: net -1 from gains 7 / losses 8 over n=1351

### coherent86_private_alpha0p75_minus_chck82_anchor

Aggregate: net -86 from 2332 gains / 2418 losses over 170722 common discrete items; retention fraction 0.9754462925729604; mean payload Δ +0.2563.

| column | payload Δ | gains | losses | net | retention | best groups | worst groups |
|---|---:|---:|---:|---:|---:|---|---|
| BLiMP | 0.01871596348014748 | 565 | 545 | 20 | 0.9866868602975304 | wh_island +23/960, principle_A_domain_1 +20/914, superlative_quantifiers_2 +20/986 | npi_present_2 -42/914, npi_present_1 -20/909, ellipsis_n_bar_1 -10/802 |
| Supplement | 0.7021887437998018 | 27 | 37 | -10 | 0.9904392764857881 | qa_congruence_easy +2/64, qa_congruence_tricky +1/165, hypernym +1/842 | subject_aux_inversion -14/3867, turn_taking +0/280, hypernym +1/842 |
| EWoK | -0.03545332553275671 | 117 | 113 | 4 | 0.9702944269190326 | material-properties +2/170, social-interactions +2/294, social-relations +9/1548 | social-properties -4/328, physical-dynamics -1/120, spatial-relations -2/490 |
| Entity | 0.00595806970122581 | 50 | 54 | -4 | 0.9714436805922793 | regular_5_ops +1/94, ambiref_2_ops +4/413, move_contents_5_ops +1/116 | move_contents_3_ops -5/406, regular_2_ops -4/405, ambiref_0_ops -5/508 |
| COMPS | -0.14117509443596532 | 1570 | 1668 | -98 | 0.9651774530271399 | wugs_dist_in_between +23/13896, base -25/49340, wugs -22/13896 | wugs_dist_before -74/13896, wugs -22/13896, base -25/49340 |
| GlobalPIQA | 0.98733009708738 | 3 | 1 | 2 | 0.9868421052631579 | GlobalPIQA_nonparallel +1/100, GlobalPIQA_parallel +1/103 | GlobalPIQA_parallel +1/103, GlobalPIQA_nonparallel +1/100 |

Focus nets:
- EWoK_relation_like: net 5 from gains 88 / losses 83 over n=5408
- Entity_high_operation: net -3 from gains 21 / losses 24 over n=2748
- Supplement_selected: net 4 from gains 14 / losses 10 over n=1351

### coherent86_private_alpha1_minus_ordinary86_backbone

Aggregate: net -233 from 4796 gains / 5029 losses over 170722 common discrete items; retention fraction 0.9489938739908312; mean payload Δ +0.3800.

| column | payload Δ | gains | losses | net | retention | best groups | worst groups |
|---|---:|---:|---:|---:|---:|---|---|
| BLiMP | 0.04999999999999716 | 1131 | 1089 | 42 | 0.973390348197923 | wh_island +43/960, tough_vs_raising_1 +28/948, principle_A_c_command +27/946 | tough_vs_raising_2 -32/920, wh_questions_subject_gap -25/898, wh_vs_that_with_gap -24/919 |
| Supplement | 0.9699999999999989 | 65 | 86 | -21 | 0.9778179004384834 | qa_congruence_easy +2/64, qa_congruence_tricky +4/165, hypernym +0/842 | subject_aux_inversion -27/3867, hypernym +0/842, turn_taking +0/280 |
| EWoK | -0.23000000000000398 | 218 | 245 | -27 | 0.9360146252285192 | social-interactions +4/294, material-properties +2/170, physical-interactions +5/556 | spatial-relations -8/490, material-dynamics -10/770, social-relations -15/1548 |
| Entity | -0.13999999999999702 | 131 | 145 | -14 | 0.924163179916318 | move_contents_5_ops +5/116, regular_3_ops +5/425, ambiref_3_ops +3/409 | regular_4_ops -9/388, move_contents_4_ops -6/353, ambiref_5_ops -2/123 |
| COMPS | -0.29999999999999716 | 3242 | 3459 | -217 | 0.9279074614422677 | wugs_dist_before +17/13896, base -71/49340, wugs -44/13896 | wugs_dist_in_between -119/13896, wugs -44/13896, base -71/49340 |
| GlobalPIQA | 1.9299999999999997 | 9 | 5 | 4 | 0.9315068493150684 | GlobalPIQA_parallel +5/103, GlobalPIQA_nonparallel -1/100 | GlobalPIQA_nonparallel -1/100, GlobalPIQA_parallel +5/103 |

Focus nets:
- EWoK_relation_like: net -32 from gains 150 / losses 182 over n=5408
- Entity_high_operation: net -10 from gains 57 / losses 67 over n=2748
- Supplement_selected: net 6 from gains 28 / losses 22 over n=1351

### coherent86_private_alpha1_minus_shuffled86_private

Aggregate: net -1113 from 14551 gains / 15664 losses over 170722 common discrete items; retention fraction 0.8425348827857976; mean payload Δ +0.1850.

| column | payload Δ | gains | losses | net | retention | best groups | worst groups |
|---|---:|---:|---:|---:|---:|---|---|
| BLiMP | -0.710000000000008 | 3148 | 3569 | -421 | 0.9137672755388035 | adjunct_island +173/928, matrix_question_npi_licensor_present +94/929, existential_there_quantifiers_2 +90/911 | npi_present_2 -137/914, only_npi_scope -125/837, wh_questions_object_gap -109/859 |
| Supplement | 3.8399999999999963 | 227 | 163 | 64 | 0.9570147679324894 | qa_congruence_tricky +19/165, qa_congruence_easy +4/64, hypernym +18/842 | turn_taking -4/280, subject_aux_inversion +27/3867, hypernym +18/842 |
| EWoK | -1.5300000000000011 | 692 | 726 | -34 | 0.8107403545359749 | physical-relations +17/818, social-interactions +3/294, material-dynamics +7/770 | physical-dynamics -14/120, physical-interactions -16/556, spatial-relations -11/490 |
| Entity | 1.6700000000000017 | 292 | 212 | 80 | 0.8833883388338833 | move_contents_5_ops +11/116, regular_5_ops +6/94, ambiref_2_ops +19/413 | ambiref_5_ops -2/123, move_contents_3_ops -4/406, regular_0_ops -2/517 |
| COMPS | -0.6599999999999966 | 10181 | 10980 | -799 | 0.7738972859437421 | wugs_dist_in_between +1455/13896, wugs -126/13896, base -603/49340 | wugs_dist_before -1525/13896, base -603/49340, wugs -126/13896 |
| GlobalPIQA | -1.5 | 11 | 14 | -3 | 0.825 | GlobalPIQA_parallel +0/103, GlobalPIQA_nonparallel -3/100 | GlobalPIQA_nonparallel -3/100, GlobalPIQA_parallel +0/103 |

Focus nets:
- EWoK_relation_like: net -52 from gains 510 / losses 562 over n=5408
- Entity_high_operation: net 46 from gains 127 / losses 81 over n=2748
- Supplement_selected: net 37 from gains 102 / losses 65 over n=1351

### coherent86_private_alpha1_minus_spanbreak86_private

Aggregate: net 376 from 8768 gains / 8392 losses over 170722 common discrete items; retention fraction 0.9143559859981426; mean payload Δ +1.1367.

| column | payload Δ | gains | losses | net | retention | best groups | worst groups |
|---|---:|---:|---:|---:|---:|---|---|
| BLiMP | 0.3399999999999892 | 1882 | 1670 | 212 | 0.9590234327076432 | matrix_question_npi_licensor_present +75/929, only_npi_scope +34/837, superlative_quantifiers_2 +34/986 | only_npi_licensor_present -63/882, existential_there_quantifiers_2 -51/911, left_branch_island_echo_question -39/947 |
| Supplement | 1.2899999999999991 | 135 | 106 | 29 | 0.972302064280115 | qa_congruence_tricky +6/165, hypernym +14/842, qa_congruence_easy +1/64 | turn_taking -2/280, subject_aux_inversion +10/3867, qa_congruence_easy +1/64 |
| EWoK | 0.5399999999999991 | 424 | 414 | 10 | 0.8908227848101266 | physical-dynamics +7/120, material-properties +6/170, social-interactions +5/294 | social-properties -9/328, quantitative-properties -5/314, spatial-relations -4/490 |
| Entity | 4.200000000000003 | 636 | 480 | 156 | 0.7244546498277842 | move_contents_5_ops +31/116, move_contents_4_ops +47/353, ambiref_4_ops +47/434 | regular_0_ops -75/517, move_contents_0_ops -57/516, ambiref_0_ops -51/508 |
| COMPS | -0.04999999999999716 | 5686 | 5718 | -32 | 0.8803640548174495 | wugs_dist_in_between +498/13896, base -5/49340, wugs -13/13896 | wugs_dist_before -512/13896, wugs -13/13896, base -5/49340 |
| GlobalPIQA | 0.5 | 5 | 4 | 1 | 0.9473684210526315 | GlobalPIQA_nonparallel +1/100, GlobalPIQA_parallel +0/103 | GlobalPIQA_parallel +0/103, GlobalPIQA_nonparallel +1/100 |

Focus nets:
- EWoK_relation_like: net 6 from gains 314 / losses 308 over n=5408
- Entity_high_operation: net 266 from gains 373 / losses 107 over n=2748
- Supplement_selected: net 19 from gains 44 / losses 25 over n=1351

### coherent86_private_alpha0p5_minus_coherent86_private_alpha1

Aggregate: net 94 from 1663 gains / 1569 losses over 170722 common discrete items; retention fraction 0.984048880168356; mean payload Δ +0.0842.

| column | payload Δ | gains | losses | net | retention | best groups | worst groups |
|---|---:|---:|---:|---:|---:|---|---|
| BLiMP | 0.020000000000010232 | 383 | 380 | 3 | 0.9907242414626407 | npi_present_2 +36/914, ellipsis_n_bar_1 +15/802, npi_present_1 +15/909 | existential_there_quantifiers_2 -15/911, superlative_quantifiers_2 -16/986, principle_A_domain_1 -12/914 |
| Supplement | -0.5 | 24 | 22 | 2 | 0.9942946058091287 | turn_taking +1/280, subject_aux_inversion +6/3867, hypernym -2/842 | qa_congruence_easy -1/64, qa_congruence_tricky -2/165, hypernym -2/842 |
| EWoK | 0.09000000000000341 | 77 | 65 | 12 | 0.9829037348763808 | social-properties +3/328, physical-dynamics +1/120, spatial-relations +2/490 | physical-interactions -7/556, material-properties -2/170, quantitative-properties +0/314 |
| Entity | -0.21000000000000085 | 28 | 40 | -12 | 0.9789251844046365 | regular_5_ops +1/94, regular_2_ops +2/405, regular_0_ops +2/517 | move_contents_5_ops -3/116, regular_1_ops -5/409, move_contents_0_ops -5/516 |
| COMPS | 0.11999999999999744 | 1149 | 1062 | 87 | 0.9777652157527793 | wugs_dist_before +36/13896, wugs_dist_in_between +17/13896, base +32/49340 | wugs +2/13896, base +32/49340, wugs_dist_in_between +17/13896 |
| GlobalPIQA | 0.9849999999999994 | 2 | 0 | 2 | 1.0 | GlobalPIQA_nonparallel +1/100, GlobalPIQA_parallel +1/103 | GlobalPIQA_parallel +1/103, GlobalPIQA_nonparallel +1/100 |

Focus nets:
- EWoK_relation_like: net 4 from gains 55 / losses 51 over n=5408
- Entity_high_operation: net -5 from gains 10 / losses 15 over n=2748
- Supplement_selected: net -4 from gains 7 / losses 11 over n=1351

### coherent86_private_alpha0p75_minus_coherent86_private_alpha1

Aggregate: net 29 from 827 gains / 798 losses over 170722 common discrete items; retention fraction 0.991887193355225; mean payload Δ +0.0883.

| column | payload Δ | gains | losses | net | retention | best groups | worst groups |
|---|---:|---:|---:|---:|---:|---|---|
| BLiMP | -0.009999999999990905 | 190 | 200 | -10 | 0.9951180218224425 | npi_present_2 +15/914, npi_present_1 +11/909, left_branch_island_simple_question +5/951 | existential_there_quantifiers_2 -11/911, superlative_quantifiers_2 -10/986, principle_A_domain_3 -8/941 |
| Supplement | -0.00999999999999801 | 14 | 10 | 4 | 0.9974066390041494 | turn_taking +1/280, hypernym +1/842, subject_aux_inversion +3/3867 | qa_congruence_tricky -1/165, qa_congruence_easy +0/64, subject_aux_inversion +3/3867 |
| EWoK | 0.11000000000000654 | 37 | 31 | 6 | 0.9918463966333508 | physical-dynamics +1/120, quantitative-properties +2/314, physical-relations +3/818 | physical-interactions -4/556, material-dynamics -2/770, social-relations -2/1548 |
| Entity | -0.120000000000001 | 13 | 24 | -11 | 0.9873551106427819 | regular_5_ops +2/94, regular_0_ops +2/517, move_contents_4_ops +1/353 | move_contents_5_ops -2/116, regular_4_ops -3/388, move_contents_0_ops -3/516 |
| COMPS | 0.05999999999999517 | 572 | 533 | 39 | 0.9888407344597282 | wugs_dist_before +11/13896, wugs_dist_in_between +11/13896, wugs +4/13896 | base +13/49340, wugs +4/13896, wugs_dist_before +11/13896 |
| GlobalPIQA | 0.5 | 1 | 0 | 1 | 1.0 | GlobalPIQA_nonparallel +1/100, GlobalPIQA_parallel +0/103 | GlobalPIQA_parallel +0/103, GlobalPIQA_nonparallel +1/100 |

Focus nets:
- EWoK_relation_like: net -1 from gains 25 / losses 26 over n=5408
- Entity_high_operation: net -4 from gains 5 / losses 9 over n=2748
- Supplement_selected: net 1 from gains 5 / losses 4 over n=1351

### coherent86_private_alpha0p5_minus_ordinary86_backbone

Aggregate: net -139 from 4824 gains / 4963 losses over 170722 common discrete items; retention fraction 0.9496632723437056; mean payload Δ +0.4642.

| column | payload Δ | gains | losses | net | retention | best groups | worst groups |
|---|---:|---:|---:|---:|---:|---|---|
| BLiMP | 0.07000000000000739 | 1121 | 1076 | 45 | 0.9737080024434942 | wh_island +32/960, tough_vs_raising_1 +26/948, only_npi_licensor_present +23/882 | tough_vs_raising_2 -31/920, wh_questions_subject_gap -24/898, intransitive -21/868 |
| Supplement | 0.46999999999999886 | 62 | 81 | -19 | 0.9791075573897343 | qa_congruence_easy +1/64, qa_congruence_tricky +2/165, turn_taking +1/280 | subject_aux_inversion -21/3867, hypernym -2/842, turn_taking +1/280 |
| EWoK | -0.14000000000000057 | 227 | 242 | -15 | 0.9367981196134761 | social-interactions +5/294, social-properties +4/328, agent-properties +13/2210 | spatial-relations -6/490, material-dynamics -9/770, quantitative-properties -3/314 |
| Entity | -0.34999999999999787 | 121 | 147 | -26 | 0.9231171548117155 | move_contents_5_ops +2/116, ambiref_3_ops +3/409, regular_3_ops +3/425 | regular_4_ops -11/388, ambiref_5_ops -2/123, ambiref_4_ops -7/434 |
| COMPS | -0.17999999999999972 | 3283 | 3413 | -130 | 0.9288661942476032 | wugs_dist_before +53/13896, base -39/49340, wugs -42/13896 | wugs_dist_in_between -102/13896, wugs -42/13896, base -39/49340 |
| GlobalPIQA | 2.914999999999999 | 10 | 4 | 6 | 0.9452054794520548 | GlobalPIQA_parallel +6/103, GlobalPIQA_nonparallel +0/100 | GlobalPIQA_nonparallel +0/100, GlobalPIQA_parallel +6/103 |

Focus nets:
- EWoK_relation_like: net -28 from gains 156 / losses 184 over n=5408
- Entity_high_operation: net -15 from gains 49 / losses 64 over n=2748
- Supplement_selected: net 2 from gains 25 / losses 23 over n=1351

### coherent86_private_alpha0p75_minus_ordinary86_backbone

Aggregate: net -204 from 4763 gains / 4967 losses over 170722 common discrete items; retention fraction 0.9496227027465617; mean payload Δ +0.4683.

| column | payload Δ | gains | losses | net | retention | best groups | worst groups |
|---|---:|---:|---:|---:|---:|---|---|
| BLiMP | 0.04000000000000625 | 1111 | 1079 | 32 | 0.9736346976175931 | wh_island +38/960, tough_vs_raising_1 +31/948, principle_A_c_command +24/946 | tough_vs_raising_2 -32/920, wh_questions_subject_gap -24/898, intransitive -20/868 |
| Supplement | 0.9600000000000009 | 64 | 81 | -17 | 0.9791075573897343 | qa_congruence_easy +2/64, qa_congruence_tricky +3/165, turn_taking +1/280 | subject_aux_inversion -24/3867, hypernym +1/842, turn_taking +1/280 |
| EWoK | -0.11999999999999744 | 221 | 242 | -21 | 0.9367981196134761 | social-interactions +4/294, material-properties +2/170, agent-properties +12/2210 | material-dynamics -12/770, spatial-relations -7/490, social-relations -17/1548 |
| Entity | -0.259999999999998 | 128 | 153 | -25 | 0.9199790794979079 | move_contents_5_ops +3/116, regular_3_ops +5/425, regular_5_ops +1/94 | regular_4_ops -12/388, ambiref_5_ops -2/123, ambiref_4_ops -7/434 |
| COMPS | -0.240000000000002 | 3230 | 3408 | -178 | 0.9289704043351397 | wugs_dist_before +28/13896, base -58/49340, wugs -40/13896 | wugs_dist_in_between -108/13896, wugs -40/13896, base -58/49340 |
| GlobalPIQA | 2.4299999999999997 | 9 | 4 | 5 | 0.9452054794520548 | GlobalPIQA_parallel +5/103, GlobalPIQA_nonparallel +0/100 | GlobalPIQA_nonparallel +0/100, GlobalPIQA_parallel +5/103 |

Focus nets:
- EWoK_relation_like: net -33 from gains 148 / losses 181 over n=5408
- Entity_high_operation: net -14 from gains 53 / losses 67 over n=2748
- Supplement_selected: net 7 from gains 26 / losses 19 over n=1351

## Scientific reading

- coherent86 alpha1 cheap7 exceeds ordinary86 by +0.335714; this keeps ordinary continued backbone training from explaining the endpoint edge by itself.
- coherent86 alpha1 changes 6347 discrete common items vs the anchor and has net -115; this is a redistribution surface, not uniform added competence.

JSON: `experiments/archive/frontier_consolidation/data/private_scale_panel_analysis/private_scale_panel_analysis.json`

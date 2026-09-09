# dense focus official and mechanism state fast-screen vs partial official movement

Fast transition: `experiments/archive/functional_learning/data/dense_fast_transition_compare/dense_vs_coherent86_fast_official_transition.json`
Official partial transition: `experiments/archive/functional_learning/data/dense_official_partial_transition/dense_vs_coherent86_official_partial_official_transition.json`

| column | fast score delta | full score delta | full-fast | fast net item | full net item | common subtasks | Pearson | Spearman | nonzero same-sign |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BLiMP | -0.552239 | -0.450450 | +0.101788 | -74 | -272 | 55 | 0.848610 | 0.805050 | 0.925926 |
| Supplement | -0.400000 | -0.597876 | -0.197876 | -1 | -5 | 5 | 0.098346 | 0.000000 | 0.333333 |
| EWoK | +0.363636 | -0.097138 | -0.460774 | 4 | 13 | 11 | 0.606415 | 0.344279 | 0.555556 |
| Entity | +0.658300 | +1.044733 | +0.386433 | 7 | 42 | 6 | 1.000000 | 1.000000 | 1.000000 |

## BLiMP

### Largest fast/full delta disagreements

| subtask | fast delta | full delta | full-fast | fast n | full n |
|---|---:|---:|---:|---:|---:|
| wh_questions_object_gap | +5.500000 | +1.746217 | -3.753783 | 200 | 859 |
| left_branch_island_echo_question | -4.000000 | -1.900739 | +2.099261 | 200 | 947 |
| coordinate_structure_constraint_object_extraction | +0.500000 | -1.580611 | -2.080611 | 200 | 949 |
| transitive | -3.000000 | -1.036866 | +1.963134 | 200 | 868 |
| coordinate_structure_constraint_complex_left_branch | -2.500000 | -0.662252 | +1.837748 | 200 | 906 |
| inchoative | -2.000000 | -0.233918 | +1.766082 | 200 | 855 |
| superlative_quantifiers_2 | -4.500000 | -2.839757 | +1.660243 | 200 | 986 |
| existential_there_quantifiers_2 | +4.500000 | +3.073546 | -1.426454 | 200 | 911 |

### Sign reversals

| subtask | fast delta | full delta | full-fast | fast n | full n |
|---|---:|---:|---:|---:|---:|
| coordinate_structure_constraint_object_extraction | +0.500000 | -1.580611 | -2.080611 | 200 | 949 |
| wh_questions_subject_gap_long_distance | +1.000000 | -0.350058 | -1.350058 | 200 | 857 |
| tough_vs_raising_1 | +0.500000 | -0.843882 | -1.343882 | 200 | 948 |
| determiner_noun_agreement_irregular_2 | +0.500000 | -0.609756 | -1.109756 | 200 | 820 |

## Supplement

### Largest fast/full delta disagreements

| subtask | fast delta | full delta | full-fast | fast n | full n |
|---|---:|---:|---:|---:|---:|
| qa_congruence_tricky | +2.000000 | -0.606061 | -2.606061 | 50 | 165 |
| subject_aux_inversion | -2.000000 | +0.129299 | +2.129299 | 50 | 3867 |
| hypernym | +0.000000 | -0.950119 | -0.950119 | 50 | 842 |
| qa_congruence_easy | -2.000000 | -1.562500 | +0.437500 | 50 | 64 |
| turn_taking | +0.000000 | +0.000000 | +0.000000 | 50 | 280 |

### Sign reversals

| subtask | fast delta | full delta | full-fast | fast n | full n |
|---|---:|---:|---:|---:|---:|
| qa_congruence_tricky | +2.000000 | -0.606061 | -2.606061 | 50 | 165 |
| subject_aux_inversion | -2.000000 | +0.129299 | +2.129299 | 50 | 3867 |

## EWoK

### Largest fast/full delta disagreements

| subtask | fast delta | full delta | full-fast | fast n | full n |
|---|---:|---:|---:|---:|---:|
| spatial-relations | +4.000000 | +0.204082 | -3.795918 | 100 | 490 |
| social-relations | -2.000000 | +0.322997 | +2.322997 | 100 | 1548 |
| agent-properties | -2.000000 | -0.226244 | +1.773756 | 100 | 2210 |
| material-properties | +1.000000 | -0.588235 | -1.588235 | 100 | 170 |
| social-properties | +0.000000 | -1.524390 | -1.524390 | 100 | 328 |
| physical-relations | +1.000000 | -0.488998 | -1.488998 | 100 | 818 |
| quantitative-properties | +1.000000 | -0.318471 | -1.318471 | 100 | 314 |
| social-interactions | +0.000000 | +0.680272 | +0.680272 | 100 | 294 |

### Sign reversals

| subtask | fast delta | full delta | full-fast | fast n | full n |
|---|---:|---:|---:|---:|---:|
| social-relations | -2.000000 | +0.322997 | +2.322997 | 100 | 1548 |
| material-properties | +1.000000 | -0.588235 | -1.588235 | 100 | 170 |
| physical-relations | +1.000000 | -0.488998 | -1.488998 | 100 | 818 |
| quantitative-properties | +1.000000 | -0.318471 | -1.318471 | 100 | 314 |

## Entity

### Largest fast/full delta disagreements

| subtask | fast delta | full delta | full-fast | fast n | full n |
|---|---:|---:|---:|---:|---:|
| regular_0_ops | -4.835590 | -4.835590 | +0.000000 | 517 | 517 |
| regular_1_ops | +1.711491 | +1.711491 | +0.000000 | 409 | 409 |
| regular_2_ops | +2.469136 | +2.469136 | +0.000000 | 405 | 405 |
| regular_3_ops | +0.705882 | +0.705882 | +0.000000 | 425 | 425 |
| regular_4_ops | +2.835052 | +2.835052 | +0.000000 | 388 | 388 |
| regular_5_ops | +1.063830 | +1.063830 | +0.000000 | 94 | 94 |

## Interpretation

Fast_eval and full_eval use different item sets/sizes. Agreement supports using fast movement as a rough direction; disagreement means the official payload must dominate route judgment. This file does not infer unfinished columns.

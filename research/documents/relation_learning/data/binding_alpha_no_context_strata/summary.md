# earlier analysis binding alpha / no-context / conflict strata

Checkpoint: `experiments/archive/relation_learning/data/chck82_binding_candidate/scale_0.75/checkpoint`

## All-pair alpha scan
| alpha | full joint | full A | full B | full both-wrong | no-context joint | no-context A | no-context B | no-context both-wrong |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 5 | 78 | 124 | 3 | 10 | 134 | 73 | 3 |
| 0.25 | 18 | 69 | 146 | 3 | 15 | 130 | 80 | 5 |
| 0.5 | 32 | 70 | 160 | 2 | 16 | 123 | 81 | 12 |
| 0.75 | 71 | 90 | 181 | 0 | 15 | 121 | 80 | 14 |
| 1 | 91 | 103 | 188 | 0 | 18 | 123 | 81 | 14 |
| 1.25 | 81 | 95 | 185 | 1 | 16 | 120 | 77 | 19 |

## Conflict strata (selected)
| alpha | mode | group | n | joint | gated_frac | A | B | both_wrong |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 0 | full_context | all_pairs | 200 | 5 | 0.025 | 78 | 124 | 3 |
| 0 | full_context | genuine_both_source_low_answer_overlap | 114 | 3 | 0.026 | 49 | 66 | 2 |
| 0 | full_context | position_unchanged_first | 160 | 5 | 0.031 | 61 | 101 | 3 |
| 0 | full_context | position_updated_first | 29 | 0 | 0.000 | 15 | 14 | 0 |
| 0 | full_context | updated_entity_in_source | 119 | 3 | 0.025 | 50 | 70 | 2 |
| 0 | full_context | updated_entity_not_in_source | 81 | 2 | 0.025 | 28 | 54 | 1 |
| 0 | full_context | weak_or_location_separable | 86 | 2 | 0.023 | 29 | 58 | 1 |
| 0 | no_context | all_pairs | 200 | 10 | 0.050 | 134 | 73 | 3 |
| 0 | no_context | genuine_both_source_low_answer_overlap | 114 | 6 | 0.053 | 80 | 39 | 1 |
| 0 | no_context | position_unchanged_first | 160 | 8 | 0.050 | 106 | 59 | 3 |
| 0 | no_context | position_updated_first | 29 | 2 | 0.069 | 23 | 8 | 0 |
| 0 | no_context | updated_entity_in_source | 119 | 6 | 0.050 | 82 | 42 | 1 |
| 0 | no_context | updated_entity_not_in_source | 81 | 4 | 0.049 | 52 | 31 | 2 |
| 0 | no_context | weak_or_location_separable | 86 | 4 | 0.047 | 54 | 34 | 2 |
| 0.25 | full_context | all_pairs | 200 | 18 | 0.090 | 69 | 146 | 3 |
| 0.25 | full_context | genuine_both_source_low_answer_overlap | 114 | 9 | 0.079 | 41 | 81 | 1 |
| 0.25 | full_context | position_unchanged_first | 160 | 15 | 0.094 | 57 | 115 | 3 |
| 0.25 | full_context | position_updated_first | 29 | 2 | 0.069 | 10 | 21 | 0 |
| 0.25 | full_context | updated_entity_in_source | 119 | 10 | 0.084 | 42 | 85 | 2 |
| 0.25 | full_context | updated_entity_not_in_source | 81 | 8 | 0.099 | 27 | 61 | 1 |
| 0.25 | full_context | weak_or_location_separable | 86 | 9 | 0.105 | 28 | 65 | 2 |
| 0.25 | no_context | all_pairs | 200 | 15 | 0.075 | 130 | 80 | 5 |
| 0.25 | no_context | genuine_both_source_low_answer_overlap | 114 | 7 | 0.061 | 76 | 43 | 2 |
| 0.25 | no_context | position_unchanged_first | 160 | 14 | 0.087 | 105 | 64 | 5 |
| 0.25 | no_context | position_updated_first | 29 | 1 | 0.034 | 19 | 11 | 0 |
| 0.25 | no_context | updated_entity_in_source | 119 | 8 | 0.067 | 79 | 46 | 2 |
| 0.25 | no_context | updated_entity_not_in_source | 81 | 7 | 0.086 | 51 | 34 | 3 |
| 0.25 | no_context | weak_or_location_separable | 86 | 8 | 0.093 | 54 | 37 | 3 |
| 0.5 | full_context | all_pairs | 200 | 32 | 0.160 | 70 | 160 | 2 |
| 0.5 | full_context | genuine_both_source_low_answer_overlap | 114 | 15 | 0.132 | 39 | 89 | 1 |
| 0.5 | full_context | position_unchanged_first | 160 | 30 | 0.188 | 62 | 126 | 2 |
| 0.5 | full_context | position_updated_first | 29 | 1 | 0.034 | 6 | 24 | 0 |
| 0.5 | full_context | updated_entity_in_source | 119 | 16 | 0.134 | 41 | 93 | 1 |
| 0.5 | full_context | updated_entity_not_in_source | 81 | 16 | 0.198 | 29 | 67 | 1 |
| 0.5 | full_context | weak_or_location_separable | 86 | 17 | 0.198 | 31 | 71 | 1 |
| 0.5 | no_context | all_pairs | 200 | 16 | 0.080 | 123 | 81 | 12 |
| 0.5 | no_context | genuine_both_source_low_answer_overlap | 114 | 8 | 0.070 | 71 | 44 | 7 |
| 0.5 | no_context | position_unchanged_first | 160 | 16 | 0.100 | 101 | 66 | 9 |
| 0.5 | no_context | position_updated_first | 29 | 0 | 0.000 | 16 | 11 | 2 |
| 0.5 | no_context | updated_entity_in_source | 119 | 9 | 0.076 | 74 | 47 | 7 |
| 0.5 | no_context | updated_entity_not_in_source | 81 | 7 | 0.086 | 49 | 34 | 5 |
| 0.5 | no_context | weak_or_location_separable | 86 | 8 | 0.093 | 52 | 37 | 5 |
| 0.75 | full_context | all_pairs | 200 | 71 | 0.355 | 90 | 181 | 0 |
| 0.75 | full_context | genuine_both_source_low_answer_overlap | 114 | 39 | 0.342 | 51 | 102 | 0 |
| 0.75 | full_context | position_unchanged_first | 160 | 59 | 0.369 | 77 | 142 | 0 |
| 0.75 | full_context | position_updated_first | 29 | 10 | 0.345 | 11 | 28 | 0 |
| 0.75 | full_context | updated_entity_in_source | 119 | 40 | 0.336 | 53 | 106 | 0 |
| 0.75 | full_context | updated_entity_not_in_source | 81 | 31 | 0.383 | 37 | 75 | 0 |
| 0.75 | full_context | weak_or_location_separable | 86 | 32 | 0.372 | 39 | 79 | 0 |
| 0.75 | no_context | all_pairs | 200 | 15 | 0.075 | 121 | 80 | 14 |
| 0.75 | no_context | genuine_both_source_low_answer_overlap | 114 | 10 | 0.088 | 71 | 45 | 8 |
| 0.75 | no_context | position_unchanged_first | 160 | 10 | 0.062 | 99 | 63 | 8 |
| 0.75 | no_context | position_updated_first | 29 | 3 | 0.103 | 15 | 12 | 5 |
| 0.75 | no_context | updated_entity_in_source | 119 | 12 | 0.101 | 74 | 49 | 8 |
| 0.75 | no_context | updated_entity_not_in_source | 81 | 3 | 0.037 | 47 | 31 | 6 |
| 0.75 | no_context | weak_or_location_separable | 86 | 5 | 0.058 | 50 | 35 | 6 |
| 1 | full_context | all_pairs | 200 | 91 | 0.455 | 103 | 188 | 0 |
| 1 | full_context | genuine_both_source_low_answer_overlap | 114 | 47 | 0.412 | 54 | 107 | 0 |
| 1 | full_context | position_unchanged_first | 160 | 76 | 0.475 | 85 | 151 | 0 |
| 1 | full_context | position_updated_first | 29 | 11 | 0.379 | 13 | 27 | 0 |
| 1 | full_context | updated_entity_in_source | 119 | 49 | 0.412 | 58 | 110 | 0 |
| 1 | full_context | updated_entity_not_in_source | 81 | 42 | 0.519 | 45 | 78 | 0 |
| 1 | full_context | weak_or_location_separable | 86 | 44 | 0.512 | 49 | 81 | 0 |
| 1 | no_context | all_pairs | 200 | 18 | 0.090 | 123 | 81 | 14 |
| 1 | no_context | genuine_both_source_low_answer_overlap | 114 | 10 | 0.088 | 72 | 45 | 7 |
| 1 | no_context | position_unchanged_first | 160 | 16 | 0.100 | 102 | 67 | 7 |
| 1 | no_context | position_updated_first | 29 | 1 | 0.034 | 15 | 10 | 5 |
| 1 | no_context | updated_entity_in_source | 119 | 12 | 0.101 | 75 | 49 | 7 |
| 1 | no_context | updated_entity_not_in_source | 81 | 6 | 0.074 | 48 | 32 | 7 |
| 1 | no_context | weak_or_location_separable | 86 | 8 | 0.093 | 51 | 36 | 7 |
| 1.25 | full_context | all_pairs | 200 | 81 | 0.405 | 95 | 185 | 1 |
| 1.25 | full_context | genuine_both_source_low_answer_overlap | 114 | 42 | 0.368 | 52 | 104 | 0 |
| 1.25 | full_context | position_unchanged_first | 160 | 67 | 0.419 | 79 | 147 | 1 |
| 1.25 | full_context | position_updated_first | 29 | 9 | 0.310 | 11 | 27 | 0 |
| 1.25 | full_context | updated_entity_in_source | 119 | 45 | 0.378 | 55 | 108 | 1 |
| 1.25 | full_context | updated_entity_not_in_source | 81 | 36 | 0.444 | 40 | 77 | 0 |
| 1.25 | full_context | weak_or_location_separable | 86 | 39 | 0.453 | 43 | 81 | 1 |
| 1.25 | no_context | all_pairs | 200 | 16 | 0.080 | 120 | 77 | 19 |
| 1.25 | no_context | genuine_both_source_low_answer_overlap | 114 | 9 | 0.079 | 67 | 46 | 10 |
| 1.25 | no_context | position_unchanged_first | 160 | 13 | 0.081 | 99 | 60 | 14 |
| 1.25 | no_context | position_updated_first | 29 | 2 | 0.069 | 14 | 13 | 4 |
| 1.25 | no_context | updated_entity_in_source | 119 | 11 | 0.092 | 70 | 50 | 10 |
| 1.25 | no_context | updated_entity_not_in_source | 81 | 5 | 0.062 | 50 | 27 | 9 |
| 1.25 | no_context | weak_or_location_separable | 86 | 7 | 0.081 | 53 | 31 | 9 |

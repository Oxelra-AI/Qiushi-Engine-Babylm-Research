# earlier analysis binding alpha / no-context / conflict strata

Checkpoint: `experiments/archive/relation_learning/data/chck82_binding_candidate_ep25/scale_0.75/checkpoint`

## All-pair alpha scan
| alpha | full joint | full A | full B | full both-wrong | no-context joint | no-context A | no-context B | no-context both-wrong |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 5 | 78 | 124 | 3 | 10 | 134 | 73 | 3 |
| 0.25 | 15 | 72 | 141 | 2 | 17 | 131 | 79 | 7 |
| 0.5 | 46 | 88 | 158 | 0 | 10 | 125 | 78 | 7 |
| 0.75 | 83 | 104 | 179 | 0 | 13 | 115 | 86 | 12 |
| 1 | 92 | 106 | 184 | 2 | 21 | 118 | 94 | 9 |
| 1.25 | 84 | 107 | 176 | 1 | 15 | 109 | 92 | 14 |

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
| 0.25 | full_context | all_pairs | 200 | 15 | 0.075 | 72 | 141 | 2 |
| 0.25 | full_context | genuine_both_source_low_answer_overlap | 114 | 8 | 0.070 | 42 | 79 | 1 |
| 0.25 | full_context | position_unchanged_first | 160 | 10 | 0.062 | 58 | 110 | 2 |
| 0.25 | full_context | position_updated_first | 29 | 4 | 0.138 | 12 | 21 | 0 |
| 0.25 | full_context | updated_entity_in_source | 119 | 9 | 0.076 | 43 | 83 | 2 |
| 0.25 | full_context | updated_entity_not_in_source | 81 | 6 | 0.074 | 29 | 58 | 0 |
| 0.25 | full_context | weak_or_location_separable | 86 | 7 | 0.081 | 30 | 62 | 1 |
| 0.25 | no_context | all_pairs | 200 | 17 | 0.085 | 131 | 79 | 7 |
| 0.25 | no_context | genuine_both_source_low_answer_overlap | 114 | 7 | 0.061 | 76 | 44 | 1 |
| 0.25 | no_context | position_unchanged_first | 160 | 15 | 0.094 | 108 | 62 | 5 |
| 0.25 | no_context | position_updated_first | 29 | 2 | 0.069 | 18 | 12 | 1 |
| 0.25 | no_context | updated_entity_in_source | 119 | 8 | 0.067 | 79 | 47 | 1 |
| 0.25 | no_context | updated_entity_not_in_source | 81 | 9 | 0.111 | 52 | 32 | 6 |
| 0.25 | no_context | weak_or_location_separable | 86 | 10 | 0.116 | 55 | 35 | 6 |
| 0.5 | full_context | all_pairs | 200 | 46 | 0.230 | 88 | 158 | 0 |
| 0.5 | full_context | genuine_both_source_low_answer_overlap | 114 | 22 | 0.193 | 50 | 86 | 0 |
| 0.5 | full_context | position_unchanged_first | 160 | 40 | 0.250 | 75 | 125 | 0 |
| 0.5 | full_context | position_updated_first | 29 | 5 | 0.172 | 11 | 23 | 0 |
| 0.5 | full_context | updated_entity_in_source | 119 | 23 | 0.193 | 52 | 90 | 0 |
| 0.5 | full_context | updated_entity_not_in_source | 81 | 23 | 0.284 | 36 | 68 | 0 |
| 0.5 | full_context | weak_or_location_separable | 86 | 24 | 0.279 | 38 | 72 | 0 |
| 0.5 | no_context | all_pairs | 200 | 10 | 0.050 | 125 | 78 | 7 |
| 0.5 | no_context | genuine_both_source_low_answer_overlap | 114 | 7 | 0.061 | 70 | 46 | 5 |
| 0.5 | no_context | position_unchanged_first | 160 | 9 | 0.056 | 104 | 61 | 4 |
| 0.5 | no_context | position_updated_first | 29 | 1 | 0.034 | 15 | 12 | 3 |
| 0.5 | no_context | updated_entity_in_source | 119 | 8 | 0.067 | 73 | 49 | 5 |
| 0.5 | no_context | updated_entity_not_in_source | 81 | 2 | 0.025 | 52 | 29 | 2 |
| 0.5 | no_context | weak_or_location_separable | 86 | 3 | 0.035 | 55 | 32 | 2 |
| 0.75 | full_context | all_pairs | 200 | 83 | 0.415 | 104 | 179 | 0 |
| 0.75 | full_context | genuine_both_source_low_answer_overlap | 114 | 45 | 0.395 | 57 | 102 | 0 |
| 0.75 | full_context | position_unchanged_first | 160 | 72 | 0.450 | 92 | 140 | 0 |
| 0.75 | full_context | position_updated_first | 29 | 9 | 0.310 | 10 | 28 | 0 |
| 0.75 | full_context | updated_entity_in_source | 119 | 46 | 0.387 | 59 | 106 | 0 |
| 0.75 | full_context | updated_entity_not_in_source | 81 | 37 | 0.457 | 45 | 73 | 0 |
| 0.75 | full_context | weak_or_location_separable | 86 | 38 | 0.442 | 47 | 77 | 0 |
| 0.75 | no_context | all_pairs | 200 | 13 | 0.065 | 115 | 86 | 12 |
| 0.75 | no_context | genuine_both_source_low_answer_overlap | 114 | 7 | 0.061 | 65 | 49 | 7 |
| 0.75 | no_context | position_unchanged_first | 160 | 12 | 0.075 | 95 | 68 | 9 |
| 0.75 | no_context | position_updated_first | 29 | 1 | 0.034 | 15 | 13 | 2 |
| 0.75 | no_context | updated_entity_in_source | 119 | 8 | 0.067 | 68 | 52 | 7 |
| 0.75 | no_context | updated_entity_not_in_source | 81 | 5 | 0.062 | 47 | 34 | 5 |
| 0.75 | no_context | weak_or_location_separable | 86 | 6 | 0.070 | 50 | 37 | 5 |
| 1 | full_context | all_pairs | 200 | 92 | 0.460 | 106 | 184 | 2 |
| 1 | full_context | genuine_both_source_low_answer_overlap | 114 | 50 | 0.439 | 59 | 103 | 2 |
| 1 | full_context | position_unchanged_first | 160 | 77 | 0.481 | 88 | 147 | 2 |
| 1 | full_context | position_updated_first | 29 | 11 | 0.379 | 14 | 26 | 0 |
| 1 | full_context | updated_entity_in_source | 119 | 51 | 0.429 | 61 | 107 | 2 |
| 1 | full_context | updated_entity_not_in_source | 81 | 41 | 0.506 | 45 | 77 | 0 |
| 1 | full_context | weak_or_location_separable | 86 | 42 | 0.488 | 47 | 81 | 0 |
| 1 | no_context | all_pairs | 200 | 21 | 0.105 | 118 | 94 | 9 |
| 1 | no_context | genuine_both_source_low_answer_overlap | 114 | 12 | 0.105 | 69 | 54 | 3 |
| 1 | no_context | position_unchanged_first | 160 | 19 | 0.119 | 97 | 75 | 7 |
| 1 | no_context | position_updated_first | 29 | 1 | 0.034 | 15 | 14 | 1 |
| 1 | no_context | updated_entity_in_source | 119 | 13 | 0.109 | 72 | 57 | 3 |
| 1 | no_context | updated_entity_not_in_source | 81 | 8 | 0.099 | 46 | 37 | 6 |
| 1 | no_context | weak_or_location_separable | 86 | 9 | 0.105 | 49 | 40 | 6 |
| 1.25 | full_context | all_pairs | 200 | 84 | 0.420 | 107 | 176 | 1 |
| 1.25 | full_context | genuine_both_source_low_answer_overlap | 114 | 46 | 0.404 | 59 | 101 | 0 |
| 1.25 | full_context | position_unchanged_first | 160 | 71 | 0.444 | 89 | 141 | 1 |
| 1.25 | full_context | position_updated_first | 29 | 10 | 0.345 | 14 | 25 | 0 |
| 1.25 | full_context | updated_entity_in_source | 119 | 48 | 0.403 | 62 | 105 | 0 |
| 1.25 | full_context | updated_entity_not_in_source | 81 | 36 | 0.444 | 45 | 71 | 1 |
| 1.25 | full_context | weak_or_location_separable | 86 | 38 | 0.442 | 48 | 75 | 1 |
| 1.25 | no_context | all_pairs | 200 | 15 | 0.075 | 109 | 92 | 14 |
| 1.25 | no_context | genuine_both_source_low_answer_overlap | 114 | 10 | 0.088 | 62 | 54 | 8 |
| 1.25 | no_context | position_unchanged_first | 160 | 11 | 0.069 | 89 | 70 | 12 |
| 1.25 | no_context | position_updated_first | 29 | 2 | 0.069 | 14 | 16 | 1 |
| 1.25 | no_context | updated_entity_in_source | 119 | 12 | 0.101 | 66 | 57 | 8 |
| 1.25 | no_context | updated_entity_not_in_source | 81 | 3 | 0.037 | 43 | 35 | 6 |
| 1.25 | no_context | weak_or_location_separable | 86 | 5 | 0.058 | 47 | 38 | 6 |

# earlier analysis checkpoint complementarity across existing legal spatial repair route status checkpoints

Existing checkpoints: 20M, 70M, 80M, 100M. Discrete columns use item-level saved predictions; Reading uses saved column scores only.

## Score-table temporal bound
- cheap7_100M: 43.005724
- cheap7_per_column_best_over_20_70_80_100: 43.213992
- delta_best_columns_minus_100M: 0.208268

## Aggregate discrete complementarity
- mean_discrete_best_single_score: 47.140284
- mean_discrete_100M_score: 47.072785
- mean_discrete_oracle_any_checkpoint_score: 61.406304
- mean_discrete_oracle_minus_100M: 14.333519
- mean_discrete_oracle_minus_best_single: 14.266020
- mean_discrete_majority_string_vote_score: 47.056668
- mean_discrete_majority_correct_state_upper_bound: 46.974566
- cheap7_discrete_oracle_plus_best_reading: 53.882547
- cheap7_discrete_oracle_plus_100M_reading: 53.796571
- cheap7_majority_state_ub_plus_best_reading: 41.512485
- cheap7_majority_string_vote_plus_best_reading: 41.582858
- cheap7_per_column_best_score_table: 43.213992
- cheap7_100M_score_table: 43.005724
- needed_cheap7_if_superglue_aoa_flat: 43.702914
- delta_cheap7_table_best_minus_needed: -0.488922
- delta_majority_string_vote_plus_best_reading_minus_100M: -1.422866
- delta_majority_state_ub_plus_best_reading_minus_100M: -1.493239

## Per-column item complementarity
| column | n | 20M | 70M | 80M | 100M | best | oracle any | oracle-best | string vote | state-vote upper | lost80->100 | gained80->100 | never |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| BLiMP | 59875 | 59.556 | 65.248 | 65.941 | 65.717 | 80 (65.941) | 77.134 | 11.193 | 65.769 | 65.769 | 2.519 | 2.295 | 22.866 |
| Supplement | 5218 | 70.161 | 74.952 | 75.949 | 77.099 | 100 (77.099) | 85.243 | 8.145 | 76.926 | 76.926 | 1.035 | 2.185 | 14.757 |
| EWoK | 7618 | 50.197 | 50.840 | 50.919 | 50.748 | 80 (50.919) | 74.206 | 23.287 | 50.788 | 50.788 | 4.056 | 3.886 | 25.794 |
| Entity | 9483 | 0.264 | 0.306 | 0.348 | 0.337 | 80 (0.348) | 0.538 | 0.190 | 0.337 | 0.337 | 0.021 | 0.011 | 99.462 |
| COMPS | 91028 | 50.613 | 52.253 | 52.553 | 52.575 | 100 (52.575) | 80.085 | 27.510 | 52.560 | 52.560 | 4.780 | 4.802 | 19.915 |
| GlobalPIQA | 203 | 33.990 | 35.468 | 35.468 | 35.961 | 100 (35.961) | 51.232 | 15.271 | 35.961 | 35.468 | 1.970 | 2.463 | 48.768 |

## Interpretation
- Per-column checkpoint selection is small: the cheap7 score-table best over 20M/70M/80M/100M is only the listed delta above 100M.
- Item-level oracle is intentionally impossible for a submitted model; it measures whether errors are fixed or different. Vote rows measure a simple non-oracle combination of existing predictions.
- If vote or score-table recombination is not close to the needed cheap7, existing fixed checkpoints do not justify attention on checkpoint soups before a new learning signal.

# earlier analysis repaired existing-artifact measurements

CPU-only measurements from already-produced legal spatial repair route status artifacts. No model training or new official evaluation was run.

## Repaired temporal complementarity
- all_discrete_columns_validated_against_reports: True
- cheap7_100M_from_records: 43.00571428571429
- cheap7_per_column_best_from_records: 43.214999999999996
- delta_table_best_minus_100M: 0.20928571428570564
- needed_cheap7_if_superglue_aoa_flat: 43.70291394373706
- delta_table_best_minus_needed: -0.48791394373706254
- cheap7_majority_string_vote_plus_best_reading: 43.04640335392727
- delta_majority_string_vote_plus_best_reading_minus_100M_table: 0.04068906821297702
- cheap7_majority_state_upper_plus_best_reading: 42.88111585387425
- delta_majority_state_upper_plus_best_reading_minus_100M_table: -0.12459843184004171
- cheap7_oracle_any_checkpoint_plus_best_reading: 58.5217210958582

| column | valid | n | 100M | best single | oracle any | majority string | state-vote upper |
|---|---:|---:|---:|---:|---:|---:|---:|
| BLiMP | True | 59875 | 65.8707 | 66.1114 | 77.2464 | 65.9255 | 65.9255 |
| Supplement | True | 5218 | 61.1657 | 61.1657 | 71.5329 | 60.7611 | 60.7611 |
| EWoK | True | 7618 | 50.3932 | 51.0071 | 76.1100 | 50.7242 | 50.7242 |
| Entity | True | 6780 | 27.4008 | 27.4008 | 43.0283 | 27.1618 | 26.4902 |
| COMPS | True | 91028 | 52.0083 | 52.0083 | 81.6207 | 51.9638 | 51.9638 |
| GlobalPIQA | True | 203 | 36.0631 | 36.0631 | 51.3738 | 36.0485 | 35.5631 |

## Context length and relative-position reach
- model position config: {'max_position_embeddings': 512, 'max_relative_positions': 256, 'position_buckets': 256, 'relative_attention': True}
- overall: {'any_input_over_max_position_embeddings': False, 'total_candidates_over_max_position_embeddings': 0, 'total_candidates_over_max_relative_positions': 31, 'max_input_len_seen': 282, 'max_relative_distance_seen': 280}
- BLiMP: max input 34, max relative distance 32, over relative 0 / 119750, over absolute 0
- Supplement: max input 36, max relative distance 34, over relative 0 / 10436, over absolute 0
- EWoK: max input 43, max relative distance 41, over relative 0 / 15236, over absolute 0
- Entity: max input 267, max relative distance 265, over relative 30 / 33900, over absolute 0
- COMPS: max input 36, max relative distance 34, over relative 0 / 182056, over absolute 0
- GlobalPIQA_parallel: max input 63, max relative distance 61, over relative 0 / 412, over absolute 0
- GlobalPIQA_nonparallel: max input 206, max relative distance 204, over relative 0 / 200, over absolute 0
- Reading: max input 22, max relative distance 18, over relative 0 / 3569, over absolute 0
- AoA: max input 282, max relative distance 280, over relative 1 / 8005, over absolute 0

## AoA raw correlation
- n_valid_words_for_correlation: 236
- raw_pearson_r_before_p_threshold: -0.06467519850165698
- raw_p_value: 0.3225065702464489
- official_leaderboard_score_after_p_threshold: 0.0
- words_with_surprisal: 485

## SuperGLUE artifact inventory
- checkpoints_present: ['chck_100M']
- non_100M_prediction_files: 0

## Interpretation
- The repaired parser now validates all discrete zero-shot columns against their saved official reports, so item-level temporal measurements are usable.
- Per-column score selection is still far below the cheap7 level needed if SuperGLUE and AoA do not move. Simple prediction voting is worse than the 100M endpoint; only an impossible per-item oracle is large.
- Evaluation inputs fit within the absolute 512-position model limit. Relative-distance saturation exists mainly in COMPS and Entity tails, but not as a hidden absolute-length failure; the largest route-level gap is therefore not explained by missing evaluation reach.
- The AoA raw correlation is not a large latent score source for the existing endpoint, and no non-100M SuperGLUE artifacts exist for this trajectory.

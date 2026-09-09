# earlier analysis EWoK four-way context-target readout

This is a mechanistic measurement over existing checkpoints, not a pretraining target. Official EWoK uses only C1T1 vs C2T1; this readout additionally scores C1T2 and C2T2 and reports interaction `(s11+s22)-(s12+s21)`.

Rows: 22 from `experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/ewok_filtered`; rows/domain=2; seed=80080.

## Model table
- a02_oldtok_reinvest_100M: acc_from_scores=45.455, mean_official_margin=0.1780, mean_interaction=-1.5877, interaction_positive_rate=36.36, context_insensitive_frac=0.227, broad={'Overall_non_submittable': 42.0331347900748, 'EWoK': 53.54}
- a02_legal16_reinvest_100M: acc_from_scores=36.364, mean_official_margin=-0.8389, mean_interaction=-1.6739, interaction_positive_rate=40.91, context_insensitive_frac=0.364, broad={'Overall': 41.257770896404615, 'EWoK': 50.39}

## Validation against saved official predictions
- a02_legal16_reinvest_100M: available=True, match_rate=0.9545454545454546, matches=21/22
  first mismatch: {'uid': 'physical-dynamics_51', 'domain': 'physical-dynamics', 'index': 51, 'computed_pred': 'The speck on the balloon is rotating. The balloon is sliding.', 'saved_pred': 'The speck on the balloon is not rotating. The balloon is sliding.', 's11': -24.4483642578125, 's21': -24.4222412109375, 'official_margin': -0.026123046875}
- a02_oldtok_reinvest_100M: available=True, match_rate=1.0, matches=22/22

## Correlations across scored models
- pearson_mean_interaction_vs_sampled_official_accuracy: 1.0000000000000002
- spearman_mean_interaction_vs_sampled_official_accuracy: 1.0
- pearson_mean_interaction_vs_broad_reference_scalar: 1.0
- spearman_mean_interaction_vs_broad_reference_scalar: 1.0
- note: Small-N descriptive correlations only; used to decide whether the interaction is worth testing on legal corpus-derived contrasts.

CSV records: `experiments/archive/frontier_consolidation/data/ewok_fourway_margin_readout/pilot_2rows_validation/ewok_fourway_records.csv`
Full JSON: `experiments/archive/frontier_consolidation/data/ewok_fourway_margin_readout/pilot_2rows_validation/ewok_fourway_readout_summary.json`

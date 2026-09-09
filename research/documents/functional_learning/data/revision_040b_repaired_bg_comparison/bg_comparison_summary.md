# Step040b matched background comparison on repaired relation-first rows

Both arms use the same repaired rows, trusted coherent86 loader, private-adapter-only optimizer, explicit final-span scoring, and identical background corruption stream. The only arm difference is whether corrupted background positions contribute MLM loss.

## corrupted_answer_only

- Train final: {'n_pairs': 90, 'n_four_condition_success': 14, 'n_query_orientation_success': 67, 'n_query_orientations': 180, 'mean_U': 1.2531669992978116, 'mean_R': 0.6115778384778511, 'mean_beta_pair_average': 0.9323724188878313, 'mean_abs_alpha_pair_average': 0.7458014080960398, 'mean_min_four_signed_margin': -0.4832554359006678}
- Held final: {'n_pairs': 30, 'n_four_condition_success': 2, 'n_query_orientation_success': 16, 'n_query_orientations': 60, 'mean_U': 1.5914066003685325, 'mean_R': -0.18683762537671203, 'mean_beta_pair_average': 0.7022844874959103, 'mean_abs_alpha_pair_average': 1.2470218270482212, 'mean_min_four_signed_margin': -1.5708055486795234}
- Cumulative positions: answer=45120, bg_corrupted=305495, answer/bg overlap=0

## corrupted_answer_plus_bg

- Train final: {'n_pairs': 90, 'n_four_condition_success': 15, 'n_query_orientation_success': 66, 'n_query_orientations': 180, 'mean_U': 0.943050532300991, 'mean_R': 0.4222247012035654, 'mean_beta_pair_average': 0.6826376167522782, 'mean_abs_alpha_pair_average': 0.6973044853869079, 'mean_min_four_signed_margin': -0.5104788565921977}
- Held final: {'n_pairs': 30, 'n_four_condition_success': 2, 'n_query_orientation_success': 15, 'n_query_orientations': 60, 'mean_U': 1.5017102722615365, 'mean_R': -0.42162804830825834, 'mean_beta_pair_average': 0.5400411119766392, 'mean_abs_alpha_pair_average': 1.4374299794987502, 'mean_min_four_signed_margin': -1.9326562861649372}
- Cumulative positions: answer=45120, bg_corrupted=305495, answer/bg overlap=0

## Files

- summary JSON: `experiments/archive/functional_learning/data/revision_040b_repaired_bg_comparison/bg_comparison_summary.json`

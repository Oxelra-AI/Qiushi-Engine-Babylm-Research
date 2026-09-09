# Step040b matched background comparison on repaired relation-first rows

Both arms use the same repaired rows, trusted coherent86 loader, private-adapter-only optimizer, explicit final-span scoring, and identical background corruption stream. The only arm difference is whether corrupted background positions contribute MLM loss.

## corrupted_answer_only

- Train final: {'n_pairs': 90, 'n_four_condition_success': 68, 'n_query_orientation_success': 155, 'n_query_orientations': 180, 'mean_U': 4.10323338778737, 'mean_R': 5.079679188983659, 'mean_beta_pair_average': 4.591456288385515, 'mean_abs_alpha_pair_average': 1.5328000876856525, 'mean_min_four_signed_margin': 1.0949767904867056}
- Held final: {'n_pairs': 30, 'n_four_condition_success': 19, 'n_query_orientation_success': 44, 'n_query_orientations': 60, 'mean_U': 4.277794808568822, 'mean_R': 3.2919063663671517, 'mean_beta_pair_average': 3.7848505874679863, 'mean_abs_alpha_pair_average': 1.7041347973178036, 'mean_min_four_signed_margin': 0.4926985641907797}
- Cumulative positions: answer=45120, bg_corrupted=0, answer/bg overlap=0

## corrupted_answer_plus_bg

- Train final: {'n_pairs': 90, 'n_four_condition_success': 68, 'n_query_orientation_success': 155, 'n_query_orientations': 180, 'mean_U': 4.10323338778737, 'mean_R': 5.079679188983659, 'mean_beta_pair_average': 4.591456288385515, 'mean_abs_alpha_pair_average': 1.5328000876856525, 'mean_min_four_signed_margin': 1.0949767904867056}
- Held final: {'n_pairs': 30, 'n_four_condition_success': 19, 'n_query_orientation_success': 44, 'n_query_orientations': 60, 'mean_U': 4.277794808568822, 'mean_R': 3.2919063663671517, 'mean_beta_pair_average': 3.7848505874679863, 'mean_abs_alpha_pair_average': 1.7041347973178036, 'mean_min_four_signed_margin': 0.4926985641907797}
- Cumulative positions: answer=45120, bg_corrupted=0, answer/bg overlap=0

## Files

- summary JSON: `experiments/archive/functional_learning/data/revision_040d_clean_same_codepath/bg_comparison_summary.json`

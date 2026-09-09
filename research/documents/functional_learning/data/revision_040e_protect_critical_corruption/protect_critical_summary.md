# Step040e protected-critical corruption

15% background corruption was applied as in Step040b, but word groups containing entity A/B, source values A/B, or the shared replacement occurrence were excluded from corruption. Only final answer spans were supervised.

- Final train: {'n_pairs': 90, 'n_four_condition_success': 58, 'n_query_orientation_success': 139, 'n_query_orientations': 180, 'mean_U': 2.689818392662413, 'mean_R': 4.1016536868559585, 'mean_beta_pair_average': 3.395736039759186, 'mean_abs_alpha_pair_average': 1.1807255510985637, 'mean_min_four_signed_margin': 0.6169213642693486}
- Final held: {'n_pairs': 30, 'n_four_condition_success': 11, 'n_query_orientation_success': 34, 'n_query_orientations': 60, 'mean_U': 2.7459783145726475, 'mean_R': 1.8176664060632983, 'mean_beta_pair_average': 2.2818223603179733, 'mean_abs_alpha_pair_average': 1.528950608782082, 'mean_min_four_signed_margin': -0.7093878105189652}
- Cumulative stats: {'bg_corrupted_positions': 240711, 'bg_masked_positions': 192453, 'answer_label_positions': 45120, 'protected_positions_would_have_been_selected': 75125, 'answer_bg_overlap_positions': 0}
- Protected positions per example: {'n_examples': 360, 'mean_protected_positions': 17.58888888888889, 'protected_position_quantiles': {'min': 9, 'p10': 12.0, 'median': 18.0, 'p90': 22.100000000000023, 'max': 35}}

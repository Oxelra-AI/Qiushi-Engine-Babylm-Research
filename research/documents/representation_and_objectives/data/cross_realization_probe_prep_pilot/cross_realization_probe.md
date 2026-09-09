# cross realization probe result cross-realization probe

JSON: `experiments/archive/representation_and_objectives/data/cross_realization_probe_prep_pilot/cross_realization_probe.json`

## Preparation
- prepared_events: 97
- n_examples: 291
- by_eval_set: {'doc_disjoint_all_accepted': 21, 'doc_disjoint_quality': 25, 'source_disjoint_quality': 26, 'train_fixed_probe': 25}
- match: {'fraction_piece_match': 1.0, 'fraction_capitalized_match': 1.0, 'fraction_number_match': 1.0, 'fraction_rel_event_match': 0.958763, 'fraction_hyphenated_match': 0.948454, 'counterfactual_contains_target_fraction': 0.0, 'abs_rel_pos_mean': 0.095336, 'rewrite_word_len_delta_mean': -0.587629, 'rewrite_bpe_len_delta_mean': -1.690722, 'compression_ratio_delta_mean': 0.004732, 'source_side_overlap_delta_mean': 0.020531, 'candidate_count_median': 14.0}

## Arm main metrics (all events)
### full_compact_100M
- rewrite_advantage: n=97, mean=3.257399, boot[2.689698,4.04354], frac_gt0=1.0
- agreement_advantage: n=97, mean=0.473833, boot[0.442588,0.508521], frac_gt0=1.0
- rewrite_loss: n=97, mean=6.379612, boot[6.026927,6.860258], frac_gt0=1.0
- counterfactual_loss: n=97, mean=9.637011, boot[9.143013,10.034559], frac_gt0=1.0
- rewrite_minus_source_loss: n=97, mean=0.025022, boot[-0.351559,0.38417], frac_gt0=0.54

## Key between-arm contrasts (all events)

# cross realization probe result cross-realization probe

JSON: `experiments/archive/representation_and_objectives/data/cross_realization_probe/cross_realization_probe.json`

## Preparation
- prepared_events: 9109
- n_examples: 27327
- by_eval_set: {'doc_disjoint_all_accepted': 2290, 'doc_disjoint_quality': 721, 'source_disjoint_quality': 3097, 'train_fixed_probe': 3001}
- match: {'fraction_piece_match': 1.0, 'fraction_capitalized_match': 1.0, 'fraction_number_match': 1.0, 'fraction_rel_event_match': 0.996267, 'fraction_hyphenated_match': 0.982764, 'counterfactual_contains_target_fraction': 0.0, 'abs_rel_pos_mean': 0.014073, 'rewrite_word_len_delta_mean': -0.11044, 'rewrite_bpe_len_delta_mean': -0.349435, 'compression_ratio_delta_mean': -0.002818, 'source_side_overlap_delta_mean': 0.005957, 'candidate_count_median': 1739.0}

## Arm main metrics (all events)
### full_compact_100M
- rewrite_advantage: n=9109, mean=3.19668, boot[3.102372,3.288195], frac_gt0=1.0
- agreement_advantage: n=9109, mean=0.442726, boot[0.437476,0.448419], frac_gt0=1.0
- rewrite_loss: n=9109, mean=6.748527, boot[6.676205,6.827793], frac_gt0=1.0
- counterfactual_loss: n=9109, mean=9.945207, boot[9.867354,10.015957], frac_gt0=1.0
- rewrite_minus_source_loss: n=9109, mean=0.202073, boot[0.161897,0.241082], frac_gt0=1.0
### drop_abs_100M
- rewrite_advantage: n=9109, mean=3.135289, boot[3.039011,3.225532], frac_gt0=1.0
- agreement_advantage: n=9109, mean=0.440148, boot[0.434274,0.445754], frac_gt0=1.0
- rewrite_loss: n=9109, mean=6.943179, boot[6.865033,7.015802], frac_gt0=1.0
- counterfactual_loss: n=9109, mean=10.078468, boot[10.007489,10.153143], frac_gt0=1.0
- rewrite_minus_source_loss: n=9109, mean=0.254897, boot[0.212306,0.295483], frac_gt0=1.0
### drop_copied_word_100M
- rewrite_advantage: n=9109, mean=3.127765, boot[3.038183,3.214672], frac_gt0=1.0
- agreement_advantage: n=9109, mean=0.441706, boot[0.435917,0.447058], frac_gt0=1.0
- rewrite_loss: n=9109, mean=6.834247, boot[6.760211,6.906596], frac_gt0=1.0
- counterfactual_loss: n=9109, mean=9.962012, boot[9.889735,10.03595], frac_gt0=1.0
- rewrite_minus_source_loss: n=9109, mean=0.277098, boot[0.234519,0.319], frac_gt0=1.0
### compact_repeat_100M
- rewrite_advantage: n=9109, mean=3.065422, boot[2.975147,3.152784], frac_gt0=1.0
- agreement_advantage: n=9109, mean=0.42611, boot[0.420318,0.431989], frac_gt0=1.0
- rewrite_loss: n=9109, mean=6.754182, boot[6.680505,6.82515], frac_gt0=1.0
- counterfactual_loss: n=9109, mean=9.819604, boot[9.742518,9.898082], frac_gt0=1.0
- rewrite_minus_source_loss: n=9109, mean=0.296499, boot[0.258489,0.333241], frac_gt0=1.0
### adjbreak_100M
- rewrite_advantage: n=9109, mean=3.295155, boot[3.199665,3.386299], frac_gt0=1.0
- agreement_advantage: n=9109, mean=0.437758, boot[0.432161,0.443453], frac_gt0=1.0
- rewrite_loss: n=9109, mean=6.290713, boot[6.216189,6.363948], frac_gt0=1.0
- counterfactual_loss: n=9109, mean=9.585868, boot[9.498473,9.666287], frac_gt0=1.0
- rewrite_minus_source_loss: n=9109, mean=0.16024, boot[0.125217,0.198], frac_gt0=1.0

## Key between-arm contrasts (all events)
### full_minus_drop_abs
- rewrite_advantage: n=9109, mean=0.061391, boot[0.023771,0.103802], frac_gt0=1.0
- agreement_advantage: n=9109, mean=0.002578, boot[0.000893,0.004333], frac_gt0=0.997
- rewrite_loss: n=9109, mean=-0.194652, boot[-0.224637,-0.164394], frac_gt0=0.0
### full_minus_drop_copied_word
- rewrite_advantage: n=9109, mean=0.068915, boot[0.031621,0.104779], frac_gt0=1.0
- agreement_advantage: n=9109, mean=0.001021, boot[-0.000597,0.002626], frac_gt0=0.887
- rewrite_loss: n=9109, mean=-0.08572, boot[-0.112102,-0.058902], frac_gt0=0.0
### drop_abs_minus_drop_copied_word
- rewrite_advantage: n=9109, mean=0.007524, boot[-0.032879,0.048793], frac_gt0=0.657
- agreement_advantage: n=9109, mean=-0.001558, boot[-0.003476,0.000214], frac_gt0=0.045
- rewrite_loss: n=9109, mean=0.108932, boot[0.077149,0.139053], frac_gt0=1.0
### drop_abs_minus_compact_repeat
- rewrite_advantage: n=9109, mean=0.069867, boot[0.023318,0.114107], frac_gt0=0.999
- agreement_advantage: n=9109, mean=0.014038, boot[0.011838,0.016128], frac_gt0=1.0
- rewrite_loss: n=9109, mean=0.188997, boot[0.157563,0.222259], frac_gt0=1.0
### adjbreak_minus_compact_repeat
- rewrite_advantage: n=9109, mean=0.229733, boot[0.185337,0.272585], frac_gt0=1.0
- agreement_advantage: n=9109, mean=0.011648, boot[0.009944,0.013392], frac_gt0=1.0
- rewrite_loss: n=9109, mean=-0.463469, boot[-0.493634,-0.432666], frac_gt0=0.0
### drop_abs_minus_adjbreak
- rewrite_advantage: n=9109, mean=-0.159866, boot[-0.208126,-0.116271], frac_gt0=0.0
- agreement_advantage: n=9109, mean=0.00239, boot[0.000446,0.004153], frac_gt0=0.993
- rewrite_loss: n=9109, mean=0.652466, boot[0.616452,0.689749], frac_gt0=1.0

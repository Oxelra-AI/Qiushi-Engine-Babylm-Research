# earlier analysis cross-realization saved-checkpoint probe

Input events: 16
Arms: full_compact_100M, drop_abs_100M, drop_copied_word_100M, repeat_100M, adjbreak_100M

## All-event summaries
- full_compact_100M: compact_adv=-0.049349, source_adv=-0.015911, compact_minus_source_loss=0.033438, rep_A=0.339220, n=16
- drop_abs_100M: compact_adv=0.607022, source_adv=0.445887, compact_minus_source_loss=-0.161135, rep_A=0.323074, n=16
- drop_copied_word_100M: compact_adv=0.930055, source_adv=0.331352, compact_minus_source_loss=-0.598702, rep_A=0.316301, n=16
- repeat_100M: compact_adv=0.674048, source_adv=0.313670, compact_minus_source_loss=-0.360378, rep_A=0.323790, n=16
- adjbreak_100M: compact_adv=0.446264, source_adv=0.426999, compact_minus_source_loss=-0.019266, rep_A=0.278196, n=16

## Arm contrasts (all events)
- full_minus_drop_abs: compact_context_advantage_nats=-0.656371, source_context_advantage_nats=-0.461798, compact_minus_source_loss_nats=0.194573, rep_source_compact_advantage=0.016146, rep_cos_source_compact=0.031208, rep_cos_source_counterfactual=0.015062
- full_minus_drop_copied_word: compact_context_advantage_nats=-0.979403, source_context_advantage_nats=-0.347263, compact_minus_source_loss_nats=0.632140, rep_source_compact_advantage=0.022919, rep_cos_source_compact=0.023035, rep_cos_source_counterfactual=0.000116
- drop_abs_minus_drop_copied_word: compact_context_advantage_nats=-0.323033, source_context_advantage_nats=0.114535, compact_minus_source_loss_nats=0.437567, rep_source_compact_advantage=0.006773, rep_cos_source_compact=-0.008174, rep_cos_source_counterfactual=-0.014947
- drop_abs_minus_repeat: compact_context_advantage_nats=-0.067026, source_context_advantage_nats=0.132216, compact_minus_source_loss_nats=0.199243, rep_source_compact_advantage=-0.000716, rep_cos_source_compact=-0.029395, rep_cos_source_counterfactual=-0.028679
- full_minus_repeat: compact_context_advantage_nats=-0.723397, source_context_advantage_nats=-0.329582, compact_minus_source_loss_nats=0.393815, rep_source_compact_advantage=0.015430, rep_cos_source_compact=0.001813, rep_cos_source_counterfactual=-0.013617
- adjbreak_minus_repeat: compact_context_advantage_nats=-0.227784, source_context_advantage_nats=0.113328, compact_minus_source_loss_nats=0.341112, rep_source_compact_advantage=-0.045594, rep_cos_source_compact=-0.008791, rep_cos_source_counterfactual=0.036804
- full_minus_adjbreak: compact_context_advantage_nats=-0.495613, source_context_advantage_nats=-0.442910, compact_minus_source_loss_nats=0.052703, rep_source_compact_advantage=0.061024, rep_cos_source_compact=0.010603, rep_cos_source_counterfactual=-0.050421

JSON: `experiments/archive/frontier_consolidation/data/cross_realization_probe_smoke/cross_realization_probe_integrated.json`

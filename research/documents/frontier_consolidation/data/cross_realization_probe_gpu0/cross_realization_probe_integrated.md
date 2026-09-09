# earlier analysis cross-realization saved-checkpoint probe

Input events: 4500
Arms: full_compact_100M, drop_abs_100M, drop_copied_word_100M

## All-event summaries
- full_compact_100M: compact_adv=0.189814, source_adv=0.163787, compact_minus_source_loss=-0.026027, rep_A=0.273255, n=4500
- drop_abs_100M: compact_adv=0.177078, source_adv=0.154530, compact_minus_source_loss=-0.022548, rep_A=0.275394, n=4500
- drop_copied_word_100M: compact_adv=0.181702, source_adv=0.170591, compact_minus_source_loss=-0.011112, rep_A=0.270169, n=4500

## Arm contrasts (all events)
- full_minus_drop_abs: compact_context_advantage_nats=0.012736, source_context_advantage_nats=0.009256, compact_minus_source_loss_nats=-0.003479, rep_source_compact_advantage=-0.002139, rep_cos_source_compact=0.010092, rep_cos_source_counterfactual=0.012231
- full_minus_drop_copied_word: compact_context_advantage_nats=0.008111, source_context_advantage_nats=-0.006804, compact_minus_source_loss_nats=-0.014915, rep_source_compact_advantage=0.003087, rep_cos_source_compact=-0.000297, rep_cos_source_counterfactual=-0.003383
- drop_abs_minus_drop_copied_word: compact_context_advantage_nats=-0.004624, source_context_advantage_nats=-0.016060, compact_minus_source_loss_nats=-0.011436, rep_source_compact_advantage=0.005225, rep_cos_source_compact=-0.010389, rep_cos_source_counterfactual=-0.015614

JSON: `experiments/archive/frontier_consolidation/data/cross_realization_probe_gpu0/cross_realization_probe_integrated.json`

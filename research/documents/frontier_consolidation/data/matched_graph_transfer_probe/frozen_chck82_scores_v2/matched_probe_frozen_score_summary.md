# matched graph transfer probe result frozen chck82 matched graph-transfer scores

Forward-only measurement on lexically matched generated probes. No training, upload, or leaderboard submission.
Input accepted examples: 10; complete paired items: 10; contexts scored: 40
Checkpoint: `experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_82M`; device `cuda`; elapsed score sec `0.5`

## Context-level target-vs-distractor margins
- query_only: n=10, target-preferred=0.9, mean delta=3.378088617324829, mean target NLL=5.161269903182983
- neutral: n=10, target-preferred=1.0, mean delta=1.9693363189697266, mean target NLL=6.409971094131469
- structured: n=10, target-preferred=0.8, mean delta=2.780471217632294, mean target NLL=4.829344838857651
- reversed: n=10, target-preferred=0.6, mean delta=1.290309238433838, mean target NLL=4.668313431739807

## Paired effects
- lift_delta_structured_minus_neutral: mean 0.8111348986625672, CI [-0.16116175651550294, 1.7343563079833983], P(mean<=0)=0.0502
- sensitivity_delta_structured_minus_reversed: mean 1.4901619791984557, CI [0.7879846215248107, 2.216939103603363], P(mean<=0)=0.0
- target_nll_improvement_neutral_minus_structured: mean 1.580626255273819, CI [0.5667293548583985, 2.6868653297424316], P(mean<=0)=0.0012
- target_nll_improvement_reversed_minus_structured: mean -0.16103140711784364, CI [-0.7517595052719116, 0.3382463216781616], P(mean<=0)=0.7028

## Family means
- entity_state_update: n=3, lift=-0.4876284996668498, sensitivity=0.9503438075383505, structured_pref_frac=0.3333333333333333
- event_temporal_causal: n=3, lift=2.0995141665140786, sensitivity=2.724886178970337, structured_pref_frac=1.0
- polarity_contrast_event: n=3, lift=0.9477783838907877, sensitivity=1.2238553365071614, structured_pref_frac=1.0
- social_belief_report: n=1, lift=0.4323568344116211, sensitivity=0.20436382293701172, structured_pref_frac=1.0

CSV: `experiments/archive/frontier_consolidation/data/matched_graph_transfer_probe/frozen_chck82_scores_v2/matched_probe_frozen_scores.csv`
Paired effects: `experiments/archive/frontier_consolidation/data/matched_graph_transfer_probe/frozen_chck82_scores_v2/matched_probe_paired_effects.csv`
JSON: `experiments/archive/frontier_consolidation/data/matched_graph_transfer_probe/frozen_chck82_scores_v2/matched_probe_frozen_score_summary.json`

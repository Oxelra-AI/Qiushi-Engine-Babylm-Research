# matched graph transfer probe result frozen chck82 matched graph-transfer scores

Forward-only measurement on lexically matched generated probes. No training, upload, or leaderboard submission.
Input accepted examples: 10; complete paired items: 10; contexts scored: 40
Checkpoint: `experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_100M`; device `cuda`; elapsed score sec `0.6`

## Context-level target-vs-distractor margins
- query_only: n=10, target-preferred=0.9, mean delta=3.4316223859786987, mean target NLL=5.197493386268616
- neutral: n=10, target-preferred=1.0, mean delta=1.655440664291382, mean target NLL=6.747540616989136
- structured: n=10, target-preferred=0.8, mean delta=2.518896555900574, mean target NLL=5.047953164577484
- reversed: n=10, target-preferred=0.6, mean delta=1.0078621864318849, mean target NLL=5.059293353557587

## Paired effects
- lift_delta_structured_minus_neutral: mean 0.8634558916091919, CI [-0.23322159051895142, 1.9359630703926087], P(mean<=0)=0.062
- sensitivity_delta_structured_minus_reversed: mean 1.511034369468689, CI [0.6977419853210449, 2.317938470840454], P(mean<=0)=0.0
- target_nll_improvement_neutral_minus_structured: mean 1.6995874524116517, CI [0.5978669762611389, 2.826200819015503], P(mean<=0)=0.0016
- target_nll_improvement_reversed_minus_structured: mean 0.011340188980102538, CI [-0.5539255619049073, 0.47891740798950194], P(mean<=0)=0.4614

## Family means
- entity_state_update: n=3, lift=-0.523881196975708, sensitivity=0.9966962337493896, structured_pref_frac=0.3333333333333333
- event_temporal_causal: n=3, lift=2.4920921325683594, sensitivity=2.8920114835103354, structured_pref_frac=1.0
- polarity_contrast_event: n=3, lift=0.7733116149902344, sensitivity=1.059750239054362, structured_pref_frac=1.0
- social_belief_report: n=1, lift=0.4099912643432617, sensitivity=0.2649698257446289, structured_pref_frac=1.0

CSV: `experiments/archive/frontier_consolidation/data/matched_graph_transfer_probe/frozen_chck100_scores_v2/matched_probe_frozen_scores.csv`
Paired effects: `experiments/archive/frontier_consolidation/data/matched_graph_transfer_probe/frozen_chck100_scores_v2/matched_probe_paired_effects.csv`
JSON: `experiments/archive/frontier_consolidation/data/matched_graph_transfer_probe/frozen_chck100_scores_v2/matched_probe_frozen_score_summary.json`

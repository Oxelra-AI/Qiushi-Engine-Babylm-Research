# curriculum parity probe — witness-grounded spatial exchange v5

Created: 2026-08-31T06:40:18Z

## Build

Rows seen: 64183; sentences: 668049; regex matches: 58; cases: **6**.

By attested relation: `{'above': 6}`. Evidence patterns: `{'on_top_of_vertical': 2, 'locative_verb_vertical': 4}`.

Generated words if trained once: counterfactual dual-query `312`, full factorial `624`. No generated text was used for training.

Rejects: `{'sentence_filter': 224854, 'token_shape': 2, 'bad_head': 45, 'freq_mismatch': 4, 'same_head': 1}`.

Cases: `experiments/archive/representation_and_objectives/data/spatial_exchange_v5/spatial_exchange_v5_cases.jsonl`; samples: `experiments/archive/representation_and_objectives/data/spatial_exchange_v5/spatial_exchange_v5_samples.jsonl`.

## Frozen checkpoint score

| metric | compact | rowblock | interleaved | rank | range |
|---|---:|---:|---:|---|---:|
| cf_higher_m | 0.4080403645833333 | 0.30517578125 | -0.041097005208333336 | compact > rowblock > interleaved | 0.44913736979166663 |
| cf_lower_m | -0.5814208984375 | -0.3828125 | -0.0693359375 | interleaved > rowblock > compact | 0.5120849609375 |
| cf_dual_m | -0.17338053385416666 | -0.07763671875 | -0.11043294270833333 | rowblock > interleaved > compact | 0.09574381510416666 |
| target_perm_dual_m | -0.07248957951863606 | -0.07024129231770833 | -0.06730175018310547 | interleaved > rowblock > compact | 0.005187829335530594 |
| erased_dual_abs | 4.356119791666667 | 3.3583984375 | 2.8131510416666665 | compact > rowblock > interleaved | 1.5429687500000004 |
| equiv_above_gap_abs | 2.509765625 | 1.033447265625 | 1.341796875 | compact > interleaved > rowblock | 1.476318359375 |
| equiv_below_gap_abs | 3.34521484375 | 0.7779947916666666 | 1.1123860677083333 | compact > interleaved > rowblock | 2.5672200520833335 |

Interpretation: useful relation signal requires unsaturated counterfactual margins, arm separation larger than target-permutation behavior, low relation-erased absolute margins, and preferably rowblock-positive movement on spatial exchange consistent with the previous GlobalPIQA_parallel tradeoff.

Score summary: `experiments/archive/representation_and_objectives/data/spatial_exchange_v5/score_cpu/spatial_exchange_v5_score_summary.json`; rows: `experiments/archive/representation_and_objectives/data/spatial_exchange_v5/score_cpu/spatial_exchange_v5_scores.jsonl`.

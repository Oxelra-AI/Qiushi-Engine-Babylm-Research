# connectivity substrate construction and audit coordinate-connectivity learned probe

Tests whether filler-connected constraint paths enable reusable coordinate induction.

- data_root: `experiments/archive/representation_and_objectives/data/pair_binding_neutral_substrate`
- checkpoint: `experiments/archive/representation_and_objectives/training/runs/qwen_8x480_16k_wwm_to_token_100M_seed43022/hf_model/chck_80M`
- conditions: `['pair_rewired']`
- arms: `['aligned_state_bridge', 'inverted_state_bridge']`
- seeds: `[28700]`
- epochs: 40; batch_size: 16

## Central metrics (means over completed seeds)

| condition | arm | train acc | psc same changed | psc opp changed | psc same pair-both | psc opp pair-both | mixed true stmt |
|---|---|---:|---:|---:|---:|---:|---:|
| pair_rewired | aligned_state_bridge | 1.000 | 0.594 | 0.766 | 0.594 | 0.758 | 0.461 |
| pair_rewired | inverted_state_bridge | 1.000 | 0.273 | 0.742 | 0.273 | 0.719 | 0.801 |

## Files
- all_results: `experiments/archive/representation_and_objectives/data/pair_binding_probe_rewired_seed28700/all_results.json`
- aggregate: `experiments/archive/representation_and_objectives/data/pair_binding_probe_rewired_seed28700/aggregate_summary.json`
- per-row eval: `experiments/archive/representation_and_objectives/data/pair_binding_probe_rewired_seed28700/per_row_eval_predictions.jsonl`
- per-row train: `experiments/archive/representation_and_objectives/data/pair_binding_probe_rewired_seed28700/per_row_train_predictions.jsonl`

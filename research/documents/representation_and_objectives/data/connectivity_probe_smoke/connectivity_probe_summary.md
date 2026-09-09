# connectivity substrate construction and audit coordinate-connectivity learned probe

Tests whether filler-connected constraint paths enable reusable coordinate induction.

- data_root: `experiments/archive/representation_and_objectives/data/coordinate_connectivity_substrate`
- checkpoint: `experiments/archive/representation_and_objectives/training/runs/qwen_8x480_16k_wwm_to_token_100M_seed43022/hf_model/chck_80M`
- conditions: `['connected']`
- arms: `['aligned_state_bridge']`
- seeds: `[28600]`
- epochs: 1; batch_size: 16

## Central metrics (means over completed seeds)

| condition | arm | train acc | psc same changed | psc opp changed | psc same pair-both | psc opp pair-both | mixed true stmt |
|---|---|---:|---:|---:|---:|---:|---:|
| connected | aligned_state_bridge | 0.584 | n/a | 0.594 | n/a | 0.422 | 0.574 |

## Files
- all_results: `experiments/archive/representation_and_objectives/data/connectivity_probe_smoke/all_results.json`
- aggregate: `experiments/archive/representation_and_objectives/data/connectivity_probe_smoke/aggregate_summary.json`
- per-row eval: `experiments/archive/representation_and_objectives/data/connectivity_probe_smoke/per_row_eval_predictions.jsonl`
- per-row train: `experiments/archive/representation_and_objectives/data/connectivity_probe_smoke/per_row_train_predictions.jsonl`

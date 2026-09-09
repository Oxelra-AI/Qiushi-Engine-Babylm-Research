# earlier analysis fast balanced k0/k16 learned probe

This is the repaired binary learned test after the analysis framework for factorial probe static-slot confound.  It uses dynamic padding and a custom DeBERTa classification head, saving per-row logits for exact-choice analysis.

## Run settings

- data_root: `experiments/archive/representation_and_objectives/data/information_budget_substrate`
- checkpoint: `experiments/archive/representation_and_objectives/training/runs/qwen_8x480_16k_wwm_to_token_100M_seed43022/hf_model/chck_80M`
- conditions: `['replace_k16_spread']`
- arms: `['aligned_state_bridge']`
- seeds: `[28499]`
- epochs: 1; batch_size: 16; eval_batch_size: 64
- lr_encoder: 2e-05; lr_head: 0.001; max_len: 196
- errors: 0

## Central metrics (means over completed seeds)

| condition | arm | train acc | psc exact changed same | psc exact changed opposite | psc exact pair-both same | psc exact pair-both opposite | psc true-row same | psc true-row opposite | mixed true statement |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| replace_k16_spread | aligned_state_bridge | 0.557 | 0.422 | 0.625 | 0.352 | 0.438 | 0.531 | 0.617 | 0.941 |

## Interpretation guide

- k0 should display the redundant shortcut pattern: strong opposite-initial but weak same-initial exact changed choice.  This is diagnostic of anti-copy behavior.
- k16 advances the route only if exact changed choice and exact pair-both are strong on both same and opposite initial patterns, including off-diagonal cells in the JSON/per-row output.
- Signed coordinate propagation requires aligned and inverted arms to diverge on mixed held-seen true statements; state-only gains are task-local event/state learning.
- True-row statement scores are retained for comparison with older pseudolikelihood-style summaries but exact choice is the primary behavioral metric.

## Files
- all_results: `experiments/archive/representation_and_objectives/data/fast_balanced_probe_smoke/all_results.json`
- aggregate_summary: `experiments/archive/representation_and_objectives/data/fast_balanced_probe_smoke/aggregate_summary.json`
- per-row eval predictions: `experiments/archive/representation_and_objectives/data/fast_balanced_probe_smoke/per_row_eval_predictions.jsonl`
- per-row train predictions: `experiments/archive/representation_and_objectives/data/fast_balanced_probe_smoke/per_row_train_predictions.jsonl`

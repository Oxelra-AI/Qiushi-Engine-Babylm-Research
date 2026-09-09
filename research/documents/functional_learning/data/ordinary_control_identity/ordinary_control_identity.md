# earlier analysis ordinary-continuation endpoint identity

Created: `2026-09-08T06:18:23Z`

Complete identity: `True`

Ordinary endpoint: `experiments/archive/functional_learning/data/unchanged_focus_weighted_train/inherited_wwm/checkpoints/update_0080`

## Matching to the evaluated exact (M,S) run

{
  "same_tail_and_prefix": true,
  "same_train_seed": true,
  "same_updates": true,
  "same_words_per_update": true,
  "same_schedule_total": true,
  "same_schedule_offset": true,
  "same_warmup": true,
  "same_lr_peak": true,
  "same_mask_prob": true,
  "same_private_scale": true,
  "same_final_schedule_idx": true,
  "same_final_lr": true
}

## Ordinary WWM and optimizer evidence

{
  "log_checks": {
    "n_update_logs": 80,
    "all_updates_present": true,
    "schedule_idx_sequence_101_to_180": true,
    "all_qwen_rows_use_wwm": true,
    "all_focus_targets_zero": true,
    "all_targets_ordinary": true,
    "all_loss_pooled_token_mean": true,
    "total_targets_from_logs": 697102,
    "summary_total_targets": 697102,
    "total_targets_match_summary": true,
    "all_logged_lr_matches_schedule_function": true
  },
  "optimizer": {
    "trainable_tensors": 48,
    "trainable_params": 995584,
    "private_adapter_tensors": 48,
    "private_adapter_params": 995584,
    "private_only_optimizer_recorded": true,
    "private_only_weight_delta_vs_parent": true
  }
}

## Weight movement

{
  "parent_model_sha256": "e14d757ae51b41e33bf0813f841248fecd1eefeb9e040f520c4c6203343b15c8",
  "endpoint_model_sha256": "1e5b3ec6eade0559f4eb3b9bed7553e89e88ddac10e4595a421d6f2a5cd30be3",
  "parent_key_count": 266,
  "endpoint_key_count": 266,
  "missing_in_endpoint": [],
  "extra_in_endpoint": [],
  "changed_tensor_count": 48,
  "changed_private_tensor_count": 48,
  "changed_non_private_tensor_count": 0,
  "unchanged_tensor_count": 218,
  "changed_private_head": [
    "deberta.encoder.layer.0.private_adapter.down.bias",
    "deberta.encoder.layer.0.private_adapter.down.weight",
    "deberta.encoder.layer.0.private_adapter.layer_norm.bias",
    "deberta.encoder.layer.0.private_adapter.layer_norm.weight",
    "deberta.encoder.layer.0.private_adapter.up.bias",
    "deberta.encoder.layer.0.private_adapter.up.weight",
    "deberta.encoder.layer.1.private_adapter.down.bias",
    "deberta.encoder.layer.1.private_adapter.down.weight"
  ],
  "changed_non_private_head": [],
  "max_abs_delta_private": 0.002404354512691498,
  "max_abs_delta_non_private": 0.0,
  "rms_delta_private": 0.0004744391200339429,
  "rms_delta_non_private": 0.0,
  "private_only_changed": true
}

## First macro short execution

{
  "prefix_info_matches_train_config": true,
  "macro_rows": 252,
  "macro_words": 39652,
  "prepared_stats": {
    "rows": 252,
    "words": 39652,
    "ordinary_wwm_rows": 219,
    "qwen_focus_rows": 0,
    "qwen_wwm_rows": 33,
    "ordinary_target_tokens": 8852,
    "focus_target_tokens": 0,
    "zero_label_rows": 0,
    "qwen_rows": 33,
    "compact_modified_rows": 0,
    "topup_rows": 0,
    "focus_selected_groups": 0,
    "focus_candidate_groups": 0,
    "focus_selected_candidate_kind_counts": {},
    "focus_selected_pair_count": 0
  },
  "focus_label_positions_from_examples": 0,
  "ordinary_label_positions_from_examples": 8852,
  "actual_short_execution": "called earlier analysis prepare_macro with objective='inherited_wwm' on the first macro-batch"
}

The existing earlier analysis inherited_wwm endpoint is the ordinary-continuation control for seed62064: standard row-keyed WWM on all rows including Qwen rows, same prefix and learning-rate trajectory as evaluated exact (M,S), and private-adapter-only movement. It still needs complete compatible evaluation before entering the same-coordinate model table.

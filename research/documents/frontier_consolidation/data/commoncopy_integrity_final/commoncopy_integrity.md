# commoncopy and paired world design common-copy no-disentangle integrity

Created UTC: `2026-09-02T14:51:59Z`

all_commoncopy_ready_for_selected_eval: `False`

## commoncopy_compact
- ok_for_selected_eval: `False`
- run_dir: `experiments/archive/frontier_consolidation/training/runs/commoncopy_nodis_compact_deberta100M_seed43022`
- init: `{"after_copy_common_key_count": 140, "after_copy_nonexact_common_count": 0, "initial_logit_mean_abs": 0.0024677126202732325, "mode": "training_build_model_patch", "only_left_count": 32, "source_parameter_count": 34467424, "source_pos_att_type": ["p2c", "c2p"], "status": "COMMON_COPIED_NODIS_INITIALIZATION", "target_parameter_count": 30773344, "target_pos_att_type": []}`
- metrics: `{"actual_training_steps": 2529, "example_jsonl": "experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl", "example_jsonl_label": "common_copy_nodis_compact", "first_checkpoint": "chck_10M", "last_checkpoint": "chck_100M", "loss_first": 9.837443351745605, "loss_last": 4.234443187713623, "parameter_count": 30773344, "saved_checkpoint_count": 10, "tokenizer_label": "compliant16k_reinvest10M", "vocab_size": 16384, "word_exposure": 100000000}`
- problems:
  - metrics batch_size_ok failed; brief={'word_exposure': 100000000, 'actual_training_steps': 2529, 'loss_first': 9.837443351745605, 'loss_last': 4.234443187713623, 'parameter_count': 30773344, 'vocab_size': 16384, 'tokenizer_label': 'compliant16k_reinvest10M', 'saved_checkpoint_count': 10, 'first_checkpoint': 'chck_10M', 'last_checkpoint': 'chck_100M', 'example_jsonl_label': 'common_copy_nodis_compact', 'example_jsonl': 'experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl'}

## commoncopy_repeat
- ok_for_selected_eval: `False`
- run_dir: `experiments/archive/frontier_consolidation/training/runs/commoncopy_nodis_repeat_deberta100M_seed43022`
- init: `{"after_copy_common_key_count": 140, "after_copy_nonexact_common_count": 0, "initial_logit_mean_abs": 0.0024677126202732325, "mode": "training_build_model_patch", "only_left_count": 32, "source_parameter_count": 34467424, "source_pos_att_type": ["p2c", "c2p"], "status": "COMMON_COPIED_NODIS_INITIALIZATION", "target_parameter_count": 30773344, "target_pos_att_type": []}`
- metrics: `{"actual_training_steps": 2529, "example_jsonl": "experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_repeat_compact_reinvest_100M.jsonl", "example_jsonl_label": "common_copy_nodis_repeat", "first_checkpoint": "chck_10M", "last_checkpoint": "chck_100M", "loss_first": 9.839239120483398, "loss_last": 4.421522617340088, "parameter_count": 30773344, "saved_checkpoint_count": 10, "tokenizer_label": "compliant16k_reinvest10M", "vocab_size": 16384, "word_exposure": 100000000}`
- problems:
  - metrics batch_size_ok failed; brief={'word_exposure': 100000000, 'actual_training_steps': 2529, 'loss_first': 9.839239120483398, 'loss_last': 4.421522617340088, 'parameter_count': 30773344, 'vocab_size': 16384, 'tokenizer_label': 'compliant16k_reinvest10M', 'saved_checkpoint_count': 10, 'first_checkpoint': 'chck_10M', 'last_checkpoint': 'chck_100M', 'example_jsonl_label': 'common_copy_nodis_repeat', 'example_jsonl': 'experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_repeat_compact_reinvest_100M.jsonl'}

## Boundary
File-only integrity reader; no training, selected evaluation, SuperGLUE, AoA, upload, or leaderboard submission.

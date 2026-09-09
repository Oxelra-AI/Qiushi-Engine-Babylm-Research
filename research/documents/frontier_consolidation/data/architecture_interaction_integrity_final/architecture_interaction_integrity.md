# architecture_interaction_integrity

Created UTC: `2026-09-02T13:10:11Z`

all_selected_arms_ready_for_selected_eval: `True`
all_four_cells_ready_for_selected_eval: `True`

## full_compact_existing
- ok_for_selected_eval: `True`
- run_dir: `experiments/archive/frontier_consolidation/training/runs/complianttok_reinvest_seed43022_r2`
- variant/data: `full_p2c_c2p_abs` / `compact`
- metrics: `{"actual_training_steps": 2529, "ffn_mult": 4, "first_checkpoint": "chck_1M", "hidden_size": 480, "last_checkpoint": "chck_100M", "loss_first": 9.837543487548828, "loss_last": 2.5525617599487305, "mask_prob_end": 0.15, "mask_prob_start": 0.15, "masking_curriculum": "wwm_fixed", "max_seq_length": 256, "model_family": "DebertaV2ForMaskedLM", "n_head": 8, "n_layer": 8, "parameter_count": 34467424, "saved_checkpoint_count": 100, "seed": 43, "seq_length": 256, "tokenizer_label": "compliant16k_reinvest10M", "vocab_size": 16384, "word_exposure": 100000000}`
- tensor signatures:
  - chck_80M: ok=True, pos_key=16, pos_query=16, abs=True, rel=True
  - chck_100M: ok=True, pos_key=16, pos_query=16, abs=True, rel=True

## full_repeat
- ok_for_selected_eval: `True`
- run_dir: `experiments/archive/frontier_consolidation/training/runs/full_p2c_c2p_abs_repeat_deberta100M_seed43022`
- variant/data: `full_p2c_c2p_abs` / `repeat`
- metrics: `{"actual_training_steps": 2529, "ffn_mult": 4, "first_checkpoint": "chck_10M", "hidden_size": 480, "last_checkpoint": "chck_100M", "loss_first": 9.839323997497559, "loss_last": 2.5566213130950928, "mask_prob_end": 0.15, "mask_prob_start": 0.15, "masking_curriculum": "wwm_fixed", "max_seq_length": 256, "model_family": "DebertaV2ForMaskedLM", "n_head": 8, "n_layer": 8, "parameter_count": 34467424, "saved_checkpoint_count": 10, "seed": 43, "seq_length": 256, "tokenizer_label": "compliant16k_reinvest10M", "vocab_size": 16384, "word_exposure": 100000000}`
- tensor signatures:
  - chck_80M: ok=True, pos_key=16, pos_query=16, abs=True, rel=True
  - chck_100M: ok=True, pos_key=16, pos_query=16, abs=True, rel=True

## nodis_compact
- ok_for_selected_eval: `True`
- run_dir: `experiments/archive/frontier_consolidation/training/runs/no_disentangle_abs_compact_deberta100M_seed43022`
- variant/data: `no_disentangle_abs` / `compact`
- metrics: `{"actual_training_steps": 2529, "ffn_mult": 4, "first_checkpoint": "chck_10M", "hidden_size": 480, "last_checkpoint": "chck_100M", "loss_first": 9.792725563049316, "loss_last": 4.245963096618652, "mask_prob_end": 0.15, "mask_prob_start": 0.15, "masking_curriculum": "wwm_fixed", "max_seq_length": 256, "model_family": "DebertaV2ForMaskedLM", "n_head": 8, "n_layer": 8, "parameter_count": 30773344, "saved_checkpoint_count": 10, "seed": 43, "seq_length": 256, "tokenizer_label": "compliant16k_reinvest10M", "vocab_size": 16384, "word_exposure": 100000000}`
- tensor signatures:
  - chck_80M: ok=True, pos_key=0, pos_query=0, abs=True, rel=True
  - chck_100M: ok=True, pos_key=0, pos_query=0, abs=True, rel=True

## nodis_repeat
- ok_for_selected_eval: `True`
- run_dir: `experiments/archive/frontier_consolidation/training/runs/no_disentangle_abs_repeat_deberta100M_seed43022`
- variant/data: `no_disentangle_abs` / `repeat`
- metrics: `{"actual_training_steps": 2529, "ffn_mult": 4, "first_checkpoint": "chck_10M", "hidden_size": 480, "last_checkpoint": "chck_100M", "loss_first": 9.794188499450684, "loss_last": 4.909754276275635, "mask_prob_end": 0.15, "mask_prob_start": 0.15, "masking_curriculum": "wwm_fixed", "max_seq_length": 256, "model_family": "DebertaV2ForMaskedLM", "n_head": 8, "n_layer": 8, "parameter_count": 30773344, "saved_checkpoint_count": 10, "seed": 43, "seq_length": 256, "tokenizer_label": "compliant16k_reinvest10M", "vocab_size": 16384, "word_exposure": 100000000}`
- tensor signatures:
  - chck_80M: ok=True, pos_key=0, pos_query=0, abs=True, rel=True
  - chck_100M: ok=True, pos_key=0, pos_query=0, abs=True, rel=True


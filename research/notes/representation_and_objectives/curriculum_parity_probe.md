# curriculum parity probe — repaired curriculum fixed-256 parity probe

Created: 2026-08-31T06:35:51Z.

The trusted legal40k baseline records seed=43, extra_init_seed=43022, train_rng_seed=43023. The curriculum experiment design launch did not pass train_rng_seed and therefore used 43 for masking/dropout; it is not attributable to curriculum and was stopped/failed before useful output.

Fixed-256 dataset parity: `{'base_len': 1296, 'curriculum_fixed_len': 1296, 'all_rows_equal': True, 'first_mismatches': [], 'total_steps': 6}`.

First effective batch parity: `{'batch_equal': {'input_ids': True, 'attention_mask': True, 'word_group': True, 'words': True}, 'batch_words': 39370, 'input_ids_hash': '020001a6381b1248', 'word_group_hash': 'da25dd9e3325f226'}`.

First mask parity: `{'device': 'cuda:0', 'train_rng_seed': 43023, 'masked_inputs_equal': True, 'labels_equal': True, 'attention_equal': True, 'n_masked': 8247, 'mask_rate': 0.15281560953916282, 'labels_hash': '75e226f06017b8d0', 'masked_inputs_hash': '5ec0f4b9251d0a4c'}`.

Initialization parity: identical=True, diff=`{'max_abs': 0.0, 'max_key': None, 'n_nonzero_tensors': 0, 'n_tensors': 172}`.

First update parity: `{'loss_base_path': 10.683919814336338, 'loss_curriculum_fixed_path': 10.683919814336338, 'loss_abs_diff': 0.0, 'update_info_base': {'n_pred_total': 8247, 'active_microbatches': 4, 'grad_norm_before_clip': 2.2912180423736572, 'lr_after_step': 0.001, 'labels_hash': '75e226f06017b8d0', 'masked_inputs_hash': '5ec0f4b9251d0a4c'}, 'update_info_curriculum_fixed': {'n_pred_total': 8247, 'active_microbatches': 4, 'grad_norm_before_clip': 2.2912180423736572, 'lr_after_step': 0.001, 'labels_hash': '75e226f06017b8d0', 'masked_inputs_hash': '5ec0f4b9251d0a4c'}, 'post_update_state_diff': {'max_abs': 0.0, 'max_key': None, 'n_nonzero_tensors': 0, 'n_tensors': 172}, 'identical_or_close': True}`.

Short-tail subset result: `{'min_chunk_tokens_1': {'chunks': 4983, 'tail_rows_losing_tokens': 0, 'tail_tokens_lost': 0}, 'min_chunk_tokens_8': {'chunks': 4679, 'tail_rows_losing_tokens': 304, 'tail_tokens_lost': 1058}, 'interpretation': 'Endpoint runs must use min_chunk_tokens=1 so short tails are retained; min8 was only a smoke-test convenience and changes token exposure.'}`. Endpoint curriculum runs must use `--min_chunk_tokens 1 --train_rng_seed 43023`.

Summary JSON: `experiments/archive/representation_and_objectives/data/curriculum_parity_probe/curriculum_fixed256_parity_summary.json`.

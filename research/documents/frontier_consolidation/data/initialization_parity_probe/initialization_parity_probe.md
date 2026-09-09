# initialization parity and interaction interpretation DeBERTa positional-ablation initialization parity probe

Created UTC: 2026-09-02T11:47:06Z

## Main finding

Same numeric seeds do not make the full and no-disentangle variants common-initialized: 53/140 same-named tensors differ before training. Therefore the running earlier analysis four-cell result remains a valid architecture-coordinate data-treatment interaction, but attenuation or survival must be interpreted as including architecture-dependent initialization stream changes, not as a perfectly common-weight ablation of only c2p/p2c score terms. A cleaner follow-up, if the earlier analysis result is scientifically important but ambiguous, is to train no-disentangle compact/repeat models initialized by copying all common tensors from the same full DeBERTa initialization while omitting the positional projection tensors.

## Same-seed tensor parity
- `full_vs_no_disentangle_same_seed`: common=140, exact=87, nonexact=53, only_left=32, only_right=0
  - mlm_head_after_optional: 5/7 exact, mismatched=2, max_abs_max=0.146995, mean_abs_mean=0.0225761
  - embeddings: 2/4 exact, mismatched=2, max_abs_max=0.146995, mean_abs_mean=0.0225507
  - attention_output_after_optional: 24/32 exact, mismatched=8, max_abs_max=0.153065, mean_abs_mean=0.0225635
  - attention_qkv: 24/48 exact, mismatched=24, max_abs_max=0.148378, mean_abs_mean=0.0225674
  - intermediate_after_optional: 8/16 exact, mismatched=8, max_abs_max=0.148278, mean_abs_mean=0.022558
  - layer_output_after_optional: 24/32 exact, mismatched=8, max_abs_max=0.142217, mean_abs_mean=0.0225595
  - encoder_rel_embeddings: 0/1 exact, mismatched=1, max_abs_max=0.126257, mean_abs_mean=0.0225321
- `full_vs_c2p_only_same_seed`: common=156, exact=95, nonexact=61, only_left=16, only_right=0
  - mlm_head_after_optional: 5/7 exact, mismatched=2, max_abs_max=0.148009, mean_abs_mean=0.0225749
  - embeddings: 2/4 exact, mismatched=2, max_abs_max=0.148009, mean_abs_mean=0.0225545
  - attention_output_after_optional: 24/32 exact, mismatched=8, max_abs_max=0.139364, mean_abs_mean=0.02256
  - attention_qkv: 24/48 exact, mismatched=24, max_abs_max=0.14458, mean_abs_mean=0.0225785
  - removed_pos_projection: 8/16 exact, mismatched=8, max_abs_max=0.149308, mean_abs_mean=0.0225375
  - intermediate_after_optional: 8/16 exact, mismatched=8, max_abs_max=0.158163, mean_abs_mean=0.0225639
  - layer_output_after_optional: 24/32 exact, mismatched=8, max_abs_max=0.154487, mean_abs_mean=0.0225589
  - encoder_rel_embeddings: 0/1 exact, mismatched=1, max_abs_max=0.126494, mean_abs_mean=0.0225504
- `full_vs_p2c_only_same_seed`: common=156, exact=95, nonexact=61, only_left=16, only_right=0
  - mlm_head_after_optional: 5/7 exact, mismatched=2, max_abs_max=0.148009, mean_abs_mean=0.0225749
  - embeddings: 2/4 exact, mismatched=2, max_abs_max=0.148009, mean_abs_mean=0.0225545
  - attention_output_after_optional: 24/32 exact, mismatched=8, max_abs_max=0.139364, mean_abs_mean=0.02256
  - attention_qkv: 24/48 exact, mismatched=24, max_abs_max=0.14458, mean_abs_mean=0.0225785
  - removed_pos_projection: 8/16 exact, mismatched=8, max_abs_max=0.148557, mean_abs_mean=0.0225598
  - intermediate_after_optional: 8/16 exact, mismatched=8, max_abs_max=0.158163, mean_abs_mean=0.0225639
  - layer_output_after_optional: 24/32 exact, mismatched=8, max_abs_max=0.154487, mean_abs_mean=0.0225589
  - encoder_rel_embeddings: 0/1 exact, mismatched=1, max_abs_max=0.126494, mean_abs_mean=0.0225504
- `c2p_only_vs_p2c_only_same_seed`: common=140, exact=140, nonexact=0, only_left=16, only_right=16
  - mlm_head_after_optional: 7/7 exact, mismatched=0, max_abs_max=0, mean_abs_mean=0
  - embeddings: 4/4 exact, mismatched=0, max_abs_max=0, mean_abs_mean=0
  - attention_output_after_optional: 32/32 exact, mismatched=0, max_abs_max=0, mean_abs_mean=0
  - attention_qkv: 48/48 exact, mismatched=0, max_abs_max=0, mean_abs_mean=0
  - intermediate_after_optional: 16/16 exact, mismatched=0, max_abs_max=0, mean_abs_mean=0
  - layer_output_after_optional: 32/32 exact, mismatched=0, max_abs_max=0, mean_abs_mean=0
  - encoder_rel_embeddings: 1/1 exact, mismatched=0, max_abs_max=0, mean_abs_mean=0

## Common-copied repair object

- copied common tensors into no-disentangle target: {'copied_common_tensors': 140, 'target_total_tensors': 140, 'skipped_shape_mismatch': []}
- after copy, common exact=140/140; remaining nonexact should be zero if copy worked.

## Initialization logits on sample compact-stream texts
- full_same_seed__vs__nodis_common_copied_from_full: mean_abs=0.00241652, rms=0.00303992, max_abs=0.0180106
- full_same_seed__vs__nodis_same_seed: mean_abs=0.494446, rms=0.619697, max_abs=3.44758
- nodis_common_copied_from_full__vs__nodis_same_seed: mean_abs=0.494446, rms=0.619699, max_abs=3.45078

## Boundary
CPU-only fresh initialization probe; no training, selected official-compatible evaluation, SuperGLUE, AoA, upload, or leaderboard submission.

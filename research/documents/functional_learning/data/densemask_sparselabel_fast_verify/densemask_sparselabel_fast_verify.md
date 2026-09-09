# earlier analysis dense-mask/sparse-label fast verification

Status: `PASS_FULLPREFIX_EXACT_SPARSE_LABELS_DENSE_INPUT`

Prefix: `20475` rows, `3162742` words, Qwen rows `3831`, Qwen pair segments `11778`.

## Core invariants

- rows_checked: `3831`
- no_rows_over_dense_cap: `True`
- seed62064_labels_equal_original_sparse_by_group_logic: `True`
- seed62064_all_labels_masked: `True`
- seed62064_masks_equal_original_dense: `True`
- seed62064_masks_equal_original_dense_union_sparse: `True`
- seed62065_labels_equal_original_sparse_by_group_logic: `True`
- seed62065_all_labels_masked: `True`
- seed62065_masks_equal_original_dense: `True`
- sample_actual_constructor_status: `sample_actual_constructor_pass`

## Policy totals

- sparse_seed62064: labels `21479` groups / `28590` tokens; masks `21479` groups / `28590` tokens; mask-only `0` tokens; label/mask token ratio `1.0`.
- dense_seed62064: labels `132283` groups / `176607` tokens; masks `132283` groups / `176607` tokens; mask-only `0` tokens; label/mask token ratio `1.0`.
- dense_seed62065: labels `132283` groups / `176607` tokens; masks `132283` groups / `176607` tokens; mask-only `0` tokens; label/mask token ratio `1.0`.
- densemask_sparselabel_seed62064: labels `21479` groups / `28590` tokens; masks `132283` groups / `176607` tokens; mask-only `148017` tokens; label/mask token ratio `0.1618848630009003`.
- densemask_sparselabel_seed62065: labels `21416` groups / `28476` tokens; masks `132283` groups / `176607` tokens; mask-only `148131` tokens; label/mask token ratio `0.16123936197319472`.

## Objective arithmetic

- `lambda_focus`: `0.15`; focus loss is a mean over retained focus labels, so per-token focus weight scales inversely with focus target count within each macro update.
- Dense target count / sparse target count: `6.177229800629591`; dense-mask/sparse-label target count / dense target count: `0.1618848630009003`.
- Against sparse focus, the control keeps the same labels and therefore the same focus-target weight schedule; against dense focus, each retained focus label receives about `6.188765426100415` times the per-label focus weight, while extra dense masked tokens are unsupervised context corruption.

## Interpretation

The seed62064 causal arm is an exact full-prefix control against sparse focus for labels and against dense focus for input masks. With lambda_focus=0.15 on mean focus loss, it keeps the sparse focus-target weight schedule while exposing the same dense-masked Qwen second-view input context.

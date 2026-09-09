# earlier analysis dense-mask/sparse-label fast verification

Status: `PASS_FULLPREFIX_EXACT_SPARSE_LABELS_DENSE_INPUT`

Prefix: `20475` rows, `3162742` words, Qwen rows `3831`, Qwen pair segments `11778`.

## Core invariants

- rows_checked: `3`
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

- sparse_seed62064: labels `15` groups / `19` tokens; masks `15` groups / `19` tokens; mask-only `0` tokens; label/mask token ratio `1.0`.
- dense_seed62064: labels `127` groups / `156` tokens; masks `127` groups / `156` tokens; mask-only `0` tokens; label/mask token ratio `1.0`.
- dense_seed62065: labels `127` groups / `156` tokens; masks `127` groups / `156` tokens; mask-only `0` tokens; label/mask token ratio `1.0`.
- densemask_sparselabel_seed62064: labels `15` groups / `19` tokens; masks `127` groups / `156` tokens; mask-only `137` tokens; label/mask token ratio `0.12179487179487179`.
- densemask_sparselabel_seed62065: labels `19` groups / `24` tokens; masks `127` groups / `156` tokens; mask-only `132` tokens; label/mask token ratio `0.15384615384615385`.

## Objective arithmetic

- `lambda_focus`: `0.15`; focus loss is a mean over retained focus labels, so per-token focus weight scales inversely with focus target count within each macro update.
- Dense target count / sparse target count: `8.210526315789474`; dense-mask/sparse-label target count / dense target count: `0.12179487179487179`.
- Against sparse focus, the control keeps the same labels and therefore the same focus-target weight schedule; against dense focus, each retained focus label receives about `8.210526315789474` times the per-label focus weight, while extra dense masked tokens are unsupervised context corruption.

## Interpretation

The seed62064 causal arm is an exact full-prefix control against sparse focus for labels and against dense focus for input masks. With lambda_focus=0.15 on mean focus loss, it keeps the sparse focus-target weight schedule while exposing the same dense-masked Qwen second-view input context.

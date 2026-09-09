# dense focus profile and next uncertainty dense/sparse input-label profile

Prefix: `20475` rows, `3162742` words, Qwen rows `3831`, Qwen pair segments `11778`.

## Policy totals

- sparse_seed62064: label groups `21479`, label tokens `28590`, mask groups `21479`, mask tokens `28590`, mask-only tokens `0`, label/mask token ratio `1.0`, label copied-token fraction `0.424833857992305`, mask copied-token fraction `0.424833857992305`.
- dense_seed62064: label groups `132283`, label tokens `176607`, mask groups `132283`, mask tokens `176607`, mask-only tokens `0`, label/mask token ratio `1.0`, label copied-token fraction `0.4171069096921413`, mask copied-token fraction `0.4171069096921413`.
- dense_seed62065: label groups `132283`, label tokens `176607`, mask groups `132283`, mask tokens `176607`, mask-only tokens `0`, label/mask token ratio `1.0`, label copied-token fraction `0.4171069096921413`, mask copied-token fraction `0.4171069096921413`.
- densemask_sparselabel_seed62064: label groups `21479`, label tokens `28590`, mask groups `132283`, mask tokens `176607`, mask-only tokens `148017`, label/mask token ratio `0.1618848630009003`, label copied-token fraction `0.424833857992305`, mask copied-token fraction `0.4171069096921413`.
- densemask_sparselabel_seed62065: label groups `21416`, label tokens `28476`, mask groups `132283`, mask tokens `176607`, mask-only tokens `148131`, label/mask token ratio `0.16123936197319472`, label copied-token fraction `0.4202135131338671`, mask copied-token fraction `0.4171069096921413`.

## Validation against actual completed training summaries

- sparse_seed62064: target-token diff profile-actual `0`, selected-group diff `0`, candidate-group diff `0`, actual targets `28590`, profile label tokens `28590`.
- dense_seed62064: target-token diff profile-actual `0`, selected-group diff `0`, candidate-group diff `0`, actual targets `176607`, profile label tokens `176607`.
- dense_seed62065: target-token diff profile-actual `0`, selected-group diff `0`, candidate-group diff `0`, actual targets `176607`, profile label tokens `176607`.

## Scientific use

The dense-mask/sparse-label control keeps supervised focus coverage near the sparse policy while exposing the model to dense second-view corruption. If full official evaluation retains source-responsive/Entity gains but costs erase the aggregate, this profile specifies a focused repair experiment separating input-side clue removal from dense supervised coverage.

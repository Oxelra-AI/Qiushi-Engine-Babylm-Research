# projected consistency calibration projected source-view consistency calibration

CPU-only calibration of fixed-subspace source/rewrite consistency losses on actual legal16k token-mean compact-view checkpoints.

No model update, official evaluation, corpus change, tokenizer change, or H100 work was performed.

## Inputs
- sample: `384` stream rows, batch `32`, max batches `4`
- projections: `32,64,128,480`; modes: `raw,row_centered`
- train SHA: `3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691`
- tokenizer SHA: `91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9`

## Per-checkpoint summaries
### tokenmean_20M
- raw d128: aux_loss=0.1311, pair-shuffle margin=0.1821, pair-same-row-other margin=0.1661, aux/MLM L2 all=0.2668, aux-span=0.9647, grad_cos=0.0027, λ@5%all≈0.1874
- raw d32: aux_loss=0.1742, pair-shuffle margin=0.2131, pair-same-row-other margin=0.1853, aux/MLM L2 all=0.7284, aux-span=2.6488, grad_cos=0.0015, λ@5%all≈0.06864
- raw d480: aux_loss=0.1327, pair-shuffle margin=0.1835, pair-same-row-other margin=0.1699, aux/MLM L2 all=0.1407, aux-span=0.5086, grad_cos=0.0053, λ@5%all≈0.3554
- raw d64: aux_loss=0.1077, pair-shuffle margin=0.1674, pair-same-row-other margin=0.1480, aux/MLM L2 all=0.3324, aux-span=1.2020, grad_cos=0.0018, λ@5%all≈0.1504
- row_centered d128: aux_loss=0.5949, pair-shuffle margin=0.6131, pair-same-row-other margin=0.7287, aux/MLM L2 all=1.0377, aux-span=3.7472, grad_cos=0.0035, λ@5%all≈0.04818
- row_centered d32: aux_loss=0.6089, pair-shuffle margin=0.5600, pair-same-row-other margin=0.6564, aux/MLM L2 all=2.2183, aux-span=8.0362, grad_cos=0.0019, λ@5%all≈0.02254
- row_centered d480: aux_loss=0.5924, pair-shuffle margin=0.6267, pair-same-row-other margin=0.7292, aux/MLM L2 all=0.5388, aux-span=1.9430, grad_cos=0.0068, λ@5%all≈0.0928
- row_centered d64: aux_loss=0.5729, pair-shuffle margin=0.6487, pair-same-row-other margin=0.7453, aux/MLM L2 all=1.5009, aux-span=5.4199, grad_cos=0.0025, λ@5%all≈0.03331
### tokenmean_80M
- raw d128: aux_loss=0.1081, pair-shuffle margin=0.2988, pair-same-row-other margin=0.2994, aux/MLM L2 all=0.1757, aux-span=0.5573, grad_cos=0.0021, λ@5%all≈0.2845
- raw d32: aux_loss=0.1384, pair-shuffle margin=0.3809, pair-same-row-other margin=0.3867, aux/MLM L2 all=0.4608, aux-span=1.4610, grad_cos=0.0009, λ@5%all≈0.1085
- raw d480: aux_loss=0.1027, pair-shuffle margin=0.3094, pair-same-row-other margin=0.3109, aux/MLM L2 all=0.0894, aux-span=0.2831, grad_cos=0.0029, λ@5%all≈0.5594
- raw d64: aux_loss=0.0926, pair-shuffle margin=0.3029, pair-same-row-other margin=0.3044, aux/MLM L2 all=0.2314, aux-span=0.7345, grad_cos=0.0008, λ@5%all≈0.2161
- row_centered d128: aux_loss=0.3667, pair-shuffle margin=0.8834, pair-same-row-other margin=0.9734, aux/MLM L2 all=0.5520, aux-span=1.7490, grad_cos=0.0021, λ@5%all≈0.09057
- row_centered d32: aux_loss=0.3559, pair-shuffle margin=0.8849, pair-same-row-other margin=0.9658, aux/MLM L2 all=1.1291, aux-span=3.5725, grad_cos=0.0010, λ@5%all≈0.04428
- row_centered d480: aux_loss=0.3499, pair-shuffle margin=0.9030, pair-same-row-other margin=0.9982, aux/MLM L2 all=0.2833, aux-span=0.8963, grad_cos=0.0028, λ@5%all≈0.1765
- row_centered d64: aux_loss=0.3322, pair-shuffle margin=0.9062, pair-same-row-other margin=1.0120, aux/MLM L2 all=0.7626, aux-span=2.4067, grad_cos=0.0007, λ@5%all≈0.06557
### tokenmean_100M
- raw d128: aux_loss=0.1061, pair-shuffle margin=0.3026, pair-same-row-other margin=0.3022, aux/MLM L2 all=0.1724, aux-span=0.5605, grad_cos=0.0019, λ@5%all≈0.2901
- raw d32: aux_loss=0.1393, pair-shuffle margin=0.4028, pair-same-row-other margin=0.4048, aux/MLM L2 all=0.4592, aux-span=1.4944, grad_cos=0.0006, λ@5%all≈0.1089
- raw d480: aux_loss=0.1011, pair-shuffle margin=0.3142, pair-same-row-other margin=0.3155, aux/MLM L2 all=0.0876, aux-span=0.2843, grad_cos=0.0023, λ@5%all≈0.571
- raw d64: aux_loss=0.0935, pair-shuffle margin=0.3108, pair-same-row-other margin=0.3100, aux/MLM L2 all=0.2311, aux-span=0.7519, grad_cos=0.0006, λ@5%all≈0.2164
- row_centered d128: aux_loss=0.3568, pair-shuffle margin=0.8871, pair-same-row-other margin=0.9779, aux/MLM L2 all=0.5395, aux-span=1.7518, grad_cos=0.0020, λ@5%all≈0.09267
- row_centered d32: aux_loss=0.3446, pair-shuffle margin=0.8992, pair-same-row-other margin=0.9753, aux/MLM L2 all=1.0917, aux-span=3.5413, grad_cos=0.0009, λ@5%all≈0.0458
- row_centered d480: aux_loss=0.3406, pair-shuffle margin=0.9091, pair-same-row-other margin=1.0055, aux/MLM L2 all=0.2760, aux-span=0.8947, grad_cos=0.0025, λ@5%all≈0.1812
- row_centered d64: aux_loss=0.3282, pair-shuffle margin=0.9095, pair-same-row-other margin=1.0116, aux/MLM L2 all=0.7475, aux-span=2.4196, grad_cos=0.0005, λ@5%all≈0.06689

## Interpretation
- This CPU run measures local hidden-state pressure and pair geometry for future construction; it is not BabyLM task-score evidence and does not justify a GPU run by itself.
- Raw cosine mostly measures already high same-row/topic anisotropy; row-centered variants subtract each row's hidden mean before span pooling, so they put more pressure on pair-specific residual content while still using only training-side source/rewrite spans.
- A future consistency trainer should use these measured λ scales with a small auxiliary weight and should not combine the auxiliary loss with fractional credit, minfreq50, static masking, or sequence changes in the first screen.

Full JSON: `experiments/archive/frontier_consolidation/data/projected_consistency_calibration/projected_consistency_calibration.json`

# consistency decoy diagnostic and route correction pair-specificity predictor probe

Test whether an asymmetric learnable predictor recovers pair-specific source<->rewrite structure that a symmetric agreement loss would collapse on, using same-row decoys.

CPU-only. No model update, official evaluation, corpus/tokenizer change, or H100 work.

## Inputs
- center_mode: `row_centered`; ridge_alpha `10.0`; train_frac `0.6`; splits `5`
- sample: `960` stream rows, batch `32`, max batches `30`
- train SHA: `3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691`
- tokenizer SHA: `91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9`

## Per-checkpoint results
### tokenmean_80M  (pairs=171, rows=43, mode=row_centered)
- symmetric: true_cos=0.6916, same_row_decoy_cos=-0.2752, true-minus-decoy=0.9668, within_row_top1=1.0000, mean_rank=1.000 (group size mean=4.10)
- ridge predictor holdout: true-minus-decoy=0.805196046595808, top1=1.0
- identity holdout:        true-minus-decoy=0.9681701835234722, top1=1.0
- predictor advantage: true-minus-decoy=-0.1629741369276642, top1=0.0 (n_ok_splits=5)
### tokenmean_100M  (pairs=171, rows=43, mode=row_centered)
- symmetric: true_cos=0.6951, same_row_decoy_cos=-0.2755, true-minus-decoy=0.9705, within_row_top1=1.0000, mean_rank=1.000 (group size mean=4.10)
- ridge predictor holdout: true-minus-decoy=0.8068050644844178, top1=1.0
- identity holdout:        true-minus-decoy=0.9722493039256019, top1=1.0
- predictor advantage: true-minus-decoy=-0.16544423944118417, top1=0.0 (n_ok_splits=5)

## Interpretation
- Symmetric agreement diag: if true_minus_same_row_decoy <= 0 in row-centered space, a naive positive-only symmetric agreement loss is maximized by encoding row/topic identity, not pair-specific correspondence -- collapse risk confirmed.
- Ridge predictor holdout: predictor_holdout_true_minus_decoy is the out-of-sample separation an asymmetric learnable predictor achieves against SAME-ROW decoys. If it is clearly > 0 and > the identity_holdout value, genuine pair-specific structure is recoverable and a predictor-based / shared-private construction has real headroom.
- If the predictor cannot beat same-row decoys out-of-sample, source-view consistency lacks a recoverable pair-specific signal on this substrate and should not be the successor GPU screen.
- This is representation-geometry evidence, not BabyLM task-score evidence, and does not by itself authorize any GPU run.

Full JSON: `experiments/archive/frontier_consolidation/data/pair_specificity_predictor_probe/pair_specificity_predictor_probe.json`

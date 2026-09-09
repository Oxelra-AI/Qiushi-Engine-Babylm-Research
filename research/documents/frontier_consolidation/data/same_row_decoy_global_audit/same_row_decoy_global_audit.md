# consistency decoy diagnostic and route correction same-row decoy global audit

Broad changed-row audit of true source<->rewrite residual similarity versus same-row decoys, correcting the projected consistency calibration margin interpretation.

CPU-only. No model update, official evaluation, corpus/tokenizer change, or H100 work.

## Inputs
- changed rows loaded: `768` of `3005` span-map rows, sample mode `stride`
- center_mode `row_centered`, projection dims `64,128,480`
- train SHA: `3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691`
- tokenizer SHA: `91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9`

## Per-checkpoint / per-projection results
### tokenmean_80M  (pairs=3093, rows=768, mode=row_centered)
- d64: true_cos=0.6856, same_row_decoy_cos=-0.2664, random_other_cos=-0.2162, true-decoy-mean=0.9731, true-decoy-max=0.7542, top1=0.9997, mean_rank=1.000
- d128: true_cos=0.7039, same_row_decoy_cos=-0.2676, random_other_cos=-0.2175, true-decoy-mean=0.9929, true-decoy-max=0.7881, top1=0.9997, mean_rank=1.000
- d480: true_cos=0.7006, same_row_decoy_cos=-0.2676, random_other_cos=-0.2171, true-decoy-mean=0.9894, true-decoy-max=0.7935, top1=1.0000, mean_rank=1.000
### tokenmean_100M  (pairs=3093, rows=768, mode=row_centered)
- d64: true_cos=0.6887, same_row_decoy_cos=-0.2665, random_other_cos=-0.2148, true-decoy-mean=0.9763, true-decoy-max=0.7606, top1=1.0000, mean_rank=1.000
- d128: true_cos=0.7075, same_row_decoy_cos=-0.2677, random_other_cos=-0.2161, true-decoy-mean=0.9967, true-decoy-max=0.7939, top1=1.0000, mean_rank=1.000
- d480: true_cos=0.7039, same_row_decoy_cos=-0.2677, random_other_cos=-0.2157, true-decoy-mean=0.9929, true-decoy-max=0.7997, top1=1.0000, mean_rank=1.000

## Interpretation
- The printed projected consistency calibration 'pair-same-row-other margin' is actual_minus_same_row_other, not the same-row-other cosine itself. A positive value means true source<->rewrite pairs are closer than same-row decoys.
- This audit uses exact changed rows from the pair-span map, so same-row decoys are available for nearly every pair and are matched by topic/row.
- If within_row_top1 is near 1 and true_minus_same_row_max is strongly positive, the collapse story that positive-only consistency merely learns generic row/topic agreement is not supported by representation geometry.
- Even if collapse-to-row is not supported, this is not BabyLM score evidence. The remaining question is whether forcing already-strong pair-specific residual agreement improves downstream learning without erasing private details; that should be constructed if minfreq50 is weak.

Full JSON: `experiments/archive/frontier_consolidation/data/same_row_decoy_global_audit/same_row_decoy_global_audit.json`

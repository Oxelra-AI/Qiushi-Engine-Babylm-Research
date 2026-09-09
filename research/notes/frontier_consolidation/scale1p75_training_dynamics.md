# dual mechanism 20m overlap — scale1.75 100M training-dynamics comparison

CPU-only comparison of spatial repair route status legal baseline and completed adapter128 scale1.75 100M endpoint training logs. It does not run model inference.

- Identical accounting fields: `True`; errors: `[]`.
- Log rows: spatial repair route status 2529, scale1.75 2529.
- Mismatch counts for step/LR/batch/exposure/seq/mask fields: `{'step': 0, 'lr': 0, 'batch_words': 0, 'cumulative_word_exposure': 0, 'seq_len': 0, 'masked_tokens': 0, 'effective_mask_rate': 0, 'mask_mode': 0, 'mask_prob_nominal': 0}`.
- Final loss: spatial repair route status 2.5525617599487305 vs scale1.75 2.5441808700561523 (delta -0.008381).

## Milestones

| target M | step | words | LR | masked | spatial repair route status loss | scale1.75 loss | Δloss |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 26 | 1027470 | 0.000172185 | 8467 | 8.252847 | 8.244710 | -0.008137 |
| 5 | 127 | 5020600 | 0.00084106 | 8792 | 5.870701 | 5.944993 | +0.074292 |
| 10 | 253 | 10004381 | 0.000995467 | 8313 | 4.341990 | 4.358886 | +0.016896 |
| 20 | 506 | 20008711 | 0.000946012 | 8546 | 3.755582 | 3.765515 | +0.009933 |
| 30 | 759 | 30012927 | 0.000847192 | 8605 | 3.410277 | 3.448519 | +0.038242 |
| 40 | 1012 | 40017538 | 0.000709945 | 8564 | 3.006902 | 2.983126 | -0.023776 |
| 50 | 1265 | 50021468 | 0.00054946 | 8462 | 2.786305 | 2.800712 | +0.014406 |
| 60 | 1518 | 60026138 | 0.000383502 | 8434 | 2.593466 | 2.609750 | +0.016284 |
| 70 | 1771 | 70030355 | 0.000230438 | 8573 | 2.624622 | 2.602167 | -0.022455 |
| 80 | 2024 | 80034368 | 0.000107209 | 8654 | 2.578651 | 2.583350 | +0.004699 |
| 90 | 2277 | 90038971 | 2.74538e-05 | 8487 | 2.535604 | 2.536050 | +0.000446 |
| 95 | 2403 | 95021029 | 6.91121e-06 | 8523 | 2.428128 | 2.400444 | -0.027684 |
| 99 | 2504 | 99014689 | 2.72682e-07 | 8255 | 2.486636 | 2.473474 | -0.013162 |
| 100 | 2529 | 100000000 | 0 | 7510 | 2.552562 | 2.544181 | -0.008381 |

## Loss windows

| window M | base mean | scale mean | Δmean | steps |
|---|---:|---:|---:|---:|
| 0-20 | 4.937210 | 4.945925 | +0.008715 | 505 |
| 20-50 | 3.152178 | 3.170024 | +0.017846 | 759 |
| 50-80 | 2.629192 | 2.624617 | -0.004574 | 759 |
| 80-100 | 2.476387 | 2.465365 | -0.011022 | 506 |
| 90-100 | 2.465900 | 2.454828 | -0.011072 | 253 |
| 95-100 | 2.455001 | 2.443225 | -0.011776 | 127 |
| 99-100 | 2.475104 | 2.464946 | -0.010158 | 26 |

Scientific consequence: the endpoint uses the same batch-word exposure, LR schedule, sequence length, and mask-count stream as spatial repair route status for every one of 2,529 updates. The official score difference, once measured, should be read as the effect of the separately routed residual branch on the learned trajectory rather than a hidden data/order/accounting change.

JSON: `experiments/archive/frontier_consolidation/data/scale1p75_training_dynamics/scale1p75_training_dynamics.json`

# counterfactual micro world v3 static repair micro-world v3 deeper interpretation

Status: **MICRO_WORLD_V3_INTERPRETED**

## Main scientific reading

1. The v3 scorer is mechanically coherent for the simple A/B/C families: mature models are high on crossed-sign success, shuffled cross-family contexts are low, and renamed controls are stable.
2. The aggregate is not a trustworthy surrogate for official full-EWoK interaction strength. The sign-aware correlations are weak for EWoK accuracy and have the wrong sign for stable failures: higher v3 success often comes with more EWoK stable failures. Legal40k/depth models have better official EWoK but lower v3 success than legal16k/scale1.75.
3. v3 does reject the sparse20 turnover artifact in one important sense: coupled aligned/shuffled do not score high on v3, even though they reduce EWoK stable-failure counts through turnover. However, that also shows v3 is strongly affected by broad trajectory damage/maturity; it is not a standalone selector for training.
4. The only non-saturated mature-model pocket is D_state_update_order: all mature checkpoints fail it while solving A/B/C. This is a possible real noncommutative final-state update weakness, but before using it scientifically we need a no-training surface/polarity ablation to test whether the templates themselves force the wrong answer through target priors or redundant-event wording.

## Family matrix

| target | target family | A crossed | B crossed | C crossed | D crossed | A median M | B median M | C median M | D median M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| scale1p75_77M | scale1p75_late_ladder | 1.0000 | 0.9900 | 1.0000 | 0.0000 | 7.3559 | 12.5698 | 7.3800 | -9.5688 |
| scale1p75_78M | scale1p75_late_ladder | 1.0000 | 0.9900 | 1.0000 | 0.0000 | 7.3552 | 11.5675 | 7.4971 | -10.2238 |
| scale1p75_79M | scale1p75_late_ladder | 1.0000 | 0.9900 | 1.0000 | 0.0000 | 7.3425 | 11.4608 | 7.3924 | -9.7684 |
| scale1p75_80M | scale1p75_late_ladder | 1.0000 | 0.9800 | 1.0000 | 0.0000 | 7.2240 | 10.0583 | 7.2839 | -9.0821 |
| scale1p75_81M | scale1p75_late_ladder | 1.0000 | 0.9600 | 1.0000 | 0.0000 | 7.5549 | 9.0343 | 7.6744 | -9.4241 |
| scale1p75_82M | scale1p75_late_ladder | 1.0000 | 0.9800 | 1.0000 | 0.0000 | 7.3960 | 10.7429 | 7.5843 | -9.5771 |
| scale1p75_83M | scale1p75_late_ladder | 1.0000 | 0.9800 | 1.0000 | 0.0000 | 7.3136 | 9.4665 | 7.4804 | -8.9120 |
| scale1p75_100M | scale1p75_late_ladder | 1.0000 | 0.9900 | 1.0000 | 0.0000 | 7.5772 | 10.8933 | 7.6323 | -9.4048 |
| legal16k_base_80M_seed43022 | legal16k_compact_seed43022 | 1.0000 | 0.9900 | 1.0000 | 0.0000 | 7.8717 | 10.5392 | 7.5537 | -8.4883 |
| legal16k_base_100M_seed43022 | legal16k_compact_seed43022 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 8.2910 | 11.9714 | 7.6698 | -9.1962 |
| legal16k_80M_seed43122 | legal16k_compact_seed43122 | 1.0000 | 0.9800 | 1.0000 | 0.0000 | 8.0149 | 12.9034 | 8.2062 | -7.7860 |
| legal16k_100M_seed43122 | legal16k_compact_seed43122 | 1.0000 | 0.9800 | 1.0000 | 0.0000 | 8.2272 | 10.3905 | 8.2623 | -7.1348 |
| legal40k_8x480_100M_seed43022 | legal40k_8x480 | 1.0000 | 0.8200 | 1.0000 | 0.0000 | 7.0332 | 5.9230 | 7.4327 | -8.3699 |
| legal40k_depth12_100M_seed43022 | legal40k_depth12 | 0.9900 | 0.7600 | 1.0000 | 0.0000 | 6.2153 | 5.1921 | 7.1877 | -9.4472 |
| fw_compact_100M_seed43022 | fineweb_allocation | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 8.9000 | 16.0298 | 8.9013 | -9.9174 |
| fw_rowblock_100M_seed43022 | fineweb_allocation | 1.0000 | 0.9900 | 1.0000 | 0.0000 | 5.5059 | 13.1315 | 5.4969 | -6.3855 |
| mlm_only_20M | coupled_sparse20_turnover_panel | 0.3500 | 0.3000 | 0.3000 | 0.0000 | 1.1112 | 1.2281 | 0.6174 | -1.8319 |
| coupled_aligned_20M | coupled_sparse20_turnover_panel | 0.0000 | 0.0200 | 0.0000 | 0.0000 | 0.0322 | 0.0325 | 0.0295 | -0.0216 |
| coupled_shuffled_20M | coupled_sparse20_turnover_panel | 0.0400 | 0.0000 | 0.0200 | 0.0000 | 0.0258 | -0.0144 | 0.0187 | -0.0590 |

## Corrected coupled-turnover reading

```json
{
  "aligned_crossed": 0.005,
  "aligned_minus_mlm": -0.23249999999999998,
  "aligned_minus_shuffled": -0.009999999999999998,
  "full_ewok_aligned_stable_frac": 0.310580204778157,
  "full_ewok_mlm_stable_frac": 0.3431346810186401,
  "full_ewok_shuffled_stable_frac": 0.29364662641113154,
  "mlm_crossed": 0.2375,
  "reading": "v3 does not reward the sparse20 full-EWoK turnover artifact: both coupled variants have far lower crossed-sign success than MLM-only, and aligned is not better than shuffled. This is useful as a rejection of the sparse20 mechanism, but it also means v3 is strongly sensitive to broad trajectory damage/maturity and cannot by itself be a training selector.",
  "shuffled_crossed": 0.015,
  "shuffled_minus_mlm": -0.22249999999999998
}
```

## D-state summary

| target | D crossed | D shuffled | Δ1 med | Δ2 med | M med | min-margin med | no-context Δ med |
|---|---:|---:|---:|---:|---:|---:|---:|
| scale1p75_77M | 0.0000 | 0.1600 | -6.4374 | 4.4299 | -9.5688 | -6.4374 | -0.0888 |
| scale1p75_78M | 0.0000 | 0.1300 | -6.5887 | 4.9817 | -10.2238 | -6.5887 | -0.3636 |
| scale1p75_79M | 0.0000 | 0.1300 | -6.7797 | 4.6314 | -9.7684 | -6.7797 | -0.3900 |
| scale1p75_80M | 0.0000 | 0.0700 | -5.7960 | 4.5254 | -9.0821 | -5.7960 | -0.0872 |
| scale1p75_81M | 0.0000 | 0.2000 | -6.6919 | 3.8982 | -9.4241 | -6.6919 | -0.5903 |
| scale1p75_82M | 0.0000 | 0.1600 | -6.5744 | 3.9811 | -9.5771 | -6.5744 | -0.7039 |
| scale1p75_83M | 0.0000 | 0.1500 | -6.0580 | 3.9578 | -8.9120 | -6.0580 | -0.4930 |
| scale1p75_100M | 0.0000 | 0.1300 | -5.8922 | 4.5502 | -9.4048 | -5.8922 | -0.0486 |
| legal16k_base_80M_seed43022 | 0.0000 | 0.1300 | -4.1603 | 3.8434 | -8.4883 | -4.9291 | 0.6653 |
| legal16k_base_100M_seed43022 | 0.0000 | 0.0800 | -4.3500 | 4.5779 | -9.1962 | -4.8293 | 1.0327 |
| legal16k_80M_seed43122 | 0.0000 | 0.0900 | -3.6858 | 4.4332 | -7.7860 | -4.4332 | 0.1924 |
| legal16k_100M_seed43122 | 0.0000 | 0.1100 | -3.5207 | 4.0055 | -7.1348 | -4.0055 | 0.3282 |
| legal40k_8x480_100M_seed43022 | 0.0000 | 0.1400 | -5.0428 | 3.6224 | -8.3699 | -5.1589 | -0.5079 |
| legal40k_depth12_100M_seed43022 | 0.0000 | 0.0600 | -4.6483 | 5.2521 | -9.4472 | -5.2683 | -0.4138 |
| fw_compact_100M_seed43022 | 0.0000 | 0.2000 | -4.3995 | 4.3966 | -9.9174 | -5.5921 | -0.1206 |
| fw_rowblock_100M_seed43022 | 0.0000 | 0.2700 | -3.3690 | 3.7044 | -6.3855 | -3.7044 | 0.8092 |
| mlm_only_20M | 0.0000 | 0.0000 | -1.8119 | 0.4678 | -1.8319 | -1.8651 | -0.3504 |
| coupled_aligned_20M | 0.0000 | 0.0000 | -1.1440 | -1.1727 | -0.0216 | -1.1440 | -1.2843 |
| coupled_shuffled_20M | 0.0000 | 0.0000 | -1.6347 | -1.5674 | -0.0590 | -1.6347 | -1.0531 |

JSON: `experiments/archive/representation_and_objectives/data/counterfactual_micro_world_v3_scores/micro_world_v3_deeper_interpretation.json`

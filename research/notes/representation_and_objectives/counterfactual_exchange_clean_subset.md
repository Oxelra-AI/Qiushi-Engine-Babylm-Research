# counterfactual exchange clean subset — clean subset of counterfactual exchange probe
Created: 2026-08-31T05:14:40Z
Clean cases: **324** from scored pilot; row counts by clean family: `{'clean_temporal_order': 516, 'clean_causal_direction': 162, 'clean_spatial_vertical': 171, 'clean_comparative_scalar': 99, 'clean_comparative_more_less': 24}`.
Rejects: `{'filter_reject': 1728}`.

## Arm means
| metric | compact | rowblock | interleaved | rank | range |
|---|---:|---:|---:|---|---:|
| true_m | 0.29640022619270984 | -0.45253619146935736 | -0.11513903700275185 | compact > interleaved > rowblock | 0.7489364176620672 |
| target_perm_m | -0.007710109522313248 | -0.0047097588762824915 | 0.034353947933809255 | interleaved > rowblock > compact | 0.0420640574561225 |
| ctx0_margin | 0.03589323714927391 | -0.047419474448686764 | -0.20079821716120214 | compact > rowblock > interleaved | 0.23669145431047606 |
| ctx1_margin | 0.26050698904343594 | -0.4051167170206706 | 0.08565918015845028 | compact > interleaved > rowblock | 0.6656237060641066 |
| target_main_bias | -0.22461375189416202 | 0.3576972425719838 | -0.2864573973196524 | rowblock > compact > interleaved | 0.6441546398916362 |

## Clean-family true_m means
| family | n | compact | rowblock | interleaved | rank | range |
|---|---:|---:|---:|---:|---|---:|
| clean_causal_direction | 54 | 0.6135152004383229 | -0.21008329038266782 | 0.10364451231779875 | compact > interleaved > rowblock | 0.8235984908209907 |
| clean_comparative_more_less | 8 | 0.0017242431640625 | 0.06607043743133545 | 0.11601638793945312 | interleaved > rowblock > compact | 0.11429214477539062 |
| clean_comparative_scalar | 33 | 0.06381687973484848 | 0.04417653517289595 | 0.13693693912390506 | interleaved > compact > rowblock | 0.09276040395100911 |
| clean_spatial_vertical | 57 | 0.0217060457196152 | 0.8651401620162161 | -0.765420445224695 | rowblock > compact > interleaved | 1.6305606072409111 |
| clean_temporal_order | 172 | 0.3462026562801627 | -1.0847477136656296 | -0.027441490528195403 | compact > interleaved > rowblock | 1.4309503699457924 |

Files: `experiments/archive/representation_and_objectives/data/counterfactual_exchange_clean_subset/clean_subset_summary.json`, `experiments/archive/representation_and_objectives/data/counterfactual_exchange_clean_subset/clean_subset_scores.jsonl`, `experiments/archive/representation_and_objectives/data/counterfactual_exchange_clean_subset/clean_subset_samples.jsonl`.

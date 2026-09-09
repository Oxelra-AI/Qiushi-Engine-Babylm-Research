# counterfactual exchange clean subset — clean subset of counterfactual exchange probe
Created: 2026-08-31T05:21:48Z
Clean cases: **892** from scored pilot; row counts by clean family: `{'clean_temporal_order': 1689, 'clean_causal_direction': 270, 'clean_spatial_vertical': 468, 'clean_comparative_scalar': 216, 'clean_comparative_more_less': 33}`.
Rejects: `{'filter_reject': 4440}`.

## Arm means
| metric | compact | rowblock | interleaved | rank | range |
|---|---:|---:|---:|---|---:|
| true_m | 0.29548659132200505 | -0.5780679698482223 | -0.1300846054949568 | compact > interleaved > rowblock | 0.8735545611702273 |
| target_perm_m | -0.009537295910275035 | 2.7286632178610217e-05 | 0.008365330674723125 | interleaved > rowblock > compact | 0.01790262658499816 |
| ctx0_margin | -0.1767191202651225 | -0.2219693126165279 | -0.17527015006061092 | interleaved > compact > rowblock | 0.04669916255591697 |
| ctx1_margin | 0.47220571158712754 | -0.35609865723169437 | 0.04518554456565412 | compact > interleaved > rowblock | 0.8283043688188219 |
| target_main_bias | -0.64892483185225 | 0.13412934461516648 | -0.22045569462626505 | rowblock > interleaved > compact | 0.7830541764674165 |

## Clean-family true_m means
| family | n | compact | rowblock | interleaved | rank | range |
|---|---:|---:|---:|---:|---|---:|
| clean_causal_direction | 90 | 0.7276170412699382 | -0.21656885147094726 | 0.14744671715630425 | compact > interleaved > rowblock | 0.9441858927408855 |
| clean_comparative_more_less | 11 | -0.021206942471590908 | 0.11172199249267578 | 0.14936135031960227 | interleaved > rowblock > compact | 0.17056829279119318 |
| clean_comparative_scalar | 72 | -0.04405809773339166 | 0.007838924725850424 | 0.0474598937564426 | interleaved > rowblock > compact | 0.09151799148983425 |
| clean_spatial_vertical | 156 | 0.022575574043469552 | 0.6858847263531808 | -0.7709052990644406 | rowblock > compact > interleaved | 1.4567900254176214 |
| clean_temporal_order | 563 | 0.3516379673146439 | -1.074488089309069 | -0.025052456830365197 | compact > interleaved > rowblock | 1.4261260566237128 |

Files: `experiments/archive/representation_and_objectives/data/counterfactual_exchange_clean_subset_full/clean_subset_summary.json`, `experiments/archive/representation_and_objectives/data/counterfactual_exchange_clean_subset_full/clean_subset_scores.jsonl`, `experiments/archive/representation_and_objectives/data/counterfactual_exchange_clean_subset_full/clean_subset_samples.jsonl`.

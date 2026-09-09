# counterfactual exchange clean subset — rule-certified counterfactual exchange probe
Created: 2026-08-31T05:08:02Z
## Build
Cases scored: **1200** / input 1200
Generated pair word count if trained once: **35372** words; by family: `{'causal_direction': 592, 'comparative_more_less': 28, 'comparative_scalar': 194, 'spatial_vertical': 632, 'temporal_order': 554}`.

## Frozen checkpoint arm means
| metric | compact | rowblock | interleaved | rank | range |
|---|---:|---:|---:|---|---:|
| true_m | 0.38320605834325155 | -0.46713836669921877 | 0.6395964805285136 | interleaved > compact > rowblock | 1.1067348472277323 |
| target_perm_m | 0.005967765649159749 | -0.020125753084818523 | -0.014251034259796142 | compact > interleaved > rowblock | 0.026093518733978274 |
| ctx0_margin | 1.6200794688860576 | 0.7734205977121988 | 0.8223600792884826 | compact > interleaved > rowblock | 0.8466588711738587 |
| ctx1_margin | -1.236873410542806 | -1.2405589644114177 | -0.1827635987599691 | interleaved > compact > rowblock | 1.0577953656514485 |
| target_main_bias | 2.8569528794288637 | 2.0139795621236165 | 1.0051236780484518 | compact > rowblock > interleaved | 1.8518292013804118 |

## Family true_m means
| family | n | compact | rowblock | interleaved | rank | range |
|---|---:|---:|---:|---:|---|---:|
| causal_direction | 400 | 0.4818612170219421 | -0.07172355890274047 | 1.1017895865440368 | interleaved > compact > rowblock | 1.1735131454467773 |
| comparative_more_less | 28 | 0.24819292340959823 | 0.09075924328395299 | -0.20928410121372767 | compact > rowblock > interleaved | 0.4574770246233259 |
| comparative_scalar | 193 | 0.10876814689043272 | -0.056705222846312846 | 0.040809463342854395 | compact > interleaved > rowblock | 0.16547336973674556 |
| spatial_vertical | 400 | 0.4278945541381836 | -0.7635663151741028 | 0.8646227288246154 | interleaved > compact > rowblock | 1.6281890439987183 |
| temporal_order | 179 | 0.3799067449303313 | -1.218141012351606 | -0.11768391678453158 | compact > interleaved > rowblock | 1.5980477572819374 |

## Interpretation hook
The object is a no-update probe. It is useful as a possible training object only if true exchange margins are not merely saturated local copying, if arm separation exceeds the target-permutation control, and if the direction follows the known relation-hard-row tradeoff rather than the compact broad-capability ordering.

Files:
- summary: `experiments/archive/representation_and_objectives/data/counterfactual_exchange_probe_pilot20k_v2/counterfactual_exchange_score_summary.json`
- scores: `experiments/archive/representation_and_objectives/data/counterfactual_exchange_probe_pilot20k_v2/counterfactual_exchange_scores.jsonl`
- csv: `experiments/archive/representation_and_objectives/data/counterfactual_exchange_probe_pilot20k_v2/counterfactual_exchange_scores.csv`

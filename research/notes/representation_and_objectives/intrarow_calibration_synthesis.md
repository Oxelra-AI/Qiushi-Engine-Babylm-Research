# intrarow calibration synthesis — intra-row calibration synthesis

Common scored pairs: 10,377. This is a file-level synthesis of the pilot no-update four-cell scores; no model is loaded.

## Key subset table
| subset | n | true order | true range | rowblock-compact true mean | rowblock>true frac | compact true mean | low compact<2 frac | tperm range | cperm range |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| all | 10377 | compact > rowblock > interleaved | 0.4510 | -0.4492 | 0.456 | 13.4463 | 0.093 | 0.0222 | 0.0110 |
| not_bad_targets | 9686 | compact > interleaved > rowblock | 0.4709 | -0.4709 | 0.454 | 13.2691 | 0.098 | 0.0260 | 0.0110 |
| low_mean_true_lt_2 | 960 | compact > interleaved > rowblock | 0.0339 | -0.0339 | 0.502 | -0.9237 | 0.893 | 0.0708 | 0.1067 |
| low_mean_true_lt_5 | 1884 | compact > rowblock > interleaved | 0.1262 | -0.0913 | 0.499 | 1.3399 | 0.512 | 0.0200 | 0.0222 |
| compact_true_lt_2 | 970 | rowblock > interleaved > compact | 0.4744 | 0.4744 | 0.599 | -1.0702 | 1.000 | 0.1088 | 0.0549 |
| rowblock_beats_compact_true | 4727 | rowblock > interleaved > compact | 1.6261 | 1.6261 | 1.000 | 11.3710 | 0.123 | 0.0424 | 0.0315 |
| rowblock_beats_compact_true_not_bad | 4396 | rowblock > interleaved > compact | 1.6362 | 1.6362 | 1.000 | 11.1534 | 0.130 | 0.0393 | 0.0276 |
| balanced_distance | 5561 | compact > rowblock > interleaved | 0.5051 | -0.4933 | 0.452 | 13.7571 | 0.094 | 0.0291 | 0.0259 |
| causal_connector | 2456 | compact > rowblock > interleaved | 0.3406 | -0.3395 | 0.465 | 13.4988 | 0.080 | 0.0531 | 0.0322 |
| comparative | 1346 | compact > rowblock > interleaved | 0.5945 | -0.5527 | 0.442 | 13.8826 | 0.088 | 0.0797 | 0.0404 |
| spatial | 2987 | compact > interleaved > rowblock | 0.5097 | -0.5097 | 0.448 | 13.1118 | 0.116 | 0.0237 | 0.0334 |
| temporal | 3588 | compact > rowblock > interleaved | 0.4409 | -0.4352 | 0.461 | 13.5250 | 0.086 | 0.0575 | 0.0296 |

## Top pivots by rowblock-minus-compact true margin, n>=50
| pivot | n | true order | rb-compact | compact true mean | low mean<5 frac |
|---|---:|---|---:|---:|---:|
| towards | 53 | rowblock > compact > interleaved | 0.2985 | 11.4941 | 0.245 |
| onto | 58 | rowblock > compact > interleaved | 0.1550 | 14.5832 | 0.138 |
| once | 213 | compact > interleaved > rowblock | -0.0440 | 12.0657 | 0.211 |
| if | 1330 | compact > rowblock > interleaved | -0.1002 | 13.1326 | 0.154 |
| though | 246 | interleaved > compact > rowblock | -0.1202 | 13.0484 | 0.175 |
| above | 127 | compact > rowblock > interleaved | -0.1447 | 15.0105 | 0.150 |
| when | 1312 | compact > rowblock > interleaved | -0.2784 | 13.2123 | 0.171 |
| into | 723 | compact > interleaved > rowblock | -0.2938 | 12.6912 | 0.213 |
| before | 508 | compact > rowblock > interleaved | -0.2949 | 12.8928 | 0.154 |
| within | 128 | compact > rowblock > interleaved | -0.3591 | 14.8560 | 0.188 |
| inside | 98 | compact > interleaved > rowblock | -0.3862 | 13.4073 | 0.184 |
| outside | 116 | compact > rowblock > interleaved | -0.4066 | 12.6360 | 0.259 |
| better | 261 | compact > rowblock > interleaved | -0.4122 | 14.2012 | 0.176 |
| behind | 130 | compact > interleaved > rowblock | -0.4463 | 12.0133 | 0.223 |
| more | 215 | compact > rowblock > interleaved | -0.4625 | 13.9073 | 0.191 |
| since | 168 | compact > rowblock > interleaved | -0.4665 | 14.3537 | 0.119 |
| than | 484 | compact > interleaved > rowblock | -0.4913 | 13.5453 | 0.205 |
| until | 245 | compact > interleaved > rowblock | -0.5709 | 13.1547 | 0.216 |
| between | 248 | compact > rowblock > interleaved | -0.6223 | 13.7071 | 0.194 |
| after | 563 | compact > interleaved > rowblock | -0.6433 | 13.6400 | 0.197 |

Files:
- summary JSON: `experiments/archive/representation_and_objectives/data/intrarow_calibration_synthesis/intrarow_calibration_synthesis.json`
- low-margin examples: `experiments/archive/representation_and_objectives/data/intrarow_calibration_synthesis/low_mean_true_lt_5.jsonl`
- rowblock-positive examples: `experiments/archive/representation_and_objectives/data/intrarow_calibration_synthesis/rowblock_beats_compact_true_not_bad.jsonl`

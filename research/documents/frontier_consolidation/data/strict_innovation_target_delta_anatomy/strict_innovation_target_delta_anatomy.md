# strict innovation target readout before scores strict-innovation target paired deltas

Paired target-level deltas on the fixed strict-innovation sample, with row-bootstrap intervals. Negative true_loss delta means easier target prediction; positive source_help/same_decoy deltas mean stronger source-conditioned dependence.

| comparison | n | rows | Δ true loss | Δ source help | Δ same decoy adv | Δ cross decoy adv | Δ masked-same |
|---|---:|---:|---:|---:|---:|---:|---:|
| tokenmean_80M_minus_tokenmean_70M | 320 | 263 | -0.2010 [-0.2713,-0.1251] | 0.1666 [0.0886,0.2464] | 0.1947 [0.1141,0.2688] | 0.1155 [0.0318,0.1869] | -0.0281 |
| tokenmean_100M_minus_tokenmean_80M | 320 | 263 | -0.0811 [-0.1203,-0.0447] | 0.0573 [0.0166,0.0934] | 0.0804 [0.0400,0.1221] | 0.0423 [0.0062,0.0815] | -0.0231 |
| tokenmean_80M_minus_clean_80M | 320 | 263 | -1.0175 [-1.2063,-0.8448] | 0.4418 [0.2678,0.6065] | 0.6129 [0.4275,0.8141] | 0.4928 [0.3228,0.6620] | -0.1711 |
| reference_70M_minus_tokenmean_70M | 320 | 263 | -0.8061 [-0.9618,-0.6736] | 0.2846 [0.1263,0.4335] | 0.0484 [-0.1258,0.2146] | 0.1805 [0.0170,0.3196] | 0.2363 |
| reference_80M_minus_tokenmean_80M | 320 | 263 | -0.6476 [-0.8020,-0.5136] | 0.1532 [0.0126,0.3028] | -0.0865 [-0.2401,0.0826] | 0.1110 [-0.0314,0.2493] | 0.2397 |

## Reading
- Tokenmean 80M versus clean 80M quantifies how much the compact-view reinvest substrate already improves this exact target object.
- Tokenmean 80M->100M quantifies natural late-exposure movement on the same target sample, useful as a reference for earlier analysis target deltas.
- earlier analysis deltas are absent until managed checkpoints are delivered and the target probe is rerun including earlier analysis labels.

Full JSON: `experiments/archive/frontier_consolidation/data/strict_innovation_target_delta_anatomy/strict_innovation_target_delta_anatomy.json`

# strict innovation target readout before scores strict row-unique innovation target probe

Fixed CPU readout of repaired row-unique content-innovation target prediction for token-mean and, once available, earlier analysis checkpoints.

- examples loaded: `768` / requested `768`
- selected targets: `320`; selected rows: `263`; token labels: `493`
- target manifest: `experiments/archive/frontier_consolidation/data/strict_innovation_target_probe/strict_innovation_target_manifest.jsonl`
- train SHA: `3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691`
- tokenizer SHA: `91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9`

## Results
| checkpoint | status | n | true loss | source help | same-row decoy advantage | cross-row decoy advantage | masked-same | source help bootstrap 5-95 | same decoy bootstrap 5-95 |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| tokenmean_70M | ok | 320 | 5.5737 | 1.0553 | 1.6693 | 1.0895 | -0.6140 | [0.8594,1.2782] | [1.4041,1.9303] |
| tokenmean_80M | ok | 320 | 5.3727 | 1.2219 | 1.8640 | 1.2051 | -0.6421 | [1.0380,1.4296] | [1.6127,2.1330] |
| tokenmean_100M | ok | 320 | 5.2916 | 1.2792 | 1.9444 | 1.2474 | -0.6653 | [1.0674,1.4898] | [1.6807,2.2150] |
| clean_80M | ok | 320 | 6.3903 | 0.7800 | 1.2511 | 0.7123 | -0.4711 | [0.5479,1.0058] | [1.0033,1.4674] |
| reference_70M | ok | 320 | 4.7676 | 1.3399 | 1.7176 | 1.2700 | -0.3778 | [1.1485,1.5170] | [1.5125,1.9643] |
| reference_80M | ok | 320 | 4.7251 | 1.3750 | 1.7775 | 1.3161 | -0.4025 | [1.1842,1.5554] | [1.5757,2.0185] |

## Direct comparisons
- `reference_70M_minus_tokenmean_70M`: delta_step075_70M_minus_tokenmean_70M_true_loss=-0.8061, delta_step075_70M_minus_tokenmean_70M_source_help=0.2846, delta_step075_70M_minus_tokenmean_70M_same_decoy_advantage=0.0484, delta_step075_70M_minus_tokenmean_70M_cross_decoy_advantage=0.1805, delta_step075_70M_minus_tokenmean_70M_masked_minus_same_decoy=0.2363
- `reference_80M_minus_tokenmean_80M`: delta_step075_80M_minus_tokenmean_80M_true_loss=-0.6476, delta_step075_80M_minus_tokenmean_80M_source_help=0.1532, delta_step075_80M_minus_tokenmean_80M_same_decoy_advantage=-0.0865, delta_step075_80M_minus_tokenmean_80M_cross_decoy_advantage=0.1110, delta_step075_80M_minus_tokenmean_80M_masked_minus_same_decoy=0.2397
- `tokenmean_80M_minus_clean_80M`: delta_tokenmean_80M_minus_clean_80M_true_loss=-1.0175, delta_tokenmean_80M_minus_clean_80M_source_help=0.4418, delta_tokenmean_80M_minus_clean_80M_same_decoy_advantage=0.6129, delta_tokenmean_80M_minus_clean_80M_cross_decoy_advantage=0.4928, delta_tokenmean_80M_minus_clean_80M_masked_minus_same_decoy=-0.1711
- `tokenmean_100M_minus_tokenmean_80M`: delta_tokenmean_100M_minus_tokenmean_80M_true_loss=-0.0811, delta_tokenmean_100M_minus_tokenmean_80M_source_help=0.0573, delta_tokenmean_100M_minus_tokenmean_80M_same_decoy_advantage=0.0804, delta_tokenmean_100M_minus_tokenmean_80M_cross_decoy_advantage=0.0423, delta_tokenmean_100M_minus_tokenmean_80M_masked_minus_same_decoy=-0.0231

## Reading
- This readout is not a BabyLM score; it measures whether the exact target class selected for innovation-biased masking becomes easier under true source context.
- For earlier analysis, a useful mechanism signature is lower true_loss and/or higher source_help on these strict targets versus tokenmean at the same exposure, interpreted together with the official-compatible cheap-column trajectory.
- If broad transfer weakens and this target readout does not improve, the innovation-masking family should not be carried forward just because exact-swap is mechanically cleaner.

Full JSON: `experiments/archive/frontier_consolidation/data/strict_innovation_target_probe/strict_innovation_target_probe.json`

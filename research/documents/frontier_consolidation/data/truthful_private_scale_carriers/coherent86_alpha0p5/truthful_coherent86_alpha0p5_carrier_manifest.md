# Truthful private-scale carrier — coherent86_alpha0p5

Status: **TRUTHFUL_PRIVATE_SCALE_CARRIER_MATERIALIZED**

Carrier: `experiments/archive/frontier_consolidation/data/truthful_private_scale_carriers/coherent86_alpha0p5/all_full_preds_truthful_coherent86_alpha0p5_mlm.json`
Carrier SHA256: `2f4dc195261062ee34010386c74de514133b5888276ba93b27951e731c723a58`
Model: `experiments/archive/frontier_consolidation/training/runs/coherent86_private_scale_0p5/hf_model/final`
Model SHA256: `e14d757ae51b41e33bf0813f841248fecd1eefeb9e040f520c4c6203343b15c8`
Private adapter scale: `0.5`

## Score arithmetic with AoA=0

| column | score | delta vs protected chck82 |
|---|---:|---:|
| BLiMP | 68.54 | +0.0487159634801 |
| Supplement | 63.15 | +0.2121887438 |
| EWoK | 50 | -0.0554533255328 |
| Entity | 28.23 | -0.0840419302988 |
| COMPS | 52.11 | -0.081175094436 |
| GlobalPIQA | 39.05 | +1.47233009709 |
| Reading | 8.165 | +0.0162864107381 |
| SuperGLUE | 69.7897551573 | +0.02357378595 |
| AoA | 0 | +0 |

Cheap7: `44.17785714285714` (delta `+0.218407266405`)
Overall(AoA0): `42.11497279525131` (delta `+0.172491627865`)

## Truthful history policy
- AoA is scalar zero only.
- No `fast_eval_results` are copied.
- Cheap and SuperGLUE predictions come from the exact materialized alpha endpoint.

## Panel reading
Panel arm: `coherent86_private_alpha0p5`
Vs-anchor aggregate: `{'total_common_items': 170722, 'total_gain_items': 1560, 'total_loss_items': 1581, 'total_gain_minus_loss': -21, 'total_changed_items': 3141, 'total_base_correct_items': 98478, 'anchor_correct_retention_fraction': 0.9839456528361664, 'gain_to_loss_ratio': 0.9867172675521821, 'discrete_payload_mean_delta': 0.25209407568330516, 'discrete_reconstructed_mean_delta': 0.2511697430303326}`

No-training private-scale carrier. Endpoint arithmetic may improve, but the mechanism remains amplitude-controlled redistribution unless item evidence shows broad positive, stable decision addition.

JSON: `experiments/archive/frontier_consolidation/data/truthful_private_scale_carriers/coherent86_alpha0p5/truthful_coherent86_alpha0p5_carrier_manifest.json`

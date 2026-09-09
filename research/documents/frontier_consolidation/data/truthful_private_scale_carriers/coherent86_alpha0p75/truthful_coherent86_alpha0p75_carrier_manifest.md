# Truthful private-scale carrier — coherent86_alpha0p75

Status: **TRUTHFUL_PRIVATE_SCALE_CARRIER_MATERIALIZED**

Carrier: `experiments/archive/frontier_consolidation/data/truthful_private_scale_carriers/coherent86_alpha0p75/all_full_preds_truthful_coherent86_alpha0p75_mlm.json`
Carrier SHA256: `40181994810e21bc823474a3e4ac84c8eb42213e03904d36a60a4698477d1994`
Model: `models/frontier`
Model SHA256: `e14d757ae51b41e33bf0813f841248fecd1eefeb9e040f520c4c6203343b15c8`
Private adapter scale: `0.75`

## Score arithmetic with AoA=0

| column | score | delta vs protected chck82 |
|---|---:|---:|
| BLiMP | 68.51 | +0.0187159634801 |
| Supplement | 63.64 | +0.7021887438 |
| EWoK | 50.02 | -0.0354533255328 |
| Entity | 28.32 | +0.00595806970123 |
| COMPS | 52.05 | -0.141175094436 |
| GlobalPIQA | 38.565 | +0.987330097087 |
| Reading | 8.165 | +0.0162864107381 |
| SuperGLUE | 69.8192223897 | +0.0530410183875 |
| AoA | 0 | +0 |

Cheap7: `44.18142857142857` (delta `+0.221978694977`)
Overall(AoA0): `42.1210247099666` (delta `+0.178543542581`)

## Truthful history policy
- AoA is scalar zero only.
- No `fast_eval_results` are copied.
- Cheap and SuperGLUE predictions come from the exact materialized alpha endpoint.

## Panel reading
Panel arm: `a0p75`
Vs-anchor aggregate: `None`

No-training private-scale carrier. Endpoint arithmetic may improve, but the mechanism remains amplitude-controlled redistribution unless item evidence shows broad positive, stable decision addition.

JSON: `experiments/archive/frontier_consolidation/data/truthful_private_scale_carriers/coherent86_alpha0p75/truthful_coherent86_alpha0p75_carrier_manifest.json`

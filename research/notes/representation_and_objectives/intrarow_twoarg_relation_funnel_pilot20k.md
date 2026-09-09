# intrarow calibration synthesis — intra-row two-argument relation funnel

This pool searches the legal active-token tail for one relation pivot with two active content arguments on opposite sides, so a later four-cell scorer can compare original left/right target assignment to swapped assignment inside the same relation context.
Segment: 20000 rows / 3115669 words starting tail row 255.
Pairs: **10,390**; rows with pairs: 7,613/20,000.

## Pairs by relation family
| family | pairs |
|---|---:|
| temporal | 3589 |
| spatial | 2991 |
| causal_connector | 2458 |
| comparative | 1352 |

Target freq delta: `{'n': 10390, 'mean': 0.8038498556304139, 'std': 0.7859599309556645, 'median': 1.0, 'p05': 0.0, 'p25': 0.0, 'p75': 1.0, 'p95': 2.0, 'min': 0.0, 'max': 2.0}`
Distance-bin delta: `{'n': 10390, 'mean': 1.1085659287776708, 'std': 0.9172466662835865, 'median': 1.0, 'p05': 0.0, 'p25': 0.0, 'p75': 2.0, 'p95': 3.0, 'min': 0.0, 'max': 4.0}`
Between-content count: `{'n': 10390, 'mean': 2.4576515880654477, 'std': 1.1838204534882362, 'median': 2.0, 'p05': 1.0, 'p25': 2.0, 'p75': 3.0, 'p95': 5.0, 'min': 1.0, 'max': 10.0}`

Top pivots:
- if: 1332
- when: 1313
- because: 767
- into: 724
- after: 563
- before: 508
- than: 488
- through: 484
- while: 393
- better: 262
- under: 249
- between: 248
- though: 246
- until: 245
- more: 216
- once: 213
- against: 209
- during: 186

Files:
- summary: `experiments/archive/representation_and_objectives/data/intrarow_twoarg_relation_funnel_pilot20k/intrarow_twoarg_funnel_summary.json`
- pairs: `experiments/archive/representation_and_objectives/data/intrarow_twoarg_relation_funnel_pilot20k/intrarow_twoarg_pair_pool.jsonl`
- csv: `experiments/archive/representation_and_objectives/data/intrarow_twoarg_relation_funnel_pilot20k/intrarow_twoarg_pair_pool_summary.csv`

# chck82 frozen private tail design sparse20 panel analysis

Status: **COMPLETE**
Route read: `supports_82M_frozen_slow_private_tail_test`

Available: `['sep_aligned', 'sep_shuffled', 'coupled_aligned']`; pending: `[]`

## Scores
| arm | cheap7 | BLiMP | Supp | EWoK | Entity | COMPS | GP | Reading | main words | aux words |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sep_aligned | 40.292857142857144 | 61.26 | 58.6 | 50.77 | 18.8 | 50.48 | 33.665 | 8.475 | 19811701 | 183484 |
| sep_shuffled | 40.06357142857143 | 60.56 | 57.45 | 50.88 | 18.35 | 50.74 | 34.195 | 8.27 | 19811701 | 183266 |
| coupled_aligned | 38.793571428571425 | 57.02 | 55.0 | 50.15 | 17.93 | 49.72 | 34.165 | 7.57 | 19811701 | 183484 |

## Observed decisions
- `sep_aligned_beats_mlm_only`: `True`
- `sep_aligned_no_major_fragile_family_damage_vs_mlm_only`: `True`
- `separation_beats_coupled_under_same_sparse_data`: `True`
- `true_correspondence_matters_under_separation`: `True`
- `coupled_sparse_beats_mlm_only`: `False`
- `sparsity_alone_improves_broad_coupled`: `True`

## Comparisons
- `sep_aligned_minus_sep_shuffled`: `{'cheap7_delta': 0.22928571428571587, 'column_deltas': {'BLiMP': 0.6999999999999957, 'Supplement': 1.1499999999999986, 'EWoK': -0.10999999999999943, 'Entity': 0.4499999999999993, 'COMPS': -0.2600000000000051, 'GlobalPIQA': -0.5300000000000011, 'Reading': 0.20500000000000007}}`
- `sep_aligned_minus_coupled_aligned`: `{'cheap7_delta': 1.499285714285719, 'column_deltas': {'BLiMP': 4.239999999999995, 'Supplement': 3.6000000000000014, 'EWoK': 0.6200000000000045, 'Entity': 0.870000000000001, 'COMPS': 0.759999999999998, 'GlobalPIQA': -0.5, 'Reading': 0.9049999999999994}}`
- `coupled_aligned_minus_broad_aligned`: `{'cheap7_delta': 0.03071428571428214, 'column_deltas': {'BLiMP': 0.30000000000000426, 'Supplement': 3.240000000000002, 'EWoK': -0.9600000000000009, 'Entity': 0.019999999999999574, 'COMPS': -0.45000000000000284, 'GlobalPIQA': -1.9350000000000023, 'Reading': 0.0}}`
- `sep_aligned_minus_mlm_only_ref`: `{'cheap7_delta': 0.5057142857142836, 'column_deltas': {'BLiMP': -0.7899999999999991, 'Supplement': 0.1700000000000017, 'EWoK': 0.6700000000000017, 'Entity': 0.4400000000000013, 'COMPS': -0.22000000000000597, 'GlobalPIQA': 0.9949999999999974, 'Reading': 2.2749999999999995}}`

This panel factors sparsity from pathway separation. A positive separated aligned-vs-shuffled effect matters only if separated aligned also exceeds the exact mlm_only adapter scaffold and avoids the known EWoK/Reading/Supplement damage; otherwise it is local correspondence learning inside an unhelpful score tradeoff.

JSON: `experiments/archive/frontier_consolidation/data/sparse20_panel_analysis/sparse20_panel_analysis.json`

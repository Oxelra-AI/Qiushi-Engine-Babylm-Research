# endpoint synthesis scale1p75 u256 — U256 100M completed non-SuperGLUE surface

This is a CPU/filesystem read of split-column artifacts already written by the hardened evaluator. It is not a complete Overall result because SuperGLUE is incomplete or still pending.

cheap7 = **43.084286**; delta vs spatial repair route status cheap7 = **+0.078561**; delta vs scale1.75 cheap7 = **-0.458874**.
Projected Overall with spatial repair route status SuperGLUE (70.279868) = **41.318874**.
Projected Overall with scale1.75 SuperGLUE (69.334599) = **41.213844**.

| Target | required full SuperGLUE | required remaining mean given completed SG tasks |
|---|---:|---:|
| overall | 69.729938 | 67.056275 |
| scale1p75_overall | 72.546719 | 76.915008 |
| sota_41p8 | 74.610000 | 84.136492 |

| Column | U256 100M | spatial repair route status 100M | Delta vs spatial repair route status | scale1.75 100M | Delta vs scale1.75 |
|---|---:|---:|---:|---:|---:|
| BLiMP | 66.810000 | 65.870718 | +0.939282 | 68.631753 | -1.821753 |
| Supplement | 60.850000 | 61.165661 | -0.315661 | 62.895171 | -2.045171 |
| EWoK | 50.390000 | 50.393237 | -0.003237 | 49.080093 | +1.309907 |
| Entity | 28.430000 | 27.400834 | +1.029166 | 27.464549 | +0.965451 |
| COMPS | 51.650000 | 52.008345 | -0.358345 | 52.303917 | -0.653917 |
| GlobalPIQA | 36.135000 | 36.063107 | +0.071893 | 36.106796 | +0.028204 |
| Reading | 7.325000 | 8.138168 | -0.813168 | 8.319840 | -0.994840 |
| AoA | 0.000000 | 0.000000 | +0.000000 | 0.000000 | +0.000000 |

SuperGLUE partial state:
- completed tasks: ['boolq', 'mrpc', 'multirc', 'rte', 'wsc']
- observed partial mean: 70.79940330385551
- missing tasks: ['mnli', 'qqp']

Payload JSON: `experiments/archive/frontier_consolidation/data/u256_100m_partial_surface/u256_100M_completed_non_superglue_payload.json`

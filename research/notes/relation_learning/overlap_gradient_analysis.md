# compact mixture model and predictions overlap-gradient analysis

This analysis joins existing held-out compact-rewrite probe rows with source/rewrite word-overlap metrics. It tests the mixture-model prediction that the identity readout should matter most when a nonidentical target sits in a source-recognizable, high-overlap pair.

## Pair overlap distribution

Pairs with overlap metadata: 1626. Mean rewrite content-word overlap fraction: 0.674; median 0.667; Jaccard mean 0.446.

## Fixed overlap bins, nonoverlap targets

| contrast | bin | n pairs | overlap mean | gain Δ | true Δ | unrel Δ | excess true cost | frac excess>0 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| RSminusC | low_<25 | 28 | 0.219 | +0.266 | -0.426 | -0.160 | -0.266 | 0.500 |
| RSminusC | mid_25_50 | 474 | 0.449 | +0.033 | -0.611 | -0.578 | -0.033 | 0.479 |
| RSminusC | high_50_75 | 603 | 0.655 | -0.089 | -0.519 | -0.608 | +0.089 | 0.516 |
| RSminusC | very_high_75_100 | 360 | 0.892 | -0.107 | -0.443 | -0.550 | +0.107 | 0.522 |
| RSminusR | low_<25 | 28 | 0.219 | +0.391 | -0.424 | -0.034 | -0.391 | 0.393 |
| RSminusR | mid_25_50 | 474 | 0.449 | +0.619 | -0.967 | -0.349 | -0.619 | 0.371 |
| RSminusR | high_50_75 | 603 | 0.655 | +0.629 | -0.926 | -0.298 | -0.629 | 0.348 |
| RSminusR | very_high_75_100 | 360 | 0.892 | +1.106 | -1.261 | -0.155 | -1.106 | 0.306 |
| RminusC | low_<25 | 28 | 0.219 | -0.125 | -0.001 | -0.126 | +0.125 | 0.464 |
| RminusC | mid_25_50 | 474 | 0.449 | -0.585 | +0.356 | -0.229 | +0.585 | 0.643 |
| RminusC | high_50_75 | 603 | 0.655 | -0.718 | +0.407 | -0.310 | +0.718 | 0.650 |
| RminusC | very_high_75_100 | 360 | 0.892 | -1.213 | +0.818 | -0.395 | +1.213 | 0.717 |
| VminusC | low_<25 | 28 | 0.219 | +0.345 | -0.558 | -0.213 | -0.345 | 0.429 |
| VminusC | mid_25_50 | 474 | 0.449 | +0.559 | -1.034 | -0.475 | -0.559 | 0.350 |
| VminusC | high_50_75 | 603 | 0.655 | +0.680 | -1.152 | -0.472 | -0.680 | 0.335 |
| VminusC | very_high_75_100 | 360 | 0.892 | +0.829 | -1.366 | -0.537 | -0.829 | 0.353 |
| VminusR | low_<25 | 28 | 0.219 | +0.470 | -0.557 | -0.087 | -0.470 | 0.393 |
| VminusR | mid_25_50 | 474 | 0.449 | +1.145 | -1.390 | -0.245 | -1.145 | 0.243 |
| VminusR | high_50_75 | 603 | 0.655 | +1.397 | -1.559 | -0.162 | -1.397 | 0.229 |
| VminusR | very_high_75_100 | 360 | 0.892 | +2.041 | -2.184 | -0.142 | -2.041 | 0.192 |

## Correlations

| token class | contrast | n pairs | corr(overlap, gain Δ) | corr(overlap, excess cost) |
|---|---|---:|---:|---:|
| nonoverlap | RSminusC | 1465 | -0.033 | +0.033 |
| nonoverlap | RSminusR | 1465 | +0.126 | -0.126 |
| nonoverlap | RminusC | 1465 | -0.152 | +0.152 |
| nonoverlap | VminusC | 1465 | +0.074 | -0.074 |
| nonoverlap | VminusR | 1465 | +0.196 | -0.196 |
| overlap | RSminusC | 1619 | +0.060 | -0.060 |
| overlap | RSminusR | 1619 | +0.055 | -0.055 |
| overlap | RminusC | 1619 | +0.002 | -0.002 |
| overlap | VminusC | 1619 | -0.031 | +0.031 |
| overlap | VminusR | 1619 | -0.032 | +0.032 |

## Reading

If the mixture picture is right, overlap should not be treated as a simple monotone scalar good. At high enough overlap, the source is recognizable; if the target token is absent from the source, identity readout can hurt and content readout can help. RS should stay near CLEAN across bins because it has the same content exposure without local source/rewrite co-occurrence. The empirical bin table should be used to refine the natural variation-set bins: the strongest REPEAT deficit is expected where overlap is high enough for recognition but the target is nonidentical.

Data outputs: `experiments/archive/relation_learning/data/overlap_gradient`.

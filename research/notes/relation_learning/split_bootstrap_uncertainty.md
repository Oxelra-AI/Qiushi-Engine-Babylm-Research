# compact mixture model and predictions REPEAT_SPLIT pair-cluster uncertainty

This bootstrap quantifies uncertainty from the finite held-out compact-rewrite pair sample after pair-averaging over 80M/90M/100M. It does not estimate training-seed variability; the second split seed is still required for that.

## Mean estimates and pair-bootstrap intervals

| contrast | metric | mean | median pair | 95% bootstrap interval | P(|mean|≤0.25) | pair q10/q90 |
|---|---|---:|---:|---:|---:|---:|
| RminusC | mean_gain_delta | -0.7852 | -0.6067 | [-0.8785, -0.6953] | 0.000 | -2.9848/+1.2380 |
| RminusC | mean_true_delta | +0.4837 | +0.3448 | [+0.3976, +0.5731] | 0.000 | -1.4621/+2.5983 |
| RminusC | mean_unrel_delta | -0.3014 | -0.2451 | [-0.3629, -0.2396] | 0.049 | -1.8338/+1.1033 |
| RminusC | mean_excess_true_cost | +0.7852 | +0.6067 | [+0.6935, +0.8775] | 0.000 | -1.2380/+2.9848 |
| RSminusC | mean_gain_delta | -0.0470 | -0.0165 | [-0.1242, +0.0308] | 1.000 | -1.9580/+1.7680 |
| RSminusC | mean_true_delta | -0.5285 | -0.4644 | [-0.6054, -0.4517] | 0.000 | -2.3603/+1.2812 |
| RSminusC | mean_unrel_delta | -0.5756 | -0.5372 | [-0.6394, -0.5126] | 0.000 | -2.1563/+0.9232 |
| RSminusC | mean_excess_true_cost | +0.0470 | +0.0165 | [-0.0314, +0.1238] | 1.000 | -1.7680/+1.9580 |
| RSminusR | mean_gain_delta | +0.7381 | +0.5940 | [+0.6473, +0.8295] | 0.000 | -1.2700/+3.0399 |
| RSminusR | mean_true_delta | -1.0123 | -0.8679 | [-1.1032, -0.9260] | 0.000 | -3.2250/+0.9686 |
| RSminusR | mean_unrel_delta | -0.2741 | -0.2947 | [-0.3345, -0.2133] | 0.220 | -1.7127/+1.1619 |
| RSminusR | mean_excess_true_cost | -0.7381 | -0.5940 | [-0.8287, -0.6488] | 0.000 | -3.0399/+1.2700 |

The pair-sample estimate of RS−C excess true-source cost is +0.0470 versus original R−C +0.7852, an attenuation of 94.0% by absolute magnitude. RS−C gain is -0.0470 versus original R−C -0.7852, an attenuation of 94.0%. The bootstrap interval for RS−C mean gain/excess stays well inside the pre-stated ±0.25 near-zero band, while original R−C is far outside it. This strengthens the one-seed locality result at the probe-sample level but does not replace the seed43122 split replicate.

The pair distribution remains broad: RS−C has about half of pairs on each side of zero, while original R−C has a strong positive excess-cost majority. The scientific object is a distributional source-specific tendency, not every-pair determinism.

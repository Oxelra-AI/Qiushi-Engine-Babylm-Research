# adapter matched horizon plan Adapter matched-horizon 20M-prefix comparison

All adapter matched horizon plan arms use the original spatial repair route status 100M LR horizon (`lr_total_steps=2529`) and stop at the exact spatial repair route status 20M checkpoint exposure (20,008,711 words / 506 batches).

| arm | BLiMP | Supp | EWoK | Entity | COMPS | GP | Read | cheap7 | Δ vs spatial repair route status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| spatial repair route status legal 20M, original 100M horizon | 59.69 | 55.45 | 50.73 | 18.65 | 50.26 | 34.20 | 8.67 | 39.6636 |  |
| adapter 20M closure adapter128, compressed 20M horizon | 53.81 | 50.13 | 51.08 | 17.07 | 50.34 | 33.75 | 7.17 | 37.6214 | -2.0421 |
| Adapter128 live, 100M LR horizon, 20M prefix | 60.38 | 56.23 | 49.49 | 18.65 | 50.44 | 32.23 | 8.31 | 39.3893 | -0.2743 |
| Adapter128 disabled path, 100M LR horizon, 20M prefix | 59.69 | 55.45 | 50.73 | 18.65 | 50.26 | 34.20 | 8.67 | 39.6636 | +0.0000 |

## Mechanism: Adapter128 live, 100M LR horizon, 20M prefix
- Adapter output RMS mean/max: 0.05656801 / 0.06986922
- Adapter up/down norm sums: 28.230628 / 66.339144
- Stock displacement vs spatial repair route status 20M: mean cosine 0.8658773877603166, mean relative L2 0.45064404707715194, max relative L2 1.7559211874599303
- Live stock displacement vs disabled128: mean cosine 0.8658773877603166, mean relative L2 0.45064404707715194, max relative L2 1.7559211874599303

## Mechanism: Adapter128 disabled path, 100M LR horizon, 20M prefix
- Adapter output RMS mean/max: 0.00000000 / 0.00000000
- Adapter up/down norm sums: 0.000000 / 54.531210
- Stock displacement vs spatial repair route status 20M: mean cosine 1.0000000014024621, mean relative L2 0.0, max relative L2 0.0

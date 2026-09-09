# earlier analysis Muon hidden-weight geometry

48 attention/FFN matrices; exact init reconstructed from spatial repair route status seeds/config.

| model | Frobenius/init | stable-rank Δ/init | entropy-rank Δ/init | top8-energy Δ/init |
|---|---:|---:|---:|---:|
| reference_20M | 1.2179 | -109.01 | -60.31 | +0.0798 |
| muon008_20M | 1.5754 | -32.56 | -9.83 | +0.0075 |
| muon012_20M | 1.9860 | -32.17 | -7.96 | +0.0053 |

## Absolute means

| model | stable rank | entropy rank | top1 energy | top8 energy | Frobenius norm |
|---|---:|---:|---:|---:|---:|
| exact_init | 152.97 | 335.13 | 0.0070 | 0.0535 | 12.80 |
| reference_20M | 43.96 | 274.82 | 0.0332 | 0.1332 | 15.55 |
| muon008_20M | 120.41 | 325.30 | 0.0088 | 0.0609 | 19.82 |
| muon012_20M | 120.81 | 327.17 | 0.0085 | 0.0588 | 25.05 |

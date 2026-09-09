# mature muon reversal and switch repair continuous Muon hidden-matrix temporal drift

All values are means over the 48 attention/FFN matrices unless otherwise noted.

## Consecutive / interval drift

| interval | rel Frobenius/start | rel Frobenius/init | flat cosine | left angle k=8 | right angle k=8 | left angle k=32 | right angle k=32 |
|---|---:|---:|---:|---:|---:|---:|---:|
| reference_20M_to_step35_70M | 0.6866 | 0.8379 | 0.8228 | 45.20 | 45.96 | 43.24 | 43.39 |
| reference_70M_to_step35_80M | 0.0742 | 0.1083 | 0.9973 | 6.68 | 6.83 | 5.90 | 5.92 |
| reference_20M_to_step35_80M | 0.6941 | 0.8470 | 0.8196 | 45.66 | 46.39 | 43.58 | 43.68 |
| muon_20M_to_muon_70M | 0.9504 | 1.5359 | 0.6918 | 59.86 | 60.04 | 54.63 | 54.79 |
| muon_70M_to_muon_80M | 0.0899 | 0.1901 | 0.9959 | 8.84 | 8.83 | 8.09 | 8.08 |
| muon_20M_to_muon_80M | 0.9563 | 1.5454 | 0.6883 | 60.24 | 60.44 | 54.91 | 55.06 |

## Muon minus spatial repair route status at same checkpoint

| ckpt | rel difference/init | stable-rank Δ | top8-energy Δ |
|---|---:|---:|---:|
| 20M | 1.4598 | +77.33 | -0.0726 |
| 70M | 2.2169 | +65.64 | -0.0748 |
| 80M | 2.2239 | +65.19 | -0.0745 |

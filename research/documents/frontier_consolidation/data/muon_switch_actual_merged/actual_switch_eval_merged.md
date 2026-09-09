# mature muon reversal and switch repair actual Muon→AdamW switch eval merge

Rows are complete only when both heavy and light part outputs exist with all cheap columns.

| ckpt | arm | BLiMP | Supp | EWoK | Entity | COMPS | GP | Read | cheap7 | Δ spatial repair route status | Δ cont.Muon | complete |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 70M | spatial repair route status | 65.39 | 59.31 | 50.47 | 26.98 | 51.82 | 35.55 | 8.740 | 42.6086 |  |  | ref |
| 70M | continuous Muon | 65.30 | 59.97 | 51.38 | 25.60 | 51.64 | 38.11 | 6.725 | 42.6743 |  |  | ref |
| 70M | Muon20M_to_AdamW | 66.04 | 61.58 | 51.39 | 22.94 | 51.95 | 37.62 | 7.970 | 42.7843 | +0.1757 | +0.1100 | True |
| 70M | Muon40M_to_AdamW | 65.59 | 61.94 | 50.46 | 23.98 | 51.62 | 35.15 | 7.505 | 42.3207 | -0.2879 | -0.3536 | True |
| 80M | spatial repair route status | 66.11 | 60.66 | 51.01 | 27.06 | 51.93 | 35.58 | 8.290 | 42.9486 |  |  | ref |
| 80M | continuous Muon | 66.35 | 60.43 | 50.72 | 24.17 | 51.52 | 39.06 | 7.260 | 42.7879 |  |  | ref |
| 80M | Muon20M_to_AdamW | 66.38 | 61.98 | 50.08 | 23.59 | 52.23 | 37.61 | 7.770 | 42.8050 | -0.1436 | +0.0171 | True |
| 80M | Muon40M_to_AdamW | 65.98 | 61.68 | 50.22 | 25.22 | 51.66 | 34.13 | 7.570 | 42.3521 | -0.5964 | -0.4357 | True |

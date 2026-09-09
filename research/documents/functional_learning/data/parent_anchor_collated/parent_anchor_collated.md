# held fitted direction test parent-anchor continuation collation

Parent common-screen source: `experiments/archive/functional_learning/data/common_eval/collated_common_eval.json`

| arm | tag | exposure | alpha | equal7 | Δ vs parent | BLiMP | Supp | EWoK | Entity | COMPS | GPIQA | Reading |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| carrier_residual | cr_87M | 87005295 |  | 44.3029 | -0.2614 | 69.04 | 65.60 | 49.18 | 27.16 | 52.21 | 38.550 | 8.380 |
| carrier_residual | cr_90M | 90005295 |  | 44.3150 | -0.2493 | 69.07 | 66.00 | 49.36 | 27.15 | 52.16 | 38.080 | 8.385 |
| carrier_residual | cr_94M | 94005295 |  | 43.8107 | -0.7536 | 68.87 | 65.20 | 48.27 | 26.12 | 52.24 | 37.590 | 8.385 |
| carrier_residual | cr_98M | 98005295 |  | 43.8914 | -0.6729 | 68.88 | 65.20 | 48.09 | 26.22 | 52.33 | 38.080 | 8.440 |
| carrier_residual | cr_100M_final | 100000000 |  | 43.8964 | -0.6679 | 68.89 | 65.20 | 48.18 | 26.18 | 52.31 | 38.080 | 8.435 |
| parent | coherent86_alpha075 |  |  | 44.5643 | +0.0000 | 69.17 | 66.40 | 49.82 | 27.78 | 52.05 | 38.565 | 8.165 |
| standard_carrier_anchor | std_87M | 87005295 |  | 44.5543 | -0.0100 | 69.04 | 67.20 | 50.18 | 27.54 | 52.11 | 37.565 | 8.245 |
| standard_carrier_anchor | std_90M | 90005295 |  | 44.4564 | -0.1079 | 69.11 | 66.80 | 49.64 | 27.77 | 52.02 | 37.580 | 8.275 |
| standard_carrier_anchor | std_94M | 94005295 |  | 44.2636 | -0.3007 | 68.91 | 66.40 | 48.64 | 26.93 | 52.12 | 38.565 | 8.280 |
| standard_carrier_anchor | std_98M | 98005295 |  | 44.3693 | -0.1950 | 69.04 | 66.40 | 48.91 | 27.24 | 52.12 | 38.565 | 8.310 |
| standard_carrier_anchor | std_100M_final | 100000000 |  | 44.3079 | -0.2564 | 69.01 | 66.00 | 48.91 | 27.24 | 52.13 | 38.565 | 8.300 |
| standard_no_kl | no_kl_87M | 87005295 |  | 44.5086 | -0.0557 | 69.10 | 67.20 | 50.00 | 27.33 | 52.11 | 37.565 | 8.255 |
| standard_no_kl | no_kl_90M | 90005295 |  | 44.2057 | -0.3586 | 69.09 | 66.40 | 49.27 | 27.83 | 51.97 | 36.605 | 8.275 |
| standard_no_kl | no_kl_94M | 94005295 |  | 44.1900 | -0.3743 | 69.01 | 66.80 | 48.00 | 27.16 | 52.04 | 38.065 | 8.255 |
| standard_no_kl | no_kl_98M | 98005295 |  | 44.4457 | -0.1186 | 69.01 | 66.80 | 48.55 | 27.28 | 52.13 | 39.050 | 8.300 |
| standard_no_kl | no_kl_100M_final | 100000000 |  | 44.4400 | -0.1243 | 68.99 | 66.80 | 48.55 | 27.28 | 52.12 | 39.050 | 8.290 |
| standard_parent_anchor | pa_87M | 87005295 |  | 44.6121 | +0.0479 | 69.09 | 67.20 | 50.00 | 27.57 | 52.12 | 38.065 | 8.240 |
| standard_parent_anchor | pa_90M | 90005295 |  | 44.5064 | -0.0579 | 69.14 | 66.80 | 49.45 | 27.83 | 51.98 | 38.080 | 8.265 |
| standard_parent_anchor | pa_94M | 94005295 |  | 44.4007 | -0.1636 | 69.00 | 66.80 | 48.27 | 26.88 | 52.04 | 39.550 | 8.265 |
| standard_parent_anchor | pa_98M | 98005295 |  | 44.4250 | -0.1393 | 69.05 | 66.80 | 48.73 | 26.95 | 52.09 | 39.065 | 8.290 |
| standard_parent_anchor | pa_100M_final | 100000000 |  | 44.4314 | -0.1329 | 69.07 | 66.80 | 48.82 | 26.91 | 52.07 | 39.065 | 8.285 |
| standard_scale_check | std_87M_alpha050 | 87005295 | 0.5 | 44.5671 | +0.0029 | 69.19 | 67.20 | 50.09 | 27.57 | 52.14 | 37.565 | 8.215 |
| standard_scale_check | std_87M_alpha100 | 87005295 | 1.0 | 44.5543 | -0.0100 | 69.07 | 66.80 | 50.27 | 27.25 | 52.16 | 38.065 | 8.265 |

## Best equal7 by arm

- carrier_residual: `cr_90M` equal7 44.3150, Δ vs parent -0.2493.
- parent: `coherent86_alpha075` equal7 44.5643, Δ vs parent +0.0000.
- standard_carrier_anchor: `std_87M` equal7 44.5543, Δ vs parent -0.0100.
- standard_no_kl: `no_kl_87M` equal7 44.5086, Δ vs parent -0.0557.
- standard_parent_anchor: `pa_87M` equal7 44.6121, Δ vs parent +0.0479.
- standard_scale_check: `std_87M_alpha050` equal7 44.5671, Δ vs parent +0.0029.

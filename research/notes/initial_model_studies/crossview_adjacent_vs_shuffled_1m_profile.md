# crossview adjacent vs shuffled 1m profile — cross-view masking: adjacent vs shuffled

Evidence JSON: `experiments/archive/initial_model_studies/data/crossview_adjacent_vs_shuffled_1m_profile.json`

Anchors and masked side were selected once from the true pair and copied unchanged to shuffled; the shuffled arm only changes visible target text. Official examples use standard WWM; pair examples use source-side anchor masking when anchors exist.

| seed | arm | BLiMP | Supp | EWoK | Entity | COMPS | Read eye | Read SPR |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | crossview_adjacent | 56.63 | 50.40 | 47.45 | 16.32 | 50.14 | 10.45 | 3.68 |
| 42 | crossview_shuffled | 56.53 | 50.40 | 49.64 | 16.32 | 50.29 | 10.45 | 3.68 |
| 43 | crossview_adjacent | 57.75 | 47.60 | 50.45 | 18.18 | 49.57 | 10.61 | 3.52 |
| 43 | crossview_shuffled | 58.53 | 49.20 | 49.64 | 18.01 | 49.69 | 10.63 | 3.50 |

## adjacent minus shuffled

| seed | BLiMP | Supp | EWoK | Entity | COMPS | Read eye | Read SPR |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.10 | 0.00 | -2.19 | 0.00 | -0.15 | 0.00 | 0.00 |
| 43 | -0.78 | -1.60 | 0.81 | 0.17 | -0.12 | -0.02 | 0.02 |

## mean delta

| BLiMP | Supp | EWoK | Entity | COMPS | Read eye | Read SPR |
|---:|---:|---:|---:|---:|---:|---:|
| -0.34 | -0.80 | -0.69 | 0.09 | -0.14 | -0.01 | 0.01 |

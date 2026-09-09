# price tradeoff split official Entity integration

This readout uses official Entity predictions for seed43022 split arms and the existing original V/R/C prediction rows. It asks whether the behavioral pattern associated with exact local recurrence is also removed by splitting source and companion rows.

## Late accuracy by relevant queried-entity updates

| group | R−C | RS−C | RS−R | V−C | VS−C | VS−V | RS−VS |
|---|---:|---:|---:|---:|---:|---:|---:|
| ALL | +0.28 | -0.66 | -0.94 | +2.75 | +0.91 | -1.84 | -1.57 |
| rel_eq0 | +8.78 | -0.54 | -9.32 | -0.91 | -10.08 | -9.17 | +9.54 |
| rel_ge1 | -2.22 | -0.70 | +1.52 | +3.83 | +4.14 | +0.31 | -4.84 |
| rel_ge2 | -2.50 | -1.03 | +1.47 | +4.85 | +5.15 | +0.30 | -6.18 |
| rel_ge3 | -3.61 | -1.08 | +2.53 | +5.28 | +5.36 | +0.08 | -6.44 |
| rel_updates_0 | +8.78 | -0.54 | -9.32 | -0.91 | -10.08 | -9.17 | +9.54 |
| rel_updates_1 | -1.39 | +0.28 | +1.66 | +0.81 | +1.16 | +0.35 | -0.88 |
| rel_updates_2 | -0.18 | -0.92 | -0.74 | +3.95 | +4.71 | +0.76 | -5.63 |
| rel_updates_3 | -5.83 | -1.98 | +3.85 | +2.63 | +3.44 | +0.81 | -5.42 |
| rel_updates_4 | -1.59 | -1.08 | +0.51 | +7.55 | +6.38 | -1.17 | -7.46 |
| rel_updates_5 | -2.06 | +2.49 | +4.55 | +7.68 | +9.31 | +1.62 | -6.82 |

A localized behavioral reading would show RS losing most of original R's zero-update advantage and update-depth cost relative to CLEAN, while VS stays closer to CLEAN than original V on the same buckets. If RS retains original R's zero-update advantage, the behavioral retrieval benefit can arise from spaced repetition or broad fit even when the source-specific NLL residual disappears.

Data outputs: `experiments/archive/relation_learning/data/split_entity_official_integration`

# seed43222 entity clean integration seed43222 Entity with CLEAN baseline

CLEAN seed43222 was evaluated on official Entity at 80M/90M/100M after seed43222 threearm probe decomposition. This note integrates its prediction-level relevant-update profile with the existing seed43222 VIEW/REPEAT readout. Entity remains a downstream correlate; the primary installed quantities are the held-out copy and rewrite-conditioning probes.

Official evaluator scores for CLEAN were 26.41, 27.15, and 26.59 at 80M/90M/100M, mean 26.72. The relevant-update table below re-scores the official prediction files with the copy and relevant update result metadata split and normalization used for V/R, giving `ALL` mean 26.49. This small metric/readout difference should be preserved; the depth-specific interpretation uses the common copy and relevant update result split below.

| group | n | V acc | R acc | C acc | V−R | V−C | R−C |
|---|---:|---:|---:|---:|---:|---:|---:|
| ALL | 6780 | 25.90 | 25.06 | 26.49 | +0.85 | -0.59 | -1.43 |
| rel_updates_0 | 1541 | 36.88 | 39.56 | 36.45 | -2.68 | +0.43 | +3.11 |
| rel_updates_1 | 1323 | 16.58 | 15.75 | 16.75 | +0.83 | -0.18 | -1.01 |
| rel_updates_2 | 1266 | 22.41 | 19.93 | 20.91 | +2.47 | +1.50 | -0.97 |
| rel_updates_3 | 1230 | 23.90 | 21.76 | 25.09 | +2.14 | -1.19 | -3.33 |
| rel_updates_4 | 1112 | 27.61 | 25.57 | 31.21 | +2.04 | -3.60 | -5.64 |
| rel_updates_5 | 308 | 27.27 | 24.89 | 29.98 | +2.38 | -2.71 | -5.09 |
| rel_ge2 | 3916 | 24.74 | 22.50 | 25.86 | +2.24 | -1.12 | -3.36 |
| rel_ge3 | 2650 | 25.85 | 23.72 | 28.23 | +2.13 | -2.38 | -4.50 |

## Scientific reading

The seed43222 V−R direction still has the expected form: REPEAT is above VIEW at zero relevant updates, while VIEW is above REPEAT at all positive update depths. CLEAN changes the interpretation. At this seed, CLEAN is not a passive midpoint and actually exceeds both intervention arms on `ALL`, `rel_ge2`, and `rel_ge3`. The zero-update result is REPEAT-specific relative to CLEAN (`R−C=+3.11`) while VIEW is nearly CLEAN-like (`V−C=+0.43`). At higher update depths, both intervention arms can fall below CLEAN, with REPEAT much lower than VIEW at rel3–rel5.

Thus the strong two-seed V−C deep-update benchmark advantage is not a stable three-seed fact. Entity should be retained as a direction-robust downstream correlate of relation-structure specialization—REPEAT toward unchanged/exact cases, VIEW relatively better after updates—but not as the primary installed quantity. The primary evidence remains the stable held-out probes: three-seed R−C active recurrence cost on nonoverlap rewrite conditioning and three-seed R−C copy advantage.

Data: `experiments/archive/relation_learning/data/seed43222_entity_clean_integration`

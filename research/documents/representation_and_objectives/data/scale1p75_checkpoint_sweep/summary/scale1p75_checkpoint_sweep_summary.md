# endpoint sweep and dualview route Scale1.75 checkpoint sweep

Frontier Overall: 41.800
Cheap7 threshold assuming 80M SuperGLUE=69.259714 and AoA=0: **43.848612**

| endpoint | cheap7 | projected Overall at 80M SG | SG required for 41.80 | BLiMP | Supp | EWoK | Entity | COMPS | GP | Reading | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| chck_77M | 43.282143 | 41.359413 | 73.225000 | 68.220 | 62.080 | 49.580 | 26.430 | 52.130 | 36.105 | 8.430 | done |
| chck_78M | 43.702143 | 41.686079 | 70.285000 | 68.020 | 62.780 | 49.040 | 27.750 | 52.340 | 37.605 | 8.380 | done |
| chck_79M | 43.578571 | 41.589968 | 71.150000 | 68.000 | 62.080 | 49.220 | 28.150 | 52.360 | 37.120 | 8.120 | done |
| chck_80M | 43.812143 | 41.771635 | 69.515000 | 68.110 | 62.620 | 49.240 | 28.200 | 52.110 | 38.105 | 8.300 | known_step156_full_score_not_reevaluated_in_step165 |
| chck_81M | 43.649286 | 41.644968 | 70.655000 | 68.220 | 62.200 | 49.380 | 27.720 | 52.060 | 37.565 | 8.400 | done |
| chck_82M | 43.960000 | 41.886635 | 68.480000 | 68.490 | 62.940 | 50.060 | 28.310 | 52.190 | 37.580 | 8.150 | done |
| chck_83M | 43.807857 | 41.768302 | 69.545000 | 68.320 | 63.370 | 49.800 | 27.970 | 51.960 | 37.090 | 8.145 | done |

Best by cheap7: `chck_82M` at 43.960000.
Decision signal: **launch_full_verification_for_candidate**.

JSON: `experiments/archive/representation_and_objectives/data/scale1p75_checkpoint_sweep/summary/scale1p75_checkpoint_sweep_summary.json`

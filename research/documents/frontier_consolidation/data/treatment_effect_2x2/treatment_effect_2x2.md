# earlier analysis — 2x2 treatment (reinvest vs clean) x seed (43022/43122) DiD

absolute score = endpoint robustness; within-seed treatment effect = whether the learning principle replicates despite inherited seed variance.

## Known full-surface Overall (nine-column mean)
- clean_seed43022: 41.3443
- clean_seed43122: 40.6501
- reinvest_seed43022: 42.0331
- reinvest_seed43122: PENDING (full official vector)

## Inherited seed spread (clean base recipe)
- Overall: -0.6942
  - BLiMP: -0.820
  - Supplement: -1.330
  - EWoK: +0.240
  - Entity: -0.500
  - COMPS: +0.280
  - SuperGLUE: -1.563
  - GlobalPIQA: -2.000
  - Reading: -0.555
  - AoA: +0.000

## Treatment effect at seed43022 (reinvest - clean, full official)
- Overall: +0.6888
  - BLiMP: +0.032
  - Supplement: +0.436
  - EWoK: +3.347
  - Entity: +1.986
  - COMPS: +0.189
  - SuperGLUE: +0.727
  - GlobalPIQA: -0.999
  - Reading: +0.482
  - AoA: +0.000

## Preliminary DiD (7-col fast reinvest vs full clean)
- DiD mean (7-col): -0.6964
  - BLiMP: TE43022=-0.210, TE43122=-0.270, DiD=-0.060
  - Supplement: TE43022=+3.560, TE43122=+1.690, DiD=-1.870
  - EWoK: TE43022=+2.900, TE43122=-1.070, DiD=-3.970
  - Entity: TE43022=+1.990, TE43122=+1.030, DiD=-0.960
  - COMPS: TE43022=+0.190, TE43122=-0.520, DiD=-0.710
  - GlobalPIQA: TE43022=-1.000, TE43122=+0.515, DiD=+1.515
  - Reading: TE43022=+0.480, TE43122=+1.660, DiD=+1.180

## Headline reasoning
The clean base recipe already loses -0.694 Overall from seed43022 to seed43122. The reinvest treatment adds +0.689 Overall at seed43022. If the same +~0.69 treatment effect holds at seed43122, reinvest_seed43122 would land near clean_seed43122 + TE = 40.650 + 0.689 = 41.339 Overall, which would be below 41.8 purely because of inherited seed variance, NOT because the learning principle failed.

Machine-readable output: `experiments/archive/frontier_consolidation/data/treatment_effect_2x2/treatment_effect_2x2.json`

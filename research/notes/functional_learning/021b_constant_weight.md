# accumulated mechanism synthesis calibration and compatible-trajectory test

Tests whether held-symbol binding loss during full-objective continuation
is inherent (body context prediction conflicts with the binding marker)
or transitional (gradient shock from uncalibrated context positions).

## Arm means at final epoch

| arm | n | start h4 | final h4 | final hB | final hSel | final train4 | final blk4 | final ctx_ce |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| direct_full | 1 | 0.958 | 0.521 | 2.101 | 0.484 | 0.646 | 0.208 | 18.981 |
| gradual_ctx100_then_full | 1 | 0.958 | 0.500 | 1.931 | 0.463 | 0.635 | 0.198 | 18.841 |

## Per-seed trajectories

### Seed 100

**direct_full**
  be=  0 [prep          ] tr4=1.000 h4=0.958 hB=+7.646 hSel=+0.901 ctx_ce=0.000 cw=0.0
  be=  1 [all_fixed     ] tr4=1.000 h4=0.917 hB=+7.456 hSel=+0.880 ctx_ce=30.615 cw=1.0
  be=  5 [all_fixed     ] tr4=1.000 h4=0.646 hB=+5.304 hSel=+0.748 ctx_ce=24.614 cw=1.0
  be= 10 [all_fixed     ] tr4=0.646 h4=0.521 hB=+2.101 hSel=+0.484 ctx_ce=18.981 cw=1.0

**gradual_ctx100_then_full**
  be=  0 [prep          ] tr4=1.000 h4=0.958 hB=+7.646 hSel=+0.901 ctx_ce=0.000 cw=0.0
  be=  1 [all_ramp      ] tr4=1.000 h4=0.917 hB=+7.456 hSel=+0.880 ctx_ce=30.615 cw=0.505
  be=  5 [all_fixed     ] tr4=1.000 h4=0.604 hB=+5.090 hSel=+0.726 ctx_ce=24.574 cw=1.0
  be= 10 [all_fixed     ] tr4=0.635 h4=0.500 hB=+1.931 hSel=+0.463 ctx_ce=18.841 cw=1.0

## Interpretation

If any compatible trajectory (gradual ramp, slow lr, interleaved reinforcement) preserves held binding while context CE drops to normal levels (~1.5 nats), the conflict is transitional and a compatible learning path exists.  If all fail, context prediction through the body inherently conflicts with the binding computation.

Readout calibration may fail to reduce context CE enough (body objective ablation showed out_only barely reduced it from ~34 to ~28 in 25 epochs), so its failure alone does not establish inherent conflict.  The gradual ramp is the strongest test because it limits gradient magnitude at every epoch.


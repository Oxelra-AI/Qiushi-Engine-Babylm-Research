# endpoint synthesis scale1p75 u256 — endpoint uncertainty and error structure

Bootstrap intervals resample official scoring units within each column. They quantify saved-prediction uncertainty, not pretraining-seed variance.

| Column | scale-spatial repair route status Δ [2.5,50,97.5] | U256-spatial repair route status Δ [2.5,50,97.5] | majority-scale Δ [2.5,50,97.5] | U256/spatial repair route status error phi | scale/U256 error phi |
|---|---:|---:|---:|---:|---:|
| BLiMP | +2.761 [+1.739,+2.730,+3.865] | +0.947 [-0.378,+0.919,+2.312] | -0.719 [-1.640,-0.710,+0.112] | 0.5981303595177012 | 0.6042112294058197 |
| Supplement | +1.730 [-1.736,+1.730,+5.481] | -0.318 [-3.621,-0.318,+3.470] | -0.346 [-1.373,-0.346,+0.634] | 0.6275154478058167 | 0.6259907893479233 |
| EWoK | -1.313 [-4.582,-1.326,+1.693] | -0.007 [-3.988,+0.006,+3.828] | +1.543 [-0.311,+1.524,+3.765] | 0.3638491199234555 | 0.4027784895912927 |
| Entity | +0.064 [-1.184,+0.104,+1.168] | +1.032 [-1.114,+1.094,+2.867] | +0.642 [-0.249,+0.613,+1.707] | 0.6049635368766294 | 0.6009143580876124 |
| COMPS | +0.296 [-7.184,+0.296,+7.684] | -0.363 [-1.385,-0.363,+0.660] | +0.049 [-3.785,+0.049,+3.981] | 0.20193555140659553 | 0.2139108513149985 |
| GlobalPIQA | +0.044 [-4.898,+0.058,+4.500] | +0.073 [-4.854,+0.066,+5.515] | -1.015 [-4.971,-1.015,+2.956] | 0.6577449947313002 | 0.593572181243418 |

## Aggregate
- six-discrete point means: {'spatial repair route status': 48.816983789120336, 'scale1p75': 49.413713207388184, 'u256': 49.04426489458392, 'majority3': 49.4394732678969}
- six-discrete point deltas: {'scale1p75_minus_step35': 0.5967294182678552, 'u256_minus_step35': 0.22728110546358474, 'scale1p75_minus_u256': 0.36944831280427043, 'majority3_minus_scale1p75': 0.025760060508716265}
- Large oracle complementarity with high pairwise error correlation and weak hard majority is consistent with multidirectional boundary rotation, not an immediately exploitable endpoint combination.

## Error-pair details
### BLiMP
- scale1p75_vs_step35: disagreement=16.753%, error_phi=0.6220460530050598, q=0.9095204582444347, both_wrong=14674, one_correct=10031
- u256_vs_step35: disagreement=17.989%, error_phi=0.5981303595177012, q=0.8924030188340889, both_wrong=14844, one_correct=10771
- scale1p75_vs_u256: disagreement=17.363%, error_phi=0.6042112294058197, q=0.8986751937864682, both_wrong=14194, one_correct=10396
### Supplement
- scale1p75_vs_step35: disagreement=13.089%, error_phi=0.6473475931414353, q=0.9360343244538635, both_wrong=930, one_correct=683
- u256_vs_step35: disagreement=14.392%, error_phi=0.6275154478058167, q=0.928845060875159, both_wrong=952, one_correct=751
- scale1p75_vs_u256: disagreement=14.757%, error_phi=0.6259907893479233, q=0.9202641210802256, both_wrong=1019, one_correct=770
### EWoK
- scale1p75_vs_step35: disagreement=28.065%, error_phi=0.43886406334589917, q=0.7361360034004087, both_wrong=2725, one_correct=2138
- u256_vs_step35: disagreement=31.806%, error_phi=0.3638491199234555, q=0.6426736695892049, both_wrong=2560, one_correct=2423
- scale1p75_vs_u256: disagreement=29.863%, error_phi=0.4027784895912927, q=0.6931541635805735, both_wrong=2676, one_correct=2275
### Entity
- scale1p75_vs_step35: disagreement=14.425%, error_phi=0.6351756147838118, q=0.9235422211468667, both_wrong=4452, one_correct=978
- u256_vs_step35: disagreement=15.678%, error_phi=0.6049635368766294, q=0.9072846069986071, both_wrong=4399, one_correct=1063
- scale1p75_vs_u256: disagreement=15.914%, error_phi=0.6009143580876124, q=0.9044365798182012, both_wrong=4376, one_correct=1079
### COMPS
- scale1p75_vs_step35: disagreement=36.063%, error_phi=0.2766390161305973, q=0.5146844564071756, both_wrong=26637, one_correct=32827
- u256_vs_step35: disagreement=39.799%, error_phi=0.20193555140659553, q=0.3886701799473145, both_wrong=25071, one_correct=36228
- scale1p75_vs_u256: disagreement=39.191%, error_phi=0.2139108513149985, q=0.4098079486052639, both_wrong=25228, one_correct=35675
### GlobalPIQA
- scale1p75_vs_step35: disagreement=12.808%, error_phi=0.7219178082191808, q=0.9529837251356239, both_wrong=117, one_correct=26
- u256_vs_step35: disagreement=15.764%, error_phi=0.6577449947313002, q=0.9241930707728754, both_wrong=114, one_correct=32
- scale1p75_vs_u256: disagreement=18.719%, error_phi=0.593572181243418, q=0.8863886703383163, both_wrong=111, one_correct=38

JSON: `experiments/archive/frontier_consolidation/data/endpoint_uncertainty_error_structure/endpoint_uncertainty_error_structure.json`

# earlier analysis common-window family readout

Stable selected-family score movement only. The MAX semantic leg is read on the common 10M--80M window until max_repeat 90M/100M arrive.

## Seed-scale separation
- cheap6_no_GlobalPIQA: trajectory-mean seed difference 0.1133; pointwise mean absolute difference 0.4199; pointwise max absolute difference 0.8600
- cheap5_no_GlobalPIQA_Reading: trajectory-mean seed difference 0.3014; pointwise mean absolute difference 0.5350; pointwise max absolute difference 0.9060
- EWoK_plus_Entity_sum: trajectory-mean seed difference 2.2620; pointwise mean absolute difference 2.5880; pointwise max absolute difference 4.4400

## semantic_VR
### dose1
- common_10M_80M: n=8 cheap6 mean -0.0625 (4/8 positive), cheap5 mean -0.1608; EWoK+Entity mean -0.3238; top abs family EWoK mean -0.8825 fraction 0.301
- available: n=10 cheap6 mean -0.0142 (6/10 positive), cheap5 mean -0.1038; EWoK+Entity mean -0.1090; top abs family EWoK mean -0.8730 fraction 0.282
### dose1p82
- common_10M_80M: n=8 cheap6 mean 0.1414 (5/8 positive), cheap5 mean 0.1432; EWoK+Entity mean -0.0987; top abs family EWoK mean -1.1650 fraction 0.367
- available: n=10 cheap6 mean 0.2190 (7/10 positive), cheap5 mean 0.2326; EWoK+Entity mean -0.1580; top abs family EWoK mean -1.1790 fraction 0.321
### dose2p64
- common_10M_80M: n=8 cheap6 mean 0.3968 (4/8 positive), cheap5 mean 0.4895; EWoK+Entity mean 1.4563; top abs family Entity mean 2.0625 fraction 0.499
  - MAX V-R cheap6 leave-one-family-out: drop BLiMP: 0.5171, drop Supplement: 0.2554, drop EWoK: 0.5974, drop Entity: 0.0636, drop COMPS: 0.4576, drop Reading: 0.4895
- available: n=8 cheap6 mean 0.3968 (4/8 positive), cheap5 mean 0.4895; EWoK+Entity mean 1.4563; top abs family Entity mean 2.0625 fraction 0.499

## cleanfree_growth_Vd_minus_V1
### dose1p82
- common_10M_80M: n=8 cheap6 mean -0.0274 (4/8 positive), cheap5 mean 0.0165; EWoK+Entity mean 0.1350; top abs family Entity mean 0.7300 fraction 0.280
- available: n=10 cheap6 mean -0.1003 (4/10 positive), cheap5 mean -0.0768; EWoK+Entity mean 0.0140; top abs family Entity mean 0.7520 fraction 0.252
### dose2p64
- common_10M_80M: n=8 cheap6 mean 0.2554 (5/8 positive), cheap5 mean 0.3400; EWoK+Entity mean 1.2563; top abs family Entity mean 0.9863 fraction 0.330
- available: n=10 cheap6 mean 0.1438 (5/10 positive), cheap5 mean 0.2168; EWoK+Entity mean 1.1160; top abs family Entity mean 0.9010 fraction 0.344

## total_VC
### dose1
- common_10M_80M: n=8 cheap6 mean 0.6499 (6/8 positive), cheap5 mean 0.6437; EWoK+Entity mean 1.5062; top abs family Entity mean 1.4550 fraction 0.373
- available: n=8 cheap6 mean 0.6499 (6/8 positive), cheap5 mean 0.6437; EWoK+Entity mean 1.5062; top abs family Entity mean 1.4550 fraction 0.373
### dose1p82
- common_10M_80M: n=8 cheap6 mean 0.6225 (7/8 positive), cheap5 mean 0.6602; EWoK+Entity mean 1.6413; top abs family Entity mean 2.1850 fraction 0.431
- available: n=8 cheap6 mean 0.6225 (7/8 positive), cheap5 mean 0.6602; EWoK+Entity mean 1.6413; top abs family Entity mean 2.1850 fraction 0.431
### dose2p64
- common_10M_80M: n=8 cheap6 mean 0.9053 (8/8 positive), cheap5 mean 0.9837; EWoK+Entity mean 2.7625; top abs family Entity mean 2.4413 fraction 0.428
- available: n=8 cheap6 mean 0.9053 (8/8 positive), cheap5 mean 0.9837; EWoK+Entity mean 2.7625; top abs family Entity mean 2.4413 fraction 0.428

## repeat_clean_RC
### dose1
- common_10M_80M: n=8 cheap6 mean 0.7124 (8/8 positive), cheap5 mean 0.8045; EWoK+Entity mean 1.8300; top abs family BLiMP mean 1.2175 fraction 0.285
- available: n=8 cheap6 mean 0.7124 (8/8 positive), cheap5 mean 0.8045; EWoK+Entity mean 1.8300; top abs family BLiMP mean 1.2175 fraction 0.285
### dose1p82
- common_10M_80M: n=8 cheap6 mean 0.4811 (7/8 positive), cheap5 mean 0.5170; EWoK+Entity mean 1.7400; top abs family Entity mean 1.1188 fraction 0.353
- available: n=8 cheap6 mean 0.4811 (7/8 positive), cheap5 mean 0.5170; EWoK+Entity mean 1.7400; top abs family Entity mean 1.1188 fraction 0.353
### dose2p64
- common_10M_80M: n=8 cheap6 mean 0.5085 (8/8 positive), cheap5 mean 0.4943; EWoK+Entity mean 1.3063; top abs family BLiMP mean 1.2875 fraction 0.366
- available: n=8 cheap6 mean 0.5085 (8/8 positive), cheap5 mean 0.4943; EWoK+Entity mean 1.3063; top abs family BLiMP mean 1.2875 fraction 0.366

## Readout
On the common 10M--80M window, V-R cheap6/cheap5 moves from -0.0625/-0.1608 at 1x, to 0.1414/0.1432 at 1.82x, to 0.3968/0.4895 at 2.64x. The 2.64x total V-C movement is broader and larger than its V-R leg: cheap6/cheap5 0.9053/0.9837, versus 1x total 0.6499/0.6437. The repeat-clean leg is positive but not dose-growing: 1x 0.7124/0.8045, 2.64x 0.5085/0.4943. The MAX V-R leg is family-concentrated rather than uniformly broad: top absolute family is Entity with mean 2.0625 and fraction 0.499; dropping Entity reduces its cheap6 leave-one-family-out mean sharply, while total V-C remains positive under family deletions. Thus the first-basin table supports a dose-amplified structured re-expression effect as an exposure-family allocation pattern, but not yet a settled general law; the second-basin MAX pair now running is needed to distinguish reproducible packet value from basin-specific family redistribution.

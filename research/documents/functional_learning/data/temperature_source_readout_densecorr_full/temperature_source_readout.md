# temperature confidence scale and densemask interpretation temperature-adjusted source-use readout

This readout fits a single logit temperature on ordinary non-Qwen legal-tail text outside the 80-update prefix, then re-scores bounded source-use tasks. It is not official BabyLM scoring and is not fit on CDI, SuperGLUE, Entity, GlobalPIQA, Reading, or the 25-item source-reversal outcomes.

## Temperature fits
- `coherent86`: T=1.1, calib NLL T1=3.984628357910193, best=3.970562728207845, positions=260, mean rank=207.52692307692308
- `ordinary_inherited_wwm_seed62064`: T=1.1, calib NLL T1=3.978598254231306, best=3.9657362829034146, positions=260, mean rank=208.03846153846155
- `densemask_sparselabel_seed62064`: T=1.1, calib NLL T1=4.020668818443441, best=4.002054445789411, positions=260, mean rank=232.6846153846154
- `clean_pres_lambda1_eval_seed62064`: T=1.1, calib NLL T1=4.004602551574891, best=3.988107125002604, positions=260, mean rank=224.62692307692308
- `clean_pres_lambda1_eval_seed62065`: T=1.1, calib NLL T1=4.001655842134586, best=3.9855348968161985, positions=260, mean rank=224.21923076923076
- `densecorr_pres_lambda1_seed62064`: T=1.1, calib NLL T1=3.9803196284633415, best=3.969116255067862, positions=260, mean rank=212.26153846153846

## Qwen source perturbation
- `coherent86`: triplets=480, spec_adv T1=3.310242757833718, spec_adv Tfit=3.2690844706250615, rank_adv=219.90489583333334, Δspec T1=0.0, Δspec Tfit=0.0, Δrank=0.0
- `ordinary_inherited_wwm_seed62064`: triplets=480, spec_adv T1=3.3105199486362595, spec_adv Tfit=3.2691598877503485, rank_adv=218.66625000000002, Δspec T1=0.0002771908025412574, Δspec Tfit=7.541712528715642e-05, Δrank=-1.2386458333333323
- `densemask_sparselabel_seed62064`: triplets=480, spec_adv T1=3.4793942878826054, spec_adv Tfit=3.415873195558929, rank_adv=286.87625, Δspec T1=0.16915153004888758, Δspec Tfit=0.14678872493386735, Δrank=66.97135416666667
- `clean_pres_lambda1_eval_seed62064`: triplets=480, spec_adv T1=3.4288377405413324, spec_adv Tfit=3.369830741009468, rank_adv=262.6013888888889, Δspec T1=0.11859498270761427, Δspec Tfit=0.10074627038440666, Δrank=42.69649305555556
- `clean_pres_lambda1_eval_seed62065`: triplets=480, spec_adv T1=3.42820893460626, spec_adv Tfit=3.3694918645902847, rank_adv=263.04791666666665, Δspec T1=0.1179661767725419, Δspec Tfit=0.10040739396522338, Δrank=43.14302083333334
- `densecorr_pres_lambda1_seed62064`: triplets=480, spec_adv T1=3.3127691631021703, spec_adv Tfit=3.2729985152602765, rank_adv=221.48739583333332, Δspec T1=0.0025264052684522416, Δspec Tfit=0.003914044635215155, Δrank=1.5825

## Common source-reversal bank
- `coherent86`: both T1=20/25, both Tfit=20/25, both rank=20/25, swing T1=9.646572995271, swing Tfit=8.830628026298115, rank swing=381.22666666666663, Δswing T1=0.0, Δswing Tfit=0.0, Δrank=0.0
- `ordinary_inherited_wwm_seed62064`: both T1=21/25, both Tfit=21/25, both rank=21/25, swing T1=9.630605356097222, swing Tfit=8.815928841517085, rank swing=377.73333333333335, Δswing T1=-0.01596763917378018, Δswing Tfit=-0.014699184781029056, Δrank=-3.493333333333321
- `densemask_sparselabel_seed62064`: both T1=21/25, both Tfit=21/25, both rank=20/25, swing T1=10.25643265587943, swing Tfit=9.374983246106478, rank swing=499.02476190476193, Δswing T1=0.6098596606084277, Δswing Tfit=0.5443552198083627, Δrank=117.79809523809524
- `clean_pres_lambda1_eval_seed62064`: both T1=21/25, both Tfit=21/25, both rank=20/25, swing T1=10.033792744250524, swing Tfit=9.176501248336972, rank swing=456.1004761904762, Δswing T1=0.38721974897952294, Δswing Tfit=0.34587322203885923, Δrank=74.87380952380953
- `clean_pres_lambda1_eval_seed62065`: both T1=21/25, both Tfit=21/25, both rank=20/25, swing T1=10.02936760212694, swing Tfit=9.17257172192846, rank swing=454.36047619047616, Δswing T1=0.38279460685593725, Δswing Tfit=0.341943695630346, Δrank=73.13380952380952
- `densecorr_pres_lambda1_seed62064`: both T1=21/25, both Tfit=21/25, both rank=21/25, swing T1=9.721261718556995, swing Tfit=8.898616671306746, rank swing=388.4419047619047, Δswing T1=0.07468872328599278, Δswing Tfit=0.06798864500863208, Δrank=7.215238095238107

## Interpretation
- **confidence_scale_test**: If a model's larger NLL margins mostly come from global logit scale, fitted-temperature margins should shrink toward the parent while rank advantages stay near zero. Rank movement and source-reversal success surviving temperature indicate changed ordering/selection rather than only confidence scale.
- **dense_status**: Dense focus should be treated as improved contextual selection only for components that survive the non-benchmark temperature adjustment or appear as rank/order changes; enlarged already-correct margins alone are not enough.
- **official_scoring**: No official BabyLM score is modified here; this readout is an explanatory frozen-checkpoint analysis that must remain separate from leaderboard arithmetic.

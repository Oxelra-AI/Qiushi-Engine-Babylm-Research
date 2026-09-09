# temperature confidence scale and densemask interpretation temperature-adjusted source-use readout

This readout fits a single logit temperature on ordinary non-Qwen legal-tail text outside the 80-update prefix, then re-scores bounded source-use tasks. It is not official BabyLM scoring and is not fit on CDI, SuperGLUE, Entity, GlobalPIQA, Reading, or the 25-item source-reversal outcomes.

## Temperature fits
- `coherent86`: T=1.1, calib NLL T1=3.984628357910193, best=3.970562728207845, positions=260, mean rank=207.52692307692308
- `sparse_focus_seed62064`: T=1.1, calib NLL T1=3.951610244695957, best=3.945547396173844, positions=260, mean rank=206.23461538461538
- `densemask_sparselabel_seed62064`: T=1.1, calib NLL T1=4.020668818443441, best=4.002054445789411, positions=260, mean rank=232.6846153846154
- `dense_focus_seed62064`: T=1.1, calib NLL T1=4.016431900586646, best=3.999507721886039, positions=260, mean rank=233.40769230769232
- `dense_focus_seed62065`: T=1.1, calib NLL T1=4.017249300201925, best=4.000061088381335, positions=260, mean rank=234.01153846153846
- `clean_pres_lambda1_eval_full80`: T=1.1, calib NLL T1=4.004602551574891, best=3.988107125002604, positions=260, mean rank=224.62692307692308
- `clean_pres_lambda1_train_full80`: T=1.1, calib NLL T1=4.0111126049206804, best=3.99142394730678, positions=260, mean rank=226.10384615384615
- `pres_lambda1_trainmode_confounded`: T=1.1, calib NLL T1=4.009554623812437, best=3.9900269626138303, positions=260, mean rank=225.63076923076923

## Qwen source perturbation
- `coherent86`: triplets=480, spec_adv T1=3.310242757833718, spec_adv Tfit=3.2690844706250615, rank_adv=219.90489583333334, Δspec T1=0.0, Δspec Tfit=0.0, Δrank=0.0
- `sparse_focus_seed62064`: triplets=480, spec_adv T1=3.3119189678955205, spec_adv Tfit=3.2711407410975597, rank_adv=214.49170138888888, Δspec T1=0.0016762100618022205, Δspec Tfit=0.00205627047249841, Δrank=-5.413194444444444
- `densemask_sparselabel_seed62064`: triplets=480, spec_adv T1=3.4793942878826054, spec_adv Tfit=3.415873195558929, rank_adv=286.87625, Δspec T1=0.16915153004888758, Δspec Tfit=0.14678872493386735, Δrank=66.97135416666667
- `dense_focus_seed62064`: triplets=480, spec_adv T1=3.484557706253448, spec_adv Tfit=3.421034417086436, rank_adv=289.79520833333333, Δspec T1=0.17431494841972986, Δspec Tfit=0.15194994646137477, Δrank=69.8903125
- `dense_focus_seed62065`: triplets=480, spec_adv T1=3.4885810907441193, spec_adv Tfit=3.4244889614744007, rank_adv=291.7661111111111, Δspec T1=0.17833833291040113, Δspec Tfit=0.15540449084933952, Δrank=71.86121527777777
- `clean_pres_lambda1_eval_full80`: triplets=480, spec_adv T1=3.4288377405413324, spec_adv Tfit=3.369830741009468, rank_adv=262.6013888888889, Δspec T1=0.11859498270761427, Δspec Tfit=0.10074627038440666, Δrank=42.69649305555556
- `clean_pres_lambda1_train_full80`: triplets=480, spec_adv T1=3.4281048376394514, spec_adv Tfit=3.370268603515643, rank_adv=264.24086805555555, Δspec T1=0.11786207980573334, Δspec Tfit=0.10118413289058177, Δrank=44.335972222222225
- `pres_lambda1_trainmode_confounded`: triplets=480, spec_adv T1=3.4272940579669213, spec_adv Tfit=3.3696199587663354, rank_adv=264.0451736111111, Δspec T1=0.11705130013320338, Δspec Tfit=0.10053548814127376, Δrank=44.140277777777776

## Common source-reversal bank
- `coherent86`: both T1=20/25, both Tfit=20/25, both rank=20/25, swing T1=9.646572995271, swing Tfit=8.830628026298115, rank swing=381.22666666666663, Δswing T1=0.0, Δswing Tfit=0.0, Δrank=0.0
- `sparse_focus_seed62064`: both T1=21/25, both Tfit=21/25, both rank=21/25, swing T1=9.597765982193605, swing Tfit=8.785574579267275, rank swing=383.4657142857143, Δswing T1=-0.04880701307739541, Δswing Tfit=-0.0450534470308394, Δrank=2.2390476190476343
- `densemask_sparselabel_seed62064`: both T1=21/25, both Tfit=21/25, both rank=20/25, swing T1=10.25643265587943, swing Tfit=9.374983246106478, rank swing=499.02476190476193, Δswing T1=0.6098596606084277, Δswing Tfit=0.5443552198083627, Δrank=117.79809523809524
- `dense_focus_seed62064`: both T1=21/25, both Tfit=21/25, both rank=20/25, swing T1=10.24995811639797, swing Tfit=9.369174468971433, rank swing=496.43142857142857, Δswing T1=0.6033851211269695, Δswing Tfit=0.5385464426733197, Δrank=115.20476190476192
- `dense_focus_seed62065`: both T1=21/25, both Tfit=21/25, both rank=20/25, swing T1=10.242624523128782, swing Tfit=9.36248384905713, rank swing=497.6547619047619, Δswing T1=0.5960515278577803, Δswing Tfit=0.5318558227590153, Δrank=116.42809523809525
- `clean_pres_lambda1_eval_full80`: both T1=21/25, both Tfit=21/25, both rank=20/25, swing T1=10.033792744250524, swing Tfit=9.176501248336972, rank swing=456.1004761904762, Δswing T1=0.38721974897952294, Δswing Tfit=0.34587322203885923, Δrank=74.87380952380953
- `clean_pres_lambda1_train_full80`: both T1=21/25, both Tfit=21/25, both rank=20/25, swing T1=10.197173726842516, swing Tfit=9.324535076036339, rank swing=480.8338095238095, Δswing T1=0.5506007315715155, Δswing Tfit=0.4939070497382255, Δrank=99.60714285714286
- `pres_lambda1_trainmode_confounded`: both T1=21/25, both Tfit=21/25, both rank=21/25, swing T1=10.194746162550791, swing Tfit=9.322367198935575, rank swing=480.7471428571429, Δswing T1=0.5481731672797884, Δswing Tfit=0.4917391726374624, Δrank=99.52047619047622

## Interpretation
- **confidence_scale_test**: If a model's larger NLL margins mostly come from global logit scale, fitted-temperature margins should shrink toward the parent while rank advantages stay near zero. Rank movement and source-reversal success surviving temperature indicate changed ordering/selection rather than only confidence scale.
- **dense_status**: Dense focus should be treated as improved contextual selection only for components that survive the non-benchmark temperature adjustment or appear as rank/order changes; enlarged already-correct margins alone are not enough.
- **official_scoring**: No official BabyLM score is modified here; this readout is an explanatory frozen-checkpoint analysis that must remain separate from leaderboard arithmetic.

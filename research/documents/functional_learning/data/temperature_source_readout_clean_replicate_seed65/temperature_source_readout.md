# temperature confidence scale and densemask interpretation temperature-adjusted source-use readout

This readout fits a single logit temperature on ordinary non-Qwen legal-tail text outside the 80-update prefix, then re-scores bounded source-use tasks. It is not official BabyLM scoring and is not fit on CDI, SuperGLUE, Entity, GlobalPIQA, Reading, or the 25-item source-reversal outcomes.

## Temperature fits
- `coherent86`: T=1.1, calib NLL T1=3.984628384961532, best=3.970562770160345, positions=260, mean rank=207.52692307692308
- `sparse_focus_seed62064`: T=1.1, calib NLL T1=3.951609981747774, best=3.945547106862068, positions=260, mean rank=206.23461538461538
- `densemask_sparselabel_seed62064`: T=1.1, calib NLL T1=4.020669095103557, best=4.002054626817027, positions=260, mean rank=232.6846153846154
- `clean_pres_lambda1_eval_seed62064`: T=1.1, calib NLL T1=4.004602749932271, best=3.9881073019825495, positions=260, mean rank=224.62692307692308
- `clean_pres_lambda1_eval_seed62065`: T=1.1, calib NLL T1=4.001655974926857, best=3.9855350261124283, positions=260, mean rank=224.21923076923076

## Qwen source perturbation
- `coherent86`: triplets=480, spec_adv T1=3.3102433471754193, spec_adv Tfit=3.2690849620733773, rank_adv=219.90489583333334, Δspec T1=0.0, Δspec Tfit=0.0, Δrank=0.0
- `sparse_focus_seed62064`: triplets=480, spec_adv T1=3.311918900669035, spec_adv Tfit=3.2711406338555067, rank_adv=214.49170138888888, Δspec T1=0.0016755534936156529, Δspec Tfit=0.0020556717821293463, Δrank=-5.413194444444444
- `densemask_sparselabel_seed62064`: triplets=480, spec_adv T1=3.479394729079472, spec_adv Tfit=3.4158735619807787, rank_adv=286.87625, Δspec T1=0.16915138190405235, Δspec Tfit=0.14678859990740117, Δrank=66.97135416666667
- `clean_pres_lambda1_eval_seed62064`: triplets=480, spec_adv T1=3.428837897194964, spec_adv Tfit=3.3698308697826644, rank_adv=262.6013888888889, Δspec T1=0.11859455001954403, Δspec Tfit=0.10074590770928708, Δrank=42.69649305555556
- `clean_pres_lambda1_eval_seed62065`: triplets=480, spec_adv T1=3.4282089571385748, spec_adv Tfit=3.3694919182292913, rank_adv=263.0463541666667, Δspec T1=0.11796560996315547, Δspec Tfit=0.10040695615591377, Δrank=43.14145833333334

## Common source-reversal bank
- `coherent86`: both T1=20/25, both Tfit=20/25, both rank=20/25, swing T1=9.646570262341273, swing Tfit=8.830625585260846, rank swing=381.22666666666663, Δswing T1=0.0, Δswing Tfit=0.0, Δrank=0.0
- `sparse_focus_seed62064`: both T1=21/25, both Tfit=21/25, both rank=21/25, swing T1=9.597765118905476, swing Tfit=8.785573833911192, rank swing=383.4657142857143, Δswing T1=-0.048805143435796276, Δswing Tfit=-0.04505175134965353, Δrank=2.2390476190476343
- `densemask_sparselabel_seed62064`: both T1=21/25, both Tfit=21/25, both rank=20/25, swing T1=10.25643264520736, swing Tfit=9.374983215899695, rank swing=499.02476190476193, Δswing T1=0.6098623828660873, Δswing Tfit=0.544357630638849, Δrank=117.79809523809524
- `clean_pres_lambda1_eval_seed62064`: both T1=21/25, both Tfit=21/25, both rank=20/25, swing T1=10.033793043891588, swing Tfit=9.176501264430229, rank swing=456.1004761904762, Δswing T1=0.3872227815503166, Δswing Tfit=0.3458756791693823, Δrank=74.87380952380953
- `clean_pres_lambda1_eval_seed62065`: both T1=21/25, both Tfit=21/25, both rank=20/25, swing T1=10.029369744999068, swing Tfit=9.172573452762196, rank swing=454.36047619047616, Δswing T1=0.3827994826577958, Δswing Tfit=0.34194786750134953, Δrank=73.13380952380952

## Interpretation
- **confidence_scale_test**: If a model's larger NLL margins mostly come from global logit scale, fitted-temperature margins should shrink toward the parent while rank advantages stay near zero. Rank movement and source-reversal success surviving temperature indicate changed ordering/selection rather than only confidence scale.
- **dense_status**: Dense focus should be treated as improved contextual selection only for components that survive the non-benchmark temperature adjustment or appear as rank/order changes; enlarged already-correct margins alone are not enough.
- **official_scoring**: No official BabyLM score is modified here; this readout is an explanatory frozen-checkpoint analysis that must remain separate from leaderboard arithmetic.

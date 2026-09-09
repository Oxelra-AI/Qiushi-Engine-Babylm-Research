# temperature confidence scale and densemask interpretation temperature-adjusted source-use readout

This readout fits a single logit temperature on ordinary non-Qwen legal-tail text outside the 80-update prefix, then re-scores bounded source-use tasks. It is not official BabyLM scoring and is not fit on CDI, SuperGLUE, Entity, GlobalPIQA, Reading, or the 25-item source-reversal outcomes.

## Temperature fits
- `coherent86`: T=1.1, calib NLL T1=4.299087088499496, best=4.2786683144131255, positions=67, mean rank=226.98507462686567
- `ordinary_inherited_wwm_seed62064`: T=1.1, calib NLL T1=4.293446178311732, best=4.2742536100767445, positions=67, mean rank=227.11940298507463
- `densemask_sparselabel_seed62064`: T=1.1, calib NLL T1=4.291286135787394, best=4.273674033248603, positions=67, mean rank=227.22388059701493
- `clean_pres_lambda1_eval_seed62064`: T=1.1, calib NLL T1=4.2876479785833785, best=4.269824507048548, positions=67, mean rank=225.02985074626866

## Qwen source perturbation
- `coherent86`: triplets=96, spec_adv T1=2.761148665410777, spec_adv Tfit=2.7572585319220604, rank_adv=201.4253472222222, Δspec T1=0.0, Δspec Tfit=0.0, Δrank=0.0
- `ordinary_inherited_wwm_seed62064`: triplets=96, spec_adv T1=2.7611750814442835, spec_adv Tfit=2.7568885436695485, rank_adv=201.00868055555554, Δspec T1=2.6416033506393433e-05, Δspec Tfit=-0.0003699882525122859, Δrank=-0.416666666666665
- `densemask_sparselabel_seed62064`: triplets=96, spec_adv T1=2.9635973821083703, spec_adv Tfit=2.9362786469605955, rank_adv=227.27170138888889, Δspec T1=0.20244871669759354, Δspec Tfit=0.17902011503853524, Δrank=25.846354166666668
- `clean_pres_lambda1_eval_seed62064`: triplets=96, spec_adv T1=2.900541511209061, spec_adv Tfit=2.8802466978247847, rank_adv=216.33854166666666, Δspec T1=0.1393928457982838, Δspec Tfit=0.12298816590272406, Δrank=14.913194444444445

## Common source-reversal bank
- `coherent86`: both T1=20/25, both Tfit=20/25, both rank=20/25, swing T1=9.646570255188715, swing Tfit=8.83062559678441, rank swing=381.22666666666663, Δswing T1=0.0, Δswing Tfit=0.0, Δrank=0.0
- `ordinary_inherited_wwm_seed62064`: both T1=21/25, both Tfit=21/25, both rank=21/25, swing T1=9.630606337728954, swing Tfit=8.815929432539713, rank swing=377.73333333333335, Δswing T1=-0.015963917459760488, Δswing Tfit=-0.014696164244697113, Δrank=-3.493333333333321
- `densemask_sparselabel_seed62064`: both T1=21/25, both Tfit=21/25, both rank=20/25, swing T1=10.256432529971715, swing Tfit=9.37498306569599, rank swing=499.02476190476193, Δswing T1=0.6098622747829983, Δswing Tfit=0.5443574689115799, Δrank=117.79809523809524
- `clean_pres_lambda1_eval_seed62064`: both T1=21/25, both Tfit=21/25, both rank=20/25, swing T1=10.033792803486188, swing Tfit=9.176501037733896, rank swing=456.1004761904762, Δswing T1=0.38722254829747327, Δswing Tfit=0.3458754409494855, Δrank=74.87380952380953

## Interpretation
- **confidence_scale_test**: If a model's larger NLL margins mostly come from global logit scale, fitted-temperature margins should shrink toward the parent while rank advantages stay near zero. Rank movement and source-reversal success surviving temperature indicate changed ordering/selection rather than only confidence scale.
- **dense_status**: Dense focus should be treated as improved contextual selection only for components that survive the non-benchmark temperature adjustment or appear as rank/order changes; enlarged already-correct margins alone are not enough.
- **official_scoring**: No official BabyLM score is modified here; this readout is an explanatory frozen-checkpoint analysis that must remain separate from leaderboard arithmetic.

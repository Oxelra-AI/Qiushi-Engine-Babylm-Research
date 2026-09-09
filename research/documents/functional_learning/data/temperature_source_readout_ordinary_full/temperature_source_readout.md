# temperature confidence scale and densemask interpretation temperature-adjusted source-use readout

This readout fits a single logit temperature on ordinary non-Qwen legal-tail text outside the 80-update prefix, then re-scores bounded source-use tasks. It is not official BabyLM scoring and is not fit on CDI, SuperGLUE, Entity, GlobalPIQA, Reading, or the 25-item source-reversal outcomes.

## Temperature fits
- `coherent86`: T=1.1, calib NLL T1=3.984628384961532, best=3.970562770160345, positions=260, mean rank=207.52692307692308
- `ordinary_inherited_wwm_seed62064`: T=1.1, calib NLL T1=3.978598322547399, best=3.9657363747174923, positions=260, mean rank=208.03846153846155
- `densemask_sparselabel_seed62064`: T=1.1, calib NLL T1=4.020669095103557, best=4.002054626817027, positions=260, mean rank=232.6846153846154
- `clean_pres_lambda1_eval_seed62064`: T=1.1, calib NLL T1=4.004602749932271, best=3.9881073019825495, positions=260, mean rank=224.62692307692308
- `clean_pres_lambda1_eval_seed62065`: T=1.1, calib NLL T1=4.001655974926857, best=3.9855350261124283, positions=260, mean rank=224.21923076923076

## Qwen source perturbation
- `coherent86`: triplets=480, spec_adv T1=3.3102432442332304, spec_adv Tfit=3.269084884204462, rank_adv=219.90489583333334, Δspec T1=0.0, Δspec Tfit=0.0, Δrank=0.0
- `ordinary_inherited_wwm_seed62064`: triplets=480, spec_adv T1=3.3105201643995112, spec_adv Tfit=3.2691600195738846, rank_adv=218.66625000000002, Δspec T1=0.00027692016628054115, Δspec Tfit=7.513536942294757e-05, Δrank=-1.2386458333333323
- `densemask_sparselabel_seed62064`: triplets=480, spec_adv T1=3.4793946837551064, spec_adv Tfit=3.4158735356916115, rank_adv=286.87625, Δspec T1=0.16915143952187564, Δspec Tfit=0.14678865148714978, Δrank=66.97135416666667
- `clean_pres_lambda1_eval_seed62064`: triplets=480, spec_adv T1=3.4288378777820614, spec_adv Tfit=3.3698308517926163, rank_adv=262.6013888888889, Δspec T1=0.11859463354883094, Δspec Tfit=0.1007459675881546, Δrank=42.69649305555556
- `clean_pres_lambda1_eval_seed62065`: triplets=480, spec_adv T1=3.4282089385949073, spec_adv Tfit=3.3694919154663676, rank_adv=263.0463541666667, Δspec T1=0.11796569436167678, Δspec Tfit=0.10040703126190541, Δrank=43.14145833333334

## Common source-reversal bank
- `coherent86`: both T1=20/25, both Tfit=20/25, both rank=20/25, swing T1=9.646570255188715, swing Tfit=8.83062559678441, rank swing=381.22666666666663, Δswing T1=0.0, Δswing Tfit=0.0, Δrank=0.0
- `ordinary_inherited_wwm_seed62064`: both T1=21/25, both Tfit=21/25, both rank=21/25, swing T1=9.630606337728954, swing Tfit=8.815929432539713, rank swing=377.73333333333335, Δswing T1=-0.015963917459760488, Δswing Tfit=-0.014696164244697113, Δrank=-3.493333333333321
- `densemask_sparselabel_seed62064`: both T1=21/25, both Tfit=21/25, both rank=20/25, swing T1=10.256432529971715, swing Tfit=9.37498306569599, rank swing=499.02476190476193, Δswing T1=0.6098622747829983, Δswing Tfit=0.5443574689115799, Δrank=117.79809523809524
- `clean_pres_lambda1_eval_seed62064`: both T1=21/25, both Tfit=21/25, both rank=20/25, swing T1=10.033792803486188, swing Tfit=9.176501037733896, rank swing=456.1004761904762, Δswing T1=0.38722254829747327, Δswing Tfit=0.3458754409494855, Δrank=74.87380952380953
- `clean_pres_lambda1_eval_seed62065`: both T1=21/25, both Tfit=21/25, both rank=20/25, swing T1=10.029369887652852, swing Tfit=9.172573577534585, rank swing=454.36047619047616, Δswing T1=0.38279963246413645, Δswing Tfit=0.34194798075017474, Δrank=73.13380952380952

## Interpretation
- **confidence_scale_test**: If a model's larger NLL margins mostly come from global logit scale, fitted-temperature margins should shrink toward the parent while rank advantages stay near zero. Rank movement and source-reversal success surviving temperature indicate changed ordering/selection rather than only confidence scale.
- **dense_status**: Dense focus should be treated as improved contextual selection only for components that survive the non-benchmark temperature adjustment or appear as rank/order changes; enlarged already-correct margins alone are not enough.
- **official_scoring**: No official BabyLM score is modified here; this readout is an explanatory frozen-checkpoint analysis that must remain separate from leaderboard arithmetic.

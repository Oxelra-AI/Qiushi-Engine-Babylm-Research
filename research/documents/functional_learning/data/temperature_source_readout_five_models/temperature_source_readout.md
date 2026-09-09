# temperature confidence scale and densemask interpretation temperature-adjusted source-use readout

This readout fits a single logit temperature on ordinary non-Qwen legal-tail text outside the 80-update prefix, then re-scores bounded source-use tasks. It is not official BabyLM scoring and is not fit on CDI, SuperGLUE, Entity, GlobalPIQA, Reading, or the 25-item source-reversal outcomes.

## Temperature fits
- `coherent86`: T=1.1, calib NLL T1=3.984628384961532, best=3.970562770160345, positions=260, mean rank=207.52692307692308
- `sparse_focus_seed62064`: T=1.1, calib NLL T1=3.951609981747774, best=3.945547106862068, positions=260, mean rank=206.23461538461538
- `dense_focus_seed62064`: T=1.1, calib NLL T1=4.0164319088825815, best=3.9995077362212426, positions=260, mean rank=233.40769230769232
- `dense_focus_seed62065`: T=1.1, calib NLL T1=4.017249701573299, best=4.0000614335688836, positions=260, mean rank=234.01153846153846
- `densemask_sparselabel_seed62064`: T=1.1, calib NLL T1=4.020669095103557, best=4.002054626817027, positions=260, mean rank=232.6846153846154

## Qwen source perturbation
- `coherent86`: triplets=480, spec_adv T1=3.3102432442332304, spec_adv Tfit=3.269084884204462, rank_adv=219.90489583333334, Δspec T1=0.0, Δspec Tfit=0.0, Δrank=0.0
- `sparse_focus_seed62064`: triplets=480, spec_adv T1=3.3119188567106095, spec_adv Tfit=3.271140596695736, rank_adv=214.49170138888888, Δspec T1=0.0016756124773787116, Δspec Tfit=0.0020557124912738848, Δrank=-5.413194444444444
- `dense_focus_seed62064`: triplets=480, spec_adv T1=3.4845575445424766, spec_adv Tfit=3.4210342367534112, rank_adv=289.795625, Δspec T1=0.1743143003092458, Δspec Tfit=0.15194935254894923, Δrank=69.89072916666667
- `dense_focus_seed62065`: triplets=480, spec_adv T1=3.488580594631947, spec_adv Tfit=3.424488533686898, rank_adv=291.7661111111111, Δspec T1=0.17833735039871598, Δspec Tfit=0.15540364948243626, Δrank=71.86121527777777
- `densemask_sparselabel_seed62064`: triplets=480, spec_adv T1=3.4793946837551064, spec_adv Tfit=3.4158735356916115, rank_adv=286.87625, Δspec T1=0.16915143952187564, Δspec Tfit=0.14678865148714978, Δrank=66.97135416666667

## Common source-reversal bank
- `coherent86`: both T1=20/25, both Tfit=20/25, both rank=20/25, swing T1=9.646570255188715, swing Tfit=8.83062559678441, rank swing=381.22666666666663, Δswing T1=0.0, Δswing Tfit=0.0, Δrank=0.0
- `sparse_focus_seed62064`: both T1=21/25, both Tfit=21/25, both rank=21/25, swing T1=9.597764874526431, swing Tfit=8.785573646553924, rank swing=383.4657142857143, Δswing T1=-0.04880538066228234, Δswing Tfit=-0.04505195023048483, Δrank=2.2390476190476343
- `dense_focus_seed62064`: both T1=21/25, both Tfit=21/25, both rank=20/25, swing T1=10.249956066125915, swing Tfit=9.369172330172288, rank swing=496.43142857142857, Δswing T1=0.6033858109372003, Δswing Tfit=0.538546733387879, Δrank=115.20476190476192
- `dense_focus_seed62065`: both T1=21/25, both Tfit=21/25, both rank=20/25, swing T1=10.242625619343348, swing Tfit=9.362484845689366, rank swing=497.6547619047619, Δswing T1=0.5960553641546338, Δswing Tfit=0.531859248904955, Δrank=116.42809523809525
- `densemask_sparselabel_seed62064`: both T1=21/25, both Tfit=21/25, both rank=20/25, swing T1=10.256432529971715, swing Tfit=9.37498306569599, rank swing=499.02476190476193, Δswing T1=0.6098622747829983, Δswing Tfit=0.5443574689115799, Δrank=117.79809523809524

## Interpretation
- **confidence_scale_test**: If a model's larger NLL margins mostly come from global logit scale, fitted-temperature margins should shrink toward the parent while rank advantages stay near zero. Rank movement and source-reversal success surviving temperature indicate changed ordering/selection rather than only confidence scale.
- **dense_status**: Dense focus should be treated as improved contextual selection only for components that survive the non-benchmark temperature adjustment or appear as rank/order changes; enlarged already-correct margins alone are not enough.
- **official_scoring**: No official BabyLM score is modified here; this readout is an explanatory frozen-checkpoint analysis that must remain separate from leaderboard arithmetic.

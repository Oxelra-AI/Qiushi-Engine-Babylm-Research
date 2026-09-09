# 314 research route decision 20260822 H3 two-seed decision

Decision: **REJECT_H3_AS_PRIMARY_BOTTLENECK**

Combined within-block Spearman rho: +0.0417
Blocked permutation p: 0.315934
Positive transition blocks: 3/6
Conflict feature relative MSE change: +0.0335

| seed | centered rho | permutation p |
|---|---:|---:|
| wwm_seed42 | +0.0058 | 0.475852 |
| wwm_seed43 | +0.0537 | 0.343466 |

Next action: Do not build conflict-aware training. Move the main route to H1 coverage or a sharper H2 causal-use test.

Evidence JSON: `experiments/archive/initial_model_studies/data/h3/two_seed_aggregate.json`

# role coordinate anchor and state probe factorized role-anchor learner

Role-coordinate model: each predicate has one learned signed coordinate; composition predicts whether the positive-role entity matches across context and hypothesis.

| arm | runs | fit | held sign fit | seen | held hyp | held ctx | held both |
|---|---:|---:|---:|---:|---:|---:|---:|
| noheld_filler | 5 | 0 | nan | nan | nan | nan | nan |
| exposure_only | 5 | 0 | nan | nan | nan | nan | nan |
| true_anchor | 5 | 5 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| shuffled_anchor | 5 | 5 | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 |
| sparse_ctx_coverage | 5 | 5 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| sparse_hyp_coverage | 5 | 5 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

The mixed held-seen surfaces are the important surfaces; held-both can remain high under a global inversion of all held predicates.

Summary JSON: `experiments/archive/representation_and_objectives/data/factorized_role_anchor/factorized_role_anchor_summary.json`

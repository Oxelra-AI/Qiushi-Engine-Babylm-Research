# transfer gain per word gate transfer gain per word gate

Decision: **REJECT_CURRENT_TRANSFER_GAIN_PER_WORD_ESTIMATOR**

| checkpoint | aligned gain | aligned-same-bag CI95 | aligned-cross-pair CI95 | raw-loss rho |
|---|---:|---:|---:|---:|
| wwm_seed42_60M | +0.01200131 | [+0.00023234, +0.00536560] | [+0.00698467, +0.01315522] | -0.204 |
| wwm_seed43_60M | +0.01175031 | [-0.00002539, +0.00475211] | [+0.00698187, +0.01299418] | -0.015 |

Next action: Do not build a corpus selector from this estimator; identify which control failed before proposing another data-allocation mechanism.

Evidence JSON: `experiments/archive/initial_model_studies/data/transfer_gain_per_word_gate.json`

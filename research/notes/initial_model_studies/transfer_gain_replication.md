# transfer gain replication transfer gain replication

Decision: **REJECT_TRANSFER_GAIN_PER_WORD_AFTER_REPLICATION**

The 72 source/rewrite pairs are disjoint by pair id from transfer gain per word gate.

| checkpoint | aligned gain | aligned-same-bag mean (CI95) | contrast positive | raw-loss rho | pass |
|---|---:|---:|---:|---:|---:|
| wwm_seed42_60M | +0.01525805 | +0.00095261 ([-0.00126670, +0.00305726]) | 0.583 | +0.086 | False |
| wwm_seed43_60M | +0.01534931 | +0.00117183 ([-0.00096830, +0.00325478]) | 0.583 | +0.178 | False |

Next action: Close this estimator. Do not select a corpus with raw loss, self gain, or the current cross-view transfer measurement.

Evidence JSON: `experiments/archive/initial_model_studies/data/transfer_gain_replication.json`

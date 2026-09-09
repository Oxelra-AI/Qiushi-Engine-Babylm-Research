# bsm interpretation — BSM pre-scale measurement results

Evidence: `experiments/archive/initial_model_studies/data/bsm_pretest_results.json`

## Train-set both_correct_frac by arm

| arm | both_correct | same_value_pref | mean_margin |
|---|---:|---:|---:|
| base | 0.025 | 0.963 | +0.068 |
| mlm_only | 1.000 | 0.000 | +11.803 |
| bsm | 1.000 | 0.000 | +11.339 |
| random_corr | 0.765 | 0.235 | +2.274 |

## BSM held-out transfer

| split | both_correct | mean_margin |
|---|---:|---:|
| heldout_entities | 1.000 | +11.532 |
| heldout_values | 0.971 | +10.698 |
| heldout_templates | 1.000 | +11.191 |
| heldout_order_flip | 1.000 | +11.713 |

## Interpretation

{
  "bsm_learns_binding_on_train": false,
  "bsm_exceeds_random_corr": true,
  "bsm_transfers_to_heldout_entities": true,
  "bsm_transfers_to_heldout_values": true,
  "bsm_transfers_to_heldout_templates": true,
  "bsm_transfers_to_order_flips": true,
  "bsm_route_viable": true
}

# earlier analysis directional interaction readout: compact

Run root: `experiments/archive/representation_and_objectives/training/runs/directional_fork_compact_seed43022`
Eval root: `experiments/archive/representation_and_objectives/data/compact_directional_cheap7_eval_r2`

## Integrity

- forward_branch_summary_present: True
- reverse_branch_summary_present: True
- all_arm_manifests_present: True
- forward_branch_status: DIRECTIONAL_BRANCH_COMPLETE
- forward_prefix_shared: True
- forward_finals_diverge: True
- reverse_branch_status: DIRECTIONAL_BRANCH_COMPLETE
- reverse_prefix_shared: True
- reverse_finals_diverge: True

## Training loss interactions (reciprocal minus one-way; lower loss is better)

- loss: {'reciprocal_avg_fr_rf': 4.612782344540665, 'oneway_avg_ff_rr': 4.612143132732066, 'reciprocal_minus_oneway': 0.0006392118085987164}
- copied_loss: {'reciprocal_avg_fr_rf': 6.018568513430818, 'oneway_avg_ff_rr': 6.009827578316639, 'reciprocal_minus_oneway': 0.008740935114179393}
- noncopied_loss: {'reciprocal_avg_fr_rf': 5.931919901674548, 'oneway_avg_ff_rr': 5.911438777498977, 'reciprocal_minus_oneway': 0.020481124175571352}

## Eval interactions (higher score is better)

- BLiMP: reciprocal_minus_oneway=0.125; values={'ff': 59.37, 'fr': 58.79, 'rr': 58.38, 'rf': 59.21}
- COMPS: reciprocal_minus_oneway=-0.09999999999999432; values={'ff': 49.75, 'fr': 49.64, 'rr': 49.66, 'rf': 49.57}
- EWoK: reciprocal_minus_oneway=-0.42499999999999716; values={'ff': 50.23, 'fr': 50.03, 'rr': 50.43, 'rf': 49.78}
- Entity: reciprocal_minus_oneway=0.0799999999999983; values={'ff': 17.2, 'fr': 17.03, 'rr': 16.91, 'rf': 17.24}
- GlobalPIQA: reciprocal_minus_oneway=-0.23499999999999943; values={'ff': 33.165, 'fr': 31.695, 'rr': 33.665, 'rf': 34.665}
- GlobalPIQA_nonparallel: reciprocal_minus_oneway=0.5; values={'ff': 44.0, 'fr': 43.0, 'rr': 45.0, 'rf': 47.0}
- GlobalPIQA_parallel: reciprocal_minus_oneway=-0.9699999999999989; values={'ff': 22.33, 'fr': 20.39, 'rr': 22.33, 'rf': 22.33}
- Reading: reciprocal_minus_oneway=-0.06000000000000005; values={'ff': 1.5050000000000001, 'fr': 1.1099999999999999, 'rr': 0.915, 'rf': 1.19}
- Supplement: reciprocal_minus_oneway=0.07499999999999574; values={'ff': 52.27, 'fr': 52.41, 'rr': 52.82, 'rf': 52.83}
- cheap7: reciprocal_minus_oneway=-0.07714285714286007; values={'ff': 37.64142857142857, 'fr': 37.24357142857143, 'rr': 37.54, 'rf': 37.78357142857143}

JSON: `experiments/archive/representation_and_objectives/data/compact_directional_cheap7_eval_r2/directional_interaction_readout.json`

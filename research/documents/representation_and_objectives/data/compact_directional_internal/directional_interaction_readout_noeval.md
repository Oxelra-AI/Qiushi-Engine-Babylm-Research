# earlier analysis directional interaction readout: compact

Run root: `experiments/archive/representation_and_objectives/training/runs/directional_fork_compact_seed43022`
Eval root: pending

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

JSON: `experiments/archive/representation_and_objectives/data/compact_directional_internal/directional_interaction_readout_noeval.json`

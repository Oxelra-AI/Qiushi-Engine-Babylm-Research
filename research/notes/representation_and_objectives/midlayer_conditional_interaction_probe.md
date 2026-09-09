# midlayer base80 midlayer conditional-interaction probe

Post-hoc layerwise decoding of existing checkpoints only; no training and no official-example tuning. The scientific question is whether correct context-conditioned ordering appears in intermediate residual states and is erased by late computation, or is absent throughout the network.

Combined summary: `experiments/archive/representation_and_objectives/data/midlayer_conditional_interaction_probe/midlayer_conditional_interaction_summary.json`

## `matched_base_80M` — matched legal16k compact-view base, 80M

- EWoK selected rows: 4; final t1_acc=0.25, final both_official=0.0, final stable_failure=0.5.
- Best EWoK both_official layer 4 (0.5); best interaction-positive layer 5 (1.0).
- EWoK recoverability: final_wrong_any_mid_t1=1.0, final_stable_any_mid_both=0.5, final_stable_any_mid_interaction=1.0.

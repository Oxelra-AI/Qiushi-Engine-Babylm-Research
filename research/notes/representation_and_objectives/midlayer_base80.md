# midlayer base80 midlayer conditional-interaction probe

Post-hoc layerwise decoding of existing checkpoints only; no training and no official-example tuning. The scientific question is whether correct context-conditioned ordering appears in intermediate residual states and is erased by late computation, or is absent throughout the network.

Combined summary: `experiments/archive/representation_and_objectives/data/midlayer_base80/midlayer_conditional_interaction_summary.json`

## `matched_base_80M` — matched legal16k compact-view base, 80M

- EWoK selected rows: 929; final t1_acc=0.43487621097954793, final both_official=0.14316469321851452, final stable_failure=0.43918191603875134.
- Best EWoK both_official layer 4 (0.1722282023681378); best interaction-positive layer 4 (0.5252960172228203).
- EWoK recoverability: final_wrong_any_mid_t1=0.8723809523809524, final_stable_any_mid_both=0.39705882352941174, final_stable_any_mid_interaction=0.9093137254901961.
- GlobalPIQA final all_acc=28.155339805825243, final hard52_acc=3.8461538461538463, final hard52_mean_margin=1.6084085599901767.
- Best GlobalPIQA all layer 8 (28.155339805825243); best hard52 layer 0 (13.461538461538462).
- GlobalPIQA hard52 recoverability: final_wrong_any_mid_correct=0.22, final_wrong_any_mid_rank_le_2=0.58.

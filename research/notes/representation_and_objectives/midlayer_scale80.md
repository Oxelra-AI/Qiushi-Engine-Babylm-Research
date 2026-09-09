# midlayer base80 midlayer conditional-interaction probe

Post-hoc layerwise decoding of existing checkpoints only; no training and no official-example tuning. The scientific question is whether correct context-conditioned ordering appears in intermediate residual states and is erased by late computation, or is absent throughout the network.

Combined summary: `experiments/archive/representation_and_objectives/data/midlayer_scale80/midlayer_conditional_interaction_summary.json`

## `scale1p75_live_80M` — scale1.75 adapter live, 80M

- EWoK selected rows: 929; final t1_acc=0.4294940796555436, final both_official=0.16254036598493002, final stable_failure=0.43487621097954793.
- Best EWoK both_official layer 7 (0.1668460710441335); best interaction-positive layer 6 (0.5080731969860065).
- EWoK recoverability: final_wrong_any_mid_t1=0.8754716981132076, final_stable_any_mid_both=0.4034653465346535, final_stable_any_mid_interaction=0.8861386138613861.
- GlobalPIQA final all_acc=26.21359223300971, final hard52_acc=1.9230769230769231, final hard52_mean_margin=1.7167748575606405.
- Best GlobalPIQA all layer 4 (30.097087378640776); best hard52 layer 1 (15.384615384615385).
- GlobalPIQA hard52 recoverability: final_wrong_any_mid_correct=0.2549019607843137, final_wrong_any_mid_rank_le_2=0.6666666666666666.

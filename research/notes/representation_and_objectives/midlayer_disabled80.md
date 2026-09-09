# midlayer base80 midlayer conditional-interaction probe

Post-hoc layerwise decoding of existing checkpoints only; no training and no official-example tuning. The scientific question is whether correct context-conditioned ordering appears in intermediate residual states and is erased by late computation, or is absent throughout the network.

Combined summary: `experiments/archive/representation_and_objectives/data/midlayer_disabled80/midlayer_conditional_interaction_summary.json`

## `scale1p75_disabled_80M` — scale1.75 trained trajectory with adapters disabled at inference, 80M

- EWoK selected rows: 929; final t1_acc=0.4402583423035522, final both_official=0.15608180839612487, final stable_failure=0.40365984930032295.
- Best EWoK both_official layer 7 (0.16361679224973089); best interaction-positive layer 6 (0.5177610333692142).
- EWoK recoverability: final_wrong_any_mid_t1=0.8711538461538462, final_stable_any_mid_both=0.4026666666666667, final_stable_any_mid_interaction=0.912.
- GlobalPIQA final all_acc=24.271844660194176, final hard52_acc=1.9230769230769231, final hard52_mean_margin=1.6940698328261978.
- Best GlobalPIQA all layer 4 (30.097087378640776); best hard52 layer 1 (15.384615384615385).
- GlobalPIQA hard52 recoverability: final_wrong_any_mid_correct=0.2549019607843137, final_wrong_any_mid_rank_le_2=0.6274509803921569.

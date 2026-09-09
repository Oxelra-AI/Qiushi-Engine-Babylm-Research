# coherence margin signal isolation coherence-margin training-trace summary

Status: **COMPLETE**

Run: `experiments/archive/frontier_consolidation/training/runs/cohmargin4M_scale1p75_seed43022`
Records: 51; optimizer steps: 51; charged/coherent words: 3999724 / 1999862
Loss first→last: [9.892760753631592, 6.55319619178772]
MLM loss first→last: [9.811876773834229, 6.473403215408325]
Margin loss first→last: [0.8088392615318298, 0.7979284226894379]; mean 0.7998403205591089; last10 0.7989563912153244
NLL bad-minus-coh first→last: [-0.009171899408102036, 0.0033190837129950523]; mean -0.0006274628855559664; last10 -0.0003145403374219313; min/max -0.009372259490191936 / 0.0036313291639089584

If the margin objective is creating coherent-context preference during the pilot, nll_bad_minus_coh should become positive and margin_loss should fall below softplus(margin). Here values near softplus(0.20)=0.798 indicate little separation.

JSON: `experiments/archive/frontier_consolidation/data/cohmargin_training_trace_summary/cohmargin_training_trace_summary.json`

# changed state bias threat and route decision Entity margin numops readout

This file-only readout groups existing sampled Entity continuous margin deltas by numops. Positive values mean MAX compact view has a larger correct-vs-distractor margin than repeat on the sampled item.

| dose | checkpoint | all mean | zero-op mean | nonzero mean | nonzero minus zero | all n | zero n | nonzero n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dose1 | chck_10M | -0.0376 | -0.2616 | +0.0072 | +0.2688 | 90 | 15 | 75 |
| dose1 | chck_20M | -0.3227 | -0.0955 | -0.3682 | -0.2727 | 90 | 15 | 75 |
| dose1 | chck_30M | -1.2331 | -0.5823 | -1.3633 | -0.7810 | 90 | 15 | 75 |
| dose1 | chck_40M | +0.9403 | -1.8490 | +1.4982 | +3.3472 | 90 | 15 | 75 |
| dose1 | chck_50M | +1.3648 | +0.2881 | +1.5801 | +1.2921 | 90 | 15 | 75 |
| dose1 | chck_60M | +1.0496 | +0.7180 | +1.1159 | +0.3979 | 90 | 15 | 75 |
| dose1 | chck_70M | +0.7291 | -1.5210 | +1.1792 | +2.7001 | 90 | 15 | 75 |
| dose1 | chck_80M | +0.8673 | -0.1729 | +1.0754 | +1.2483 | 90 | 15 | 75 |
| dose1p82 | chck_10M | +0.1013 | +0.0859 | +0.1043 | +0.0184 | 90 | 15 | 75 |
| dose1p82 | chck_20M | -0.1234 | -0.2582 | -0.0964 | +0.1619 | 90 | 15 | 75 |
| dose1p82 | chck_30M | +0.4341 | -0.0900 | +0.5390 | +0.6290 | 90 | 15 | 75 |
| dose1p82 | chck_40M | +0.6296 | -0.0864 | +0.7728 | +0.8592 | 90 | 15 | 75 |
| dose1p82 | chck_50M | +1.2411 | +2.4685 | +0.9956 | -1.4730 | 90 | 15 | 75 |
| dose1p82 | chck_60M | +0.1391 | -1.4972 | +0.4663 | +1.9636 | 90 | 15 | 75 |
| dose1p82 | chck_70M | -0.1834 | -2.2449 | +0.2289 | +2.4738 | 90 | 15 | 75 |
| dose1p82 | chck_80M | +0.2796 | -1.4723 | +0.6300 | +2.1022 | 90 | 15 | 75 |
| dose2p64 | chck_10M | -0.1430 | -0.6017 | -0.0513 | +0.5504 | 90 | 15 | 75 |
| dose2p64 | chck_20M | +0.3635 | +0.5254 | +0.3312 | -0.1942 | 90 | 15 | 75 |
| dose2p64 | chck_30M | +0.1919 | +0.7389 | +0.0825 | -0.6564 | 90 | 15 | 75 |
| dose2p64 | chck_40M | -0.3242 | -4.5502 | +0.5211 | +5.0713 | 90 | 15 | 75 |
| dose2p64 | chck_50M | +0.4795 | -1.8228 | +0.9400 | +2.7628 | 90 | 15 | 75 |
| dose2p64 | chck_60M | +1.3150 | -3.7288 | +2.3237 | +6.0525 | 90 | 15 | 75 |
| dose2p64 | chck_70M | +0.0221 | -3.4557 | +0.7176 | +4.1733 | 90 | 15 | 75 |
| dose2p64 | chck_80M | +0.9792 | -2.6744 | +1.7100 | +4.3844 | 90 | 15 | 75 |

## Interpretation

- **operation_split**: If nonzero_ops is much larger than zero_ops, sampled margins align with the changed-state-bias alternative; positive zero_ops would weaken the pure bias explanation.
- **sample_scope**: The current default input is the binding content trade predeclared predictions sampled 80M margin pilot, not the full Entity dataset and not the pending common-window ladder. Re-run with the delivered binding content trade predeclared predictions ladder CSV when that task finishes.
- **relation_to_official_accuracy_split**: This margin split is read beside the file-only official Entity numops accuracy split in entity_numops_bias_readout.

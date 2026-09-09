# changed state bias threat and route decision Entity margin numops readout

This file-only readout groups existing sampled Entity continuous margin deltas by numops. Positive values mean MAX compact view has a larger correct-vs-distractor margin than repeat on the sampled item.

| dose | checkpoint | all mean | zero-op mean | nonzero mean | nonzero minus zero | all n | zero n | nonzero n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dose1 | chck_80M | +0.8673 | -0.1729 | +1.0754 | +1.2483 | 90 | 15 | 75 |
| dose1p82 | chck_80M | +0.2796 | -1.4723 | +0.6300 | +2.1022 | 90 | 15 | 75 |
| dose2p64 | chck_80M | +0.9792 | -2.6744 | +1.7100 | +4.3844 | 90 | 15 | 75 |

## Interpretation

- **operation_split**: If nonzero_ops is much larger than zero_ops, sampled margins align with the changed-state-bias alternative; positive zero_ops would weaken the pure bias explanation.
- **sample_scope**: The current default input is the binding content trade predeclared predictions sampled 80M margin pilot, not the full Entity dataset and not the pending common-window ladder. Re-run with the delivered binding content trade predeclared predictions ladder CSV when that task finishes.
- **relation_to_official_accuracy_split**: This margin split is read beside the file-only official Entity numops accuracy split in entity_numops_bias_readout.

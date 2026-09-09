# secondary retention and orientation interference secondary-retention analysis

This file compares no-direct-secondary sparse-update probes. The readout is whether a secondary ranking relation learned before the sparse update stays true after update rows that never directly query that secondary relation. Because these pilots use staged updating, they measure retention/consolidation and orientation interference; a joint no-secondary variant is needed to separate this from ordinary sequential forgetting.

## secondary_retention_full3

Source: `experiments/archive/representation_and_objectives/data/secondary_retention_full3/secondary_retention_summary.json`. Base train acc 1.000; arms: exposure, Etrue_Rtrue, Eflip_Rtrue, Etrue_Rflip, Eflip_Rflip, Etrue_only, Eflip_only, Rtrue_only, Rflip_only.

### Core familiar trainTrain surface

| arm | update acc | base-after | event acc/margin | focal acc/margin | secondary acc/margin | secondary margin delta |
|---|---:|---:|---:|---:|---:|---:|
| BASE before | - | - | 1.000/22.272 | 1.000/22.685 | 1.000/22.974 | 0.000 |
| exposure | 1.000 | 0.995 | 1.000/16.863 | 0.996/19.454 | 1.000/19.528 | -3.446 |
| Etrue_Rtrue | 1.000 | 1.000 | 1.000/21.078 | 1.000/22.108 | 1.000/22.120 | -0.853 |
| Eflip_Rtrue | 0.993 | 0.592 | 0.044/-13.964 | 0.987/18.408 | 1.000/18.522 | -4.451 |
| Etrue_Rflip | 0.999 | 0.624 | 1.000/20.578 | 0.196/-9.269 | 0.248/-8.134 | -31.107 |
| Eflip_Rflip | 1.000 | 0.002 | 0.000/-18.898 | 0.008/-20.111 | 0.000/-20.407 | -43.380 |
| Etrue_only | 1.000 | 0.999 | 0.996/18.595 | 0.992/19.171 | 1.000/19.427 | -3.546 |
| Eflip_only | 1.000 | 0.218 | 0.027/-11.665 | 0.481/-1.010 | 0.408/-1.555 | -24.528 |
| Rtrue_only | 1.000 | 0.983 | 0.998/16.859 | 1.000/21.821 | 1.000/21.645 | -1.329 |
| Rflip_only | 1.000 | 0.147 | 0.319/-5.001 | 0.004/-22.093 | 0.000/-22.603 | -45.577 |

### atp_trainTrain_heldHyp: secondary margin and accuracy

| arm | event acc | focal acc | secondary acc | secondary margin | delta from before |
|---|---:|---:|---:|---:|---:|
| BASE before | 0.963 | 0.735 | 0.992 | 16.920 | 0.000 |
| exposure | 0.804 | 0.833 | 0.979 | 4.663 | -12.257 |
| Etrue_Rtrue | 0.802 | 0.910 | 0.996 | 16.082 | -0.838 |
| Eflip_Rtrue | 0.458 | 0.608 | 0.590 | 3.262 | -13.659 |
| Etrue_Rflip | 0.900 | 0.594 | 0.769 | 4.885 | -12.036 |
| Eflip_Rflip | 0.115 | 0.208 | 0.010 | -11.224 | -28.144 |
| Etrue_only | 0.846 | 0.817 | 0.975 | 9.268 | -7.652 |
| Eflip_only | 0.240 | 0.129 | 0.015 | -3.746 | -20.666 |
| Rtrue_only | 0.735 | 0.908 | 0.971 | 4.435 | -12.486 |
| Rflip_only | 0.323 | 0.408 | 0.427 | -0.602 | -17.522 |

### atp_stateOnly_train_trainHyp: secondary margin and accuracy

| arm | event acc | focal acc | secondary acc | secondary margin | delta from before |
|---|---:|---:|---:|---:|---:|
| BASE before | 0.521 | 1.000 | 1.000 | 23.316 | 0.000 |
| exposure | 0.462 | 1.000 | 1.000 | 19.693 | -3.623 |
| Etrue_Rtrue | 0.446 | 1.000 | 1.000 | 22.482 | -0.834 |
| Eflip_Rtrue | 0.504 | 0.996 | 1.000 | 19.344 | -3.972 |
| Etrue_Rflip | 0.521 | 0.213 | 0.150 | -10.967 | -34.283 |
| Eflip_Rflip | 0.492 | 0.004 | 0.000 | -21.226 | -44.542 |
| Etrue_only | 0.504 | 1.000 | 1.000 | 19.843 | -3.473 |
| Eflip_only | 0.492 | 0.496 | 0.504 | -1.569 | -24.885 |
| Rtrue_only | 0.500 | 1.000 | 0.996 | 21.872 | -1.444 |
| Rflip_only | 0.529 | 0.008 | 0.000 | -23.175 | -46.491 |

### atp_eventOnly_train_trainHyp: secondary margin and accuracy

| arm | event acc | focal acc | secondary acc | secondary margin | delta from before |
|---|---:|---:|---:|---:|---:|
| BASE before | 1.000 | 0.513 | 0.417 | -1.408 | 0.000 |
| exposure | 1.000 | 0.517 | 0.450 | -0.909 | 0.498 |
| Etrue_Rtrue | 1.000 | 0.542 | 0.433 | -1.212 | 0.195 |
| Eflip_Rtrue | 0.058 | 0.500 | 0.396 | -1.139 | 0.269 |
| Etrue_Rflip | 1.000 | 0.450 | 0.533 | -0.462 | 0.945 |
| Eflip_Rflip | 0.000 | 0.454 | 0.521 | 0.775 | 2.182 |
| Etrue_only | 1.000 | 0.517 | 0.417 | -1.656 | -0.248 |
| Eflip_only | 0.000 | 0.550 | 0.492 | -0.371 | 1.036 |
| Rtrue_only | 1.000 | 0.513 | 0.450 | -0.599 | 0.809 |
| Rflip_only | 0.325 | 0.458 | 0.608 | 2.002 | 3.410 |

### atp_trainEvent_nonState_trainHyp: secondary margin and accuracy

| arm | event acc | focal acc | secondary acc | secondary margin | delta from before |
|---|---:|---:|---:|---:|---:|
| BASE before | 1.000 | 0.479 | 0.552 | 1.252 | 0.000 |
| exposure | 1.000 | 0.485 | 0.542 | 0.983 | -0.269 |
| Etrue_Rtrue | 1.000 | 0.481 | 0.560 | 1.356 | 0.104 |
| Eflip_Rtrue | 0.012 | 0.479 | 0.579 | 1.093 | -0.159 |
| Etrue_Rflip | 1.000 | 0.519 | 0.494 | 0.157 | -1.096 |
| Eflip_Rflip | 0.000 | 0.492 | 0.423 | -0.856 | -2.108 |
| Etrue_only | 1.000 | 0.465 | 0.542 | 0.497 | -0.755 |
| Eflip_only | 0.021 | 0.523 | 0.510 | 0.343 | -0.909 |
| Rtrue_only | 0.996 | 0.467 | 0.560 | 1.572 | 0.319 |
| Rflip_only | 0.325 | 0.542 | 0.492 | -0.255 | -1.507 |

### atp_nonEvent_trainState_trainHyp: secondary margin and accuracy

| arm | event acc | focal acc | secondary acc | secondary margin | delta from before |
|---|---:|---:|---:|---:|---:|
| BASE before | 0.502 | 0.969 | 1.000 | 22.955 | 0.000 |
| exposure | 0.475 | 0.963 | 1.000 | 19.192 | -3.763 |
| Etrue_Rtrue | 0.494 | 0.981 | 1.000 | 22.005 | -0.950 |
| Eflip_Rtrue | 0.525 | 0.977 | 1.000 | 18.347 | -4.608 |
| Etrue_Rflip | 0.483 | 0.212 | 0.208 | -8.415 | -31.370 |
| Eflip_Rflip | 0.479 | 0.023 | 0.000 | -20.189 | -43.144 |
| Etrue_only | 0.498 | 0.933 | 1.000 | 19.192 | -3.763 |
| Eflip_only | 0.502 | 0.456 | 0.438 | -1.933 | -24.888 |
| Rtrue_only | 0.479 | 0.998 | 1.000 | 21.476 | -1.479 |
| Rflip_only | 0.552 | 0.033 | 0.000 | -22.386 | -45.341 |

### Mechanism contrasts on trainTrain/trainHyp

- Event flip with rank true: event margin 21.078 -> -13.964; secondary margin 22.120 -> 18.522.
- Rank flip with event true: focal margin 22.108 -> -9.269; secondary margin 22.120 -> -8.134.
- Event-only true vs flipped: secondary margin 19.427 -> -1.555; focal margin 19.171 -> -1.010.
- Rank-only true vs flipped: event margin 16.859 -> -5.001; focal margin 21.821 -> -22.093; secondary margin 21.645 -> -22.603.


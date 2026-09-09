# secondary retention and orientation interference secondary retention without direct sparse secondary supervision

Base training uses anchor-wording event/focal/secondary labels. Sparse update rows omit secondary-state directional labels and replace them with neutral mention rows so update row counts stay matched. Positive contrastive margin means the true AB/BA fact has higher belief than the swapped alternative.

## Construction

- base rows: 2240 {'event_role': 960, 'focal_state': 640, 'untouched_state': 640} labels={'0': 1120, '1': 1120}
- update exposure: rows=224 queries={'mention_exposure': 224} labels={'0': 112, '1': 112}
- update Etrue_Rtrue: rows=224 queries={'event_role': 96, 'focal_state': 64, 'neutral_mention': 64} labels={'0': 112, '1': 112}
- update Eflip_Rtrue: rows=224 queries={'event_role': 96, 'focal_state': 64, 'neutral_mention': 64} labels={'0': 112, '1': 112}
- update Etrue_Rflip: rows=224 queries={'event_role': 96, 'focal_state': 64, 'neutral_mention': 64} labels={'0': 112, '1': 112}
- update Eflip_Rflip: rows=224 queries={'event_role': 96, 'focal_state': 64, 'neutral_mention': 64} labels={'0': 112, '1': 112}
- update Etrue_only: rows=224 queries={'event_role': 96, 'neutral_mention': 128} labels={'0': 112, '1': 112}
- update Eflip_only: rows=224 queries={'event_role': 96, 'neutral_mention': 128} labels={'0': 112, '1': 112}
- update Rtrue_only: rows=224 queries={'focal_state': 64, 'mention_exposure': 32, 'neutral_mention': 128} labels={'0': 112, '1': 112}
- update Rflip_only: rows=224 queries={'focal_state': 64, 'mention_exposure': 32, 'neutral_mention': 128} labels={'0': 112, '1': 112}

## Base before sparse update (train acc mean=1.000, std=0.000, seeds=3)

| eval | event acc/margin | focal acc/margin | secondary acc/margin |
|---|---:|---:|---:|
| atp_trainTrain_trainHyp | 1.000/22.272 | 1.000/22.685 | 1.000/22.974 |
| atp_trainTrain_heldHyp | 0.963/18.590 | 0.735/10.706 | 0.992/16.920 |
| atp_heldHeld_trainHyp | 0.908/14.468 | 0.383/-2.201 | 0.435/-1.210 |
| atp_dirDir_trainHyp | 0.954/18.013 | 0.067/-14.290 | 0.025/-17.272 |
| atp_trainTrain_stateFirst_trainHyp | 1.000/21.581 | 0.990/21.941 | 1.000/22.705 |
| atp_trainTrain_secondaryFirst_trainHyp | 0.987/20.994 | 0.990/22.000 | 1.000/22.657 |
| atp_stateOnly_train_trainHyp | 0.521/0.127 | 1.000/23.185 | 1.000/23.316 |
| atp_eventOnly_train_trainHyp | 1.000/23.479 | 0.513/0.514 | 0.417/-1.408 |
| atp_trainEvent_nonState_trainHyp | 1.000/22.273 | 0.479/-0.551 | 0.552/1.252 |
| atp_nonEvent_trainState_trainHyp | 0.502/-0.067 | 0.969/20.056 | 1.000/22.955 |

## After update arm exposure (update acc mean=1.000, base-anchor-after=0.995, seeds=3)

| eval | event acc/margin | focal acc/margin | secondary acc/margin | secondary margin delta |
|---|---:|---:|---:|---:|
| atp_trainTrain_trainHyp | 1.000/16.863 | 0.996/19.454 | 1.000/19.528 | -3.446 |
| atp_trainTrain_heldHyp | 0.804/9.733 | 0.833/3.446 | 0.979/4.663 | -12.257 |
| atp_heldHeld_trainHyp | 0.869/11.797 | 0.448/-1.371 | 0.465/-0.469 | 0.741 |
| atp_dirDir_trainHyp | 0.996/14.927 | 0.108/-11.928 | 0.073/-14.306 | 2.967 |
| atp_trainTrain_stateFirst_trainHyp | 0.994/16.481 | 0.990/18.995 | 0.998/19.294 | -3.411 |
| atp_trainTrain_secondaryFirst_trainHyp | 0.992/16.180 | 0.990/19.004 | 0.996/19.210 | -3.447 |
| atp_stateOnly_train_trainHyp | 0.462/-0.174 | 1.000/19.492 | 1.000/19.693 | -3.623 |
| atp_eventOnly_train_trainHyp | 1.000/18.015 | 0.517/0.182 | 0.450/-0.909 | 0.498 |
| atp_trainEvent_nonState_trainHyp | 1.000/16.766 | 0.485/-0.715 | 0.542/0.983 | -0.269 |
| atp_nonEvent_trainState_trainHyp | 0.475/-0.419 | 0.963/16.347 | 1.000/19.192 | -3.763 |

## After update arm Etrue_Rtrue (update acc mean=1.000, base-anchor-after=1.000, seeds=3)

| eval | event acc/margin | focal acc/margin | secondary acc/margin | secondary margin delta |
|---|---:|---:|---:|---:|
| atp_trainTrain_trainHyp | 1.000/21.078 | 1.000/22.108 | 1.000/22.120 | -0.853 |
| atp_trainTrain_heldHyp | 0.802/11.720 | 0.910/12.817 | 0.996/16.082 | -0.838 |
| atp_heldHeld_trainHyp | 0.958/15.467 | 0.369/-2.954 | 0.385/-2.254 | -1.044 |
| atp_dirDir_trainHyp | 0.998/18.363 | 0.071/-11.404 | 0.044/-14.928 | 2.345 |
| atp_trainTrain_stateFirst_trainHyp | 1.000/20.322 | 0.992/21.372 | 1.000/22.066 | -0.638 |
| atp_trainTrain_secondaryFirst_trainHyp | 0.963/19.184 | 0.992/21.506 | 0.996/21.832 | -0.825 |
| atp_stateOnly_train_trainHyp | 0.446/-0.700 | 1.000/22.388 | 1.000/22.482 | -0.834 |
| atp_eventOnly_train_trainHyp | 1.000/22.805 | 0.542/0.198 | 0.433/-1.212 | 0.195 |
| atp_trainEvent_nonState_trainHyp | 1.000/21.212 | 0.481/0.383 | 0.560/1.356 | 0.104 |
| atp_nonEvent_trainState_trainHyp | 0.494/-0.014 | 0.981/19.903 | 1.000/22.005 | -0.950 |

## After update arm Eflip_Rtrue (update acc mean=0.993, base-anchor-after=0.592, seeds=3)

| eval | event acc/margin | focal acc/margin | secondary acc/margin | secondary margin delta |
|---|---:|---:|---:|---:|
| atp_trainTrain_trainHyp | 0.044/-13.964 | 0.987/18.408 | 1.000/18.522 | -4.451 |
| atp_trainTrain_heldHyp | 0.458/-0.846 | 0.608/3.379 | 0.590/3.262 | -13.659 |
| atp_heldHeld_trainHyp | 0.256/-6.908 | 0.396/-2.250 | 0.425/-0.868 | 0.342 |
| atp_dirDir_trainHyp | 0.148/-9.785 | 0.208/-6.661 | 0.135/-9.672 | 7.600 |
| atp_trainTrain_stateFirst_trainHyp | 0.054/-13.260 | 0.987/17.103 | 1.000/18.598 | -4.106 |
| atp_trainTrain_secondaryFirst_trainHyp | 0.098/-11.864 | 0.987/17.720 | 1.000/18.784 | -3.873 |
| atp_stateOnly_train_trainHyp | 0.504/-0.857 | 0.996/18.809 | 1.000/19.344 | -3.972 |
| atp_eventOnly_train_trainHyp | 0.058/-17.042 | 0.500/0.997 | 0.396/-1.139 | 0.269 |
| atp_trainEvent_nonState_trainHyp | 0.012/-13.986 | 0.479/-0.786 | 0.579/1.093 | -0.159 |
| atp_nonEvent_trainState_trainHyp | 0.525/0.148 | 0.977/16.124 | 1.000/18.347 | -4.608 |

## After update arm Etrue_Rflip (update acc mean=0.999, base-anchor-after=0.624, seeds=3)

| eval | event acc/margin | focal acc/margin | secondary acc/margin | secondary margin delta |
|---|---:|---:|---:|---:|
| atp_trainTrain_trainHyp | 1.000/20.578 | 0.196/-9.269 | 0.248/-8.134 | -31.107 |
| atp_trainTrain_heldHyp | 0.900/13.220 | 0.594/1.414 | 0.769/4.885 | -12.036 |
| atp_heldHeld_trainHyp | 0.862/12.747 | 0.633/3.156 | 0.594/2.792 | 4.002 |
| atp_dirDir_trainHyp | 0.992/17.591 | 0.673/4.180 | 0.696/5.114 | 22.387 |
| atp_trainTrain_stateFirst_trainHyp | 0.998/19.754 | 0.198/-7.540 | 0.233/-7.728 | -30.433 |
| atp_trainTrain_secondaryFirst_trainHyp | 0.985/18.907 | 0.169/-9.166 | 0.227/-7.761 | -30.418 |
| atp_stateOnly_train_trainHyp | 0.521/0.362 | 0.213/-9.203 | 0.150/-10.967 | -34.283 |
| atp_eventOnly_train_trainHyp | 1.000/23.105 | 0.450/-1.036 | 0.533/-0.462 | 0.945 |
| atp_trainEvent_nonState_trainHyp | 1.000/20.215 | 0.519/0.152 | 0.494/0.157 | -1.096 |
| atp_nonEvent_trainState_trainHyp | 0.483/0.107 | 0.212/-7.630 | 0.208/-8.415 | -31.370 |

## After update arm Eflip_Rflip (update acc mean=1.000, base-anchor-after=0.002, seeds=3)

| eval | event acc/margin | focal acc/margin | secondary acc/margin | secondary margin delta |
|---|---:|---:|---:|---:|
| atp_trainTrain_trainHyp | 0.000/-18.898 | 0.008/-20.111 | 0.000/-20.407 | -43.380 |
| atp_trainTrain_heldHyp | 0.115/-12.486 | 0.208/-9.010 | 0.010/-11.224 | -28.144 |
| atp_heldHeld_trainHyp | 0.071/-12.852 | 0.642/4.049 | 0.625/2.995 | 4.205 |
| atp_dirDir_trainHyp | 0.065/-12.045 | 0.852/11.931 | 0.921/13.199 | 30.471 |
| atp_trainTrain_stateFirst_trainHyp | 0.000/-18.535 | 0.006/-20.162 | 0.000/-20.358 | -43.063 |
| atp_trainTrain_secondaryFirst_trainHyp | 0.002/-17.833 | 0.035/-19.354 | 0.000/-20.280 | -42.937 |
| atp_stateOnly_train_trainHyp | 0.492/0.499 | 0.004/-21.085 | 0.000/-21.226 | -44.542 |
| atp_eventOnly_train_trainHyp | 0.000/-21.800 | 0.454/-1.418 | 0.521/0.775 | 2.182 |
| atp_trainEvent_nonState_trainHyp | 0.000/-18.150 | 0.492/0.976 | 0.423/-0.856 | -2.108 |
| atp_nonEvent_trainState_trainHyp | 0.479/0.298 | 0.023/-18.222 | 0.000/-20.189 | -43.144 |

## After update arm Etrue_only (update acc mean=1.000, base-anchor-after=0.999, seeds=3)

| eval | event acc/margin | focal acc/margin | secondary acc/margin | secondary margin delta |
|---|---:|---:|---:|---:|
| atp_trainTrain_trainHyp | 0.996/18.595 | 0.992/19.171 | 1.000/19.427 | -3.546 |
| atp_trainTrain_heldHyp | 0.846/11.153 | 0.817/7.188 | 0.975/9.268 | -7.652 |
| atp_heldHeld_trainHyp | 0.902/14.425 | 0.385/-0.649 | 0.429/-0.259 | 0.951 |
| atp_dirDir_trainHyp | 0.983/15.754 | 0.088/-11.965 | 0.048/-13.836 | 3.437 |
| atp_trainTrain_stateFirst_trainHyp | 1.000/18.408 | 0.983/18.579 | 0.998/19.320 | -3.385 |
| atp_trainTrain_secondaryFirst_trainHyp | 0.975/17.127 | 0.983/18.750 | 0.996/19.021 | -3.636 |
| atp_stateOnly_train_trainHyp | 0.504/0.464 | 1.000/19.798 | 1.000/19.843 | -3.473 |
| atp_eventOnly_train_trainHyp | 1.000/20.649 | 0.517/-0.163 | 0.417/-1.656 | -0.248 |
| atp_trainEvent_nonState_trainHyp | 1.000/18.837 | 0.465/-1.228 | 0.542/0.497 | -0.755 |
| atp_nonEvent_trainState_trainHyp | 0.498/-0.168 | 0.933/16.802 | 1.000/19.192 | -3.763 |

## After update arm Eflip_only (update acc mean=1.000, base-anchor-after=0.218, seeds=3)

| eval | event acc/margin | focal acc/margin | secondary acc/margin | secondary margin delta |
|---|---:|---:|---:|---:|
| atp_trainTrain_trainHyp | 0.027/-11.665 | 0.481/-1.010 | 0.408/-1.555 | -24.528 |
| atp_trainTrain_heldHyp | 0.240/-5.593 | 0.129/-2.815 | 0.015/-3.746 | -20.666 |
| atp_heldHeld_trainHyp | 0.133/-6.299 | 0.337/-3.325 | 0.440/-1.621 | -0.411 |
| atp_dirDir_trainHyp | 0.119/-7.460 | 0.556/-0.717 | 0.585/-1.213 | 16.059 |
| atp_trainTrain_stateFirst_trainHyp | 0.065/-10.674 | 0.473/-0.621 | 0.438/-1.311 | -24.015 |
| atp_trainTrain_secondaryFirst_trainHyp | 0.081/-9.425 | 0.475/-1.161 | 0.465/-1.699 | -24.356 |
| atp_stateOnly_train_trainHyp | 0.492/-0.510 | 0.496/-1.720 | 0.504/-1.569 | -24.885 |
| atp_eventOnly_train_trainHyp | 0.000/-14.253 | 0.550/1.013 | 0.492/-0.371 | 1.036 |
| atp_trainEvent_nonState_trainHyp | 0.021/-10.874 | 0.523/-0.084 | 0.510/0.343 | -0.909 |
| atp_nonEvent_trainState_trainHyp | 0.502/0.274 | 0.456/0.331 | 0.438/-1.933 | -24.888 |

## After update arm Rtrue_only (update acc mean=1.000, base-anchor-after=0.983, seeds=3)

| eval | event acc/margin | focal acc/margin | secondary acc/margin | secondary margin delta |
|---|---:|---:|---:|---:|
| atp_trainTrain_trainHyp | 0.998/16.859 | 1.000/21.821 | 1.000/21.645 | -1.329 |
| atp_trainTrain_heldHyp | 0.735/7.580 | 0.908/4.995 | 0.971/4.435 | -12.486 |
| atp_heldHeld_trainHyp | 0.869/11.175 | 0.408/-2.623 | 0.448/-1.212 | -0.002 |
| atp_dirDir_trainHyp | 0.983/13.961 | 0.104/-14.291 | 0.098/-15.188 | 2.084 |
| atp_trainTrain_stateFirst_trainHyp | 1.000/17.175 | 0.996/21.339 | 1.000/21.589 | -1.116 |
| atp_trainTrain_secondaryFirst_trainHyp | 0.971/15.515 | 0.998/21.462 | 1.000/21.703 | -0.954 |
| atp_stateOnly_train_trainHyp | 0.500/0.124 | 1.000/21.988 | 0.996/21.872 | -1.444 |
| atp_eventOnly_train_trainHyp | 1.000/18.753 | 0.513/-0.011 | 0.450/-0.599 | 0.809 |
| atp_trainEvent_nonState_trainHyp | 0.996/17.444 | 0.467/-1.149 | 0.560/1.572 | 0.319 |
| atp_nonEvent_trainState_trainHyp | 0.479/-0.473 | 0.998/20.438 | 1.000/21.476 | -1.479 |

## After update arm Rflip_only (update acc mean=1.000, base-anchor-after=0.147, seeds=3)

| eval | event acc/margin | focal acc/margin | secondary acc/margin | secondary margin delta |
|---|---:|---:|---:|---:|
| atp_trainTrain_trainHyp | 0.319/-5.001 | 0.004/-22.093 | 0.000/-22.603 | -45.577 |
| atp_trainTrain_heldHyp | 0.323/-4.509 | 0.408/-0.715 | 0.427/-0.602 | -17.522 |
| atp_heldHeld_trainHyp | 0.185/-7.551 | 0.610/4.772 | 0.590/4.663 | 5.873 |
| atp_dirDir_trainHyp | 0.281/-4.715 | 0.817/11.117 | 0.919/14.715 | 31.988 |
| atp_trainTrain_stateFirst_trainHyp | 0.298/-4.801 | 0.006/-20.863 | 0.000/-22.052 | -44.756 |
| atp_trainTrain_secondaryFirst_trainHyp | 0.275/-4.428 | 0.012/-20.975 | 0.010/-21.742 | -44.399 |
| atp_stateOnly_train_trainHyp | 0.529/-0.031 | 0.008/-22.887 | 0.000/-23.175 | -46.491 |
| atp_eventOnly_train_trainHyp | 0.325/-2.815 | 0.458/-1.575 | 0.608/2.002 | 3.410 |
| atp_trainEvent_nonState_trainHyp | 0.325/-3.658 | 0.542/1.328 | 0.492/-0.255 | -1.507 |
| atp_nonEvent_trainState_trainHyp | 0.552/0.670 | 0.033/-16.263 | 0.000/-22.386 | -45.341 |


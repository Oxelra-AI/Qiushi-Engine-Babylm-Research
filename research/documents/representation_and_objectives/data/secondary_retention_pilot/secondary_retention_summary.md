# secondary retention and orientation interference secondary retention without direct sparse secondary supervision

Base training uses anchor-wording event/focal/secondary labels. Sparse update rows omit secondary-state directional labels and replace them with neutral mention rows so update row counts stay matched. Positive contrastive margin means the true AB/BA fact has higher belief than the swapped alternative.

## Construction

- base rows: 2240 {'event_role': 960, 'focal_state': 640, 'untouched_state': 640} labels={'0': 1120, '1': 1120}
- update exposure: rows=224 queries={'mention_exposure': 224} labels={'0': 112, '1': 112}
- update Etrue_Rtrue: rows=224 queries={'event_role': 96, 'focal_state': 64, 'neutral_mention': 64} labels={'0': 112, '1': 112}
- update Eflip_Rtrue: rows=224 queries={'event_role': 96, 'focal_state': 64, 'neutral_mention': 64} labels={'0': 112, '1': 112}
- update Etrue_Rflip: rows=224 queries={'event_role': 96, 'focal_state': 64, 'neutral_mention': 64} labels={'0': 112, '1': 112}
- update Eflip_Rflip: rows=224 queries={'event_role': 96, 'focal_state': 64, 'neutral_mention': 64} labels={'0': 112, '1': 112}

## Base before sparse update (train acc mean=1.000, std=0.000, seeds=1)

| eval | event acc/margin | focal acc/margin | secondary acc/margin |
|---|---:|---:|---:|
| atp_trainTrain_trainHyp | 1.000/20.057 | 1.000/20.294 | 1.000/20.673 |
| atp_trainTrain_heldHyp | 0.975/18.839 | 0.775/10.163 | 1.000/18.968 |
| atp_heldHeld_trainHyp | 0.981/16.102 | 0.512/3.121 | 0.500/2.965 |
| atp_dirDir_trainHyp | 0.994/17.438 | 0.087/-7.760 | 0.056/-8.609 |
| atp_trainTrain_stateFirst_trainHyp | 1.000/19.913 | 0.975/18.813 | 1.000/20.540 |
| atp_trainTrain_secondaryFirst_trainHyp | 1.000/20.156 | 1.000/19.263 | 1.000/20.713 |
| atp_stateOnly_train_trainHyp | 0.537/-0.276 | 1.000/20.783 | 1.000/20.963 |
| atp_eventOnly_train_trainHyp | 1.000/20.530 | 0.525/1.584 | 0.388/-0.739 |
| atp_trainEvent_nonState_trainHyp | 1.000/19.812 | 0.506/0.620 | 0.544/1.230 |
| atp_nonEvent_trainState_trainHyp | 0.512/0.753 | 0.994/19.573 | 1.000/20.579 |

## After update arm exposure (update acc mean=0.996, base-anchor-after=0.999, seeds=1)

| eval | event acc/margin | focal acc/margin | secondary acc/margin | secondary margin delta |
|---|---:|---:|---:|---:|
| atp_trainTrain_trainHyp | 1.000/19.330 | 1.000/19.377 | 1.000/19.274 | -1.398 |
| atp_trainTrain_heldHyp | 0.975/17.469 | 0.744/4.350 | 0.994/6.947 | -12.022 |
| atp_heldHeld_trainHyp | 1.000/17.708 | 0.512/2.993 | 0.500/1.817 | -1.147 |
| atp_dirDir_trainHyp | 1.000/17.156 | 0.094/-7.697 | 0.106/-7.360 | 1.250 |
| atp_trainTrain_stateFirst_trainHyp | 1.000/19.063 | 0.994/18.784 | 1.000/19.390 | -1.150 |
| atp_trainTrain_secondaryFirst_trainHyp | 1.000/19.179 | 1.000/18.574 | 1.000/19.368 | -1.345 |
| atp_stateOnly_train_trainHyp | 0.463/0.708 | 1.000/19.494 | 1.000/19.452 | -1.512 |
| atp_eventOnly_train_trainHyp | 1.000/19.466 | 0.500/-0.018 | 0.412/-0.349 | 0.389 |
| atp_trainEvent_nonState_trainHyp | 1.000/19.153 | 0.575/0.732 | 0.537/0.399 | -0.831 |
| atp_nonEvent_trainState_trainHyp | 0.475/0.421 | 1.000/19.235 | 1.000/19.023 | -1.557 |

## After update arm Etrue_Rtrue (update acc mean=1.000, base-anchor-after=1.000, seeds=1)

| eval | event acc/margin | focal acc/margin | secondary acc/margin | secondary margin delta |
|---|---:|---:|---:|---:|
| atp_trainTrain_trainHyp | 1.000/19.116 | 1.000/19.492 | 1.000/19.584 | -1.089 |
| atp_trainTrain_heldHyp | 0.969/17.494 | 0.794/8.910 | 1.000/15.862 | -3.106 |
| atp_heldHeld_trainHyp | 1.000/15.985 | 0.512/2.465 | 0.500/2.528 | -0.437 |
| atp_dirDir_trainHyp | 1.000/16.145 | 0.106/-6.396 | 0.094/-6.299 | 2.310 |
| atp_trainTrain_stateFirst_trainHyp | 1.000/19.020 | 0.994/18.786 | 1.000/19.767 | -0.774 |
| atp_trainTrain_secondaryFirst_trainHyp | 1.000/19.010 | 1.000/19.020 | 1.000/19.777 | -0.935 |
| atp_stateOnly_train_trainHyp | 0.500/0.229 | 1.000/19.860 | 1.000/19.927 | -1.036 |
| atp_eventOnly_train_trainHyp | 1.000/19.727 | 0.575/0.788 | 0.512/-0.027 | 0.712 |
| atp_trainEvent_nonState_trainHyp | 1.000/19.243 | 0.562/0.884 | 0.537/1.027 | -0.204 |
| atp_nonEvent_trainState_trainHyp | 0.550/0.897 | 0.988/18.834 | 1.000/19.472 | -1.107 |

## After update arm Eflip_Rtrue (update acc mean=0.804, base-anchor-after=0.794, seeds=1)

| eval | event acc/margin | focal acc/margin | secondary acc/margin | secondary margin delta |
|---|---:|---:|---:|---:|
| atp_trainTrain_trainHyp | 0.362/-0.018 | 1.000/24.792 | 1.000/25.131 | 4.458 |
| atp_trainTrain_heldHyp | 0.706/5.164 | 0.775/7.205 | 0.981/13.003 | -5.965 |
| atp_heldHeld_trainHyp | 0.450/0.082 | 0.562/1.941 | 0.606/3.406 | 0.441 |
| atp_dirDir_trainHyp | 0.281/0.106 | 0.212/-3.205 | 0.200/-3.818 | 4.791 |
| atp_trainTrain_stateFirst_trainHyp | 0.569/0.054 | 0.988/24.378 | 1.000/25.093 | 4.553 |
| atp_trainTrain_secondaryFirst_trainHyp | 0.650/0.060 | 1.000/24.987 | 1.000/25.070 | 4.357 |
| atp_stateOnly_train_trainHyp | 0.487/-0.069 | 1.000/25.428 | 1.000/25.516 | 4.553 |
| atp_eventOnly_train_trainHyp | 0.250/-0.016 | 0.575/0.117 | 0.388/0.228 | 0.966 |
| atp_trainEvent_nonState_trainHyp | 0.144/-0.003 | 0.519/0.343 | 0.581/-0.042 | -1.272 |
| atp_nonEvent_trainState_trainHyp | 0.594/0.016 | 1.000/24.982 | 1.000/25.041 | 4.462 |

## After update arm Etrue_Rflip (update acc mean=0.964, base-anchor-after=0.733, seeds=1)

| eval | event acc/margin | focal acc/margin | secondary acc/margin | secondary margin delta |
|---|---:|---:|---:|---:|
| atp_trainTrain_trainHyp | 1.000/21.384 | 0.356/-2.477 | 0.412/-1.746 | -22.419 |
| atp_trainTrain_heldHyp | 0.975/18.381 | 0.531/1.855 | 0.588/2.240 | -16.729 |
| atp_heldHeld_trainHyp | 1.000/19.643 | 0.537/1.231 | 0.425/-0.452 | -3.417 |
| atp_dirDir_trainHyp | 0.994/20.078 | 0.419/-0.040 | 0.412/-0.899 | 7.711 |
| atp_trainTrain_stateFirst_trainHyp | 1.000/21.218 | 0.500/-0.550 | 0.263/-3.288 | -23.828 |
| atp_trainTrain_secondaryFirst_trainHyp | 1.000/21.105 | 0.287/-4.356 | 0.350/-1.646 | -22.358 |
| atp_stateOnly_train_trainHyp | 0.312/-1.967 | 0.388/-1.747 | 0.287/-4.549 | -25.512 |
| atp_eventOnly_train_trainHyp | 1.000/23.613 | 0.463/-0.672 | 0.537/0.027 | 0.766 |
| atp_trainEvent_nonState_trainHyp | 1.000/21.534 | 0.475/-0.433 | 0.556/0.253 | -0.977 |
| atp_nonEvent_trainState_trainHyp | 0.494/0.413 | 0.463/-1.008 | 0.394/-2.094 | -22.673 |

## After update arm Eflip_Rflip (update acc mean=0.902, base-anchor-after=0.000, seeds=1)

| eval | event acc/margin | focal acc/margin | secondary acc/margin | secondary margin delta |
|---|---:|---:|---:|---:|
| atp_trainTrain_trainHyp | 0.000/-9.860 | 0.000/-10.148 | 0.000/-10.200 | -30.873 |
| atp_trainTrain_heldHyp | 0.037/-8.521 | 0.244/-3.719 | 0.000/-5.984 | -24.953 |
| atp_heldHeld_trainHyp | 0.013/-6.704 | 0.500/-1.695 | 0.475/-1.409 | -4.373 |
| atp_dirDir_trainHyp | 0.006/-7.799 | 0.800/3.595 | 0.863/4.202 | 12.811 |
| atp_trainTrain_stateFirst_trainHyp | 0.000/-9.503 | 0.025/-9.800 | 0.000/-10.185 | -30.726 |
| atp_trainTrain_secondaryFirst_trainHyp | 0.000/-9.337 | 0.025/-9.153 | 0.000/-10.035 | -30.748 |
| atp_stateOnly_train_trainHyp | 0.475/0.790 | 0.000/-10.969 | 0.000/-10.947 | -31.910 |
| atp_eventOnly_train_trainHyp | 0.000/-12.001 | 0.487/-0.943 | 0.575/0.576 | 1.314 |
| atp_trainEvent_nonState_trainHyp | 0.000/-9.624 | 0.494/-0.166 | 0.375/-0.447 | -1.677 |
| atp_nonEvent_trainState_trainHyp | 0.475/-0.254 | 0.013/-10.044 | 0.000/-10.256 | -30.836 |


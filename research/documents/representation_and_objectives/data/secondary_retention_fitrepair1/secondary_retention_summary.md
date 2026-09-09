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

## After update arm exposure (update acc mean=1.000, base-anchor-after=0.998, seeds=1)

| eval | event acc/margin | focal acc/margin | secondary acc/margin | secondary margin delta |
|---|---:|---:|---:|---:|
| atp_trainTrain_trainHyp | 1.000/16.194 | 1.000/16.408 | 1.000/16.143 | -4.530 |
| atp_trainTrain_heldHyp | 0.963/13.385 | 0.675/0.494 | 0.963/2.579 | -16.389 |
| atp_heldHeld_trainHyp | 0.988/14.583 | 0.531/3.321 | 0.500/2.909 | -0.055 |
| atp_dirDir_trainHyp | 1.000/14.753 | 0.212/-5.093 | 0.181/-5.037 | 3.573 |
| atp_trainTrain_stateFirst_trainHyp | 1.000/15.860 | 0.975/15.458 | 1.000/16.377 | -4.163 |
| atp_trainTrain_secondaryFirst_trainHyp | 1.000/15.985 | 0.994/15.890 | 1.000/16.567 | -4.145 |
| atp_stateOnly_train_trainHyp | 0.500/0.058 | 1.000/16.362 | 1.000/16.623 | -4.340 |
| atp_eventOnly_train_trainHyp | 1.000/16.327 | 0.525/0.015 | 0.425/0.366 | 1.105 |
| atp_trainEvent_nonState_trainHyp | 1.000/15.854 | 0.613/0.827 | 0.531/0.647 | -0.584 |
| atp_nonEvent_trainState_trainHyp | 0.500/0.452 | 1.000/16.344 | 1.000/16.370 | -4.209 |

## After update arm Etrue_Rtrue (update acc mean=1.000, base-anchor-after=1.000, seeds=1)

| eval | event acc/margin | focal acc/margin | secondary acc/margin | secondary margin delta |
|---|---:|---:|---:|---:|
| atp_trainTrain_trainHyp | 1.000/21.506 | 1.000/21.779 | 1.000/21.972 | 1.299 |
| atp_trainTrain_heldHyp | 0.812/11.679 | 1.000/16.015 | 1.000/17.915 | -1.054 |
| atp_heldHeld_trainHyp | 0.988/18.963 | 0.537/2.933 | 0.500/1.774 | -1.190 |
| atp_dirDir_trainHyp | 1.000/19.053 | 0.069/-9.339 | 0.100/-10.597 | -1.987 |
| atp_trainTrain_stateFirst_trainHyp | 1.000/21.506 | 1.000/21.637 | 1.000/22.061 | 1.521 |
| atp_trainTrain_secondaryFirst_trainHyp | 1.000/21.557 | 1.000/21.759 | 1.000/22.093 | 1.380 |
| atp_stateOnly_train_trainHyp | 0.425/-1.402 | 1.000/22.001 | 1.000/22.174 | 1.210 |
| atp_eventOnly_train_trainHyp | 1.000/22.105 | 0.613/0.502 | 0.487/-0.741 | -0.003 |
| atp_trainEvent_nonState_trainHyp | 1.000/21.248 | 0.506/1.506 | 0.562/1.245 | 0.015 |
| atp_nonEvent_trainState_trainHyp | 0.519/0.454 | 1.000/22.093 | 1.000/21.817 | 1.238 |

## After update arm Eflip_Rtrue (update acc mean=0.982, base-anchor-after=0.622, seeds=1)

| eval | event acc/margin | focal acc/margin | secondary acc/margin | secondary margin delta |
|---|---:|---:|---:|---:|
| atp_trainTrain_trainHyp | 0.106/-9.328 | 0.975/16.090 | 1.000/16.485 | -4.188 |
| atp_trainTrain_heldHyp | 0.569/0.723 | 0.644/0.682 | 0.719/1.431 | -17.537 |
| atp_heldHeld_trainHyp | 0.388/-2.236 | 0.562/1.038 | 0.537/2.756 | -0.209 |
| atp_dirDir_trainHyp | 0.256/-4.824 | 0.263/-5.401 | 0.275/-4.702 | 3.908 |
| atp_trainTrain_stateFirst_trainHyp | 0.106/-8.802 | 0.994/13.959 | 1.000/16.090 | -4.450 |
| atp_trainTrain_secondaryFirst_trainHyp | 0.169/-6.873 | 1.000/15.393 | 1.000/16.541 | -4.172 |
| atp_stateOnly_train_trainHyp | 0.550/0.039 | 1.000/16.217 | 1.000/16.601 | -4.362 |
| atp_eventOnly_train_trainHyp | 0.175/-7.862 | 0.588/1.773 | 0.450/-1.052 | -0.313 |
| atp_trainEvent_nonState_trainHyp | 0.037/-9.185 | 0.550/0.776 | 0.581/1.262 | 0.032 |
| atp_nonEvent_trainState_trainHyp | 0.475/-0.559 | 1.000/14.109 | 1.000/16.341 | -4.238 |

## After update arm Etrue_Rflip (update acc mean=1.000, base-anchor-after=0.843, seeds=1)

| eval | event acc/margin | focal acc/margin | secondary acc/margin | secondary margin delta |
|---|---:|---:|---:|---:|
| atp_trainTrain_trainHyp | 1.000/25.196 | 0.506/0.126 | 0.731/3.070 | -17.602 |
| atp_trainTrain_heldHyp | 0.988/22.315 | 0.537/0.347 | 0.781/6.260 | -12.708 |
| atp_heldHeld_trainHyp | 1.000/22.898 | 0.544/2.841 | 0.531/2.735 | -0.230 |
| atp_dirDir_trainHyp | 1.000/22.462 | 0.400/-2.738 | 0.338/-2.542 | 6.067 |
| atp_trainTrain_stateFirst_trainHyp | 1.000/25.372 | 0.544/2.556 | 0.656/2.552 | -17.988 |
| atp_trainTrain_secondaryFirst_trainHyp | 1.000/24.483 | 0.444/-2.671 | 0.637/2.596 | -18.117 |
| atp_stateOnly_train_trainHyp | 0.425/-1.553 | 0.588/2.156 | 0.450/-3.116 | -24.079 |
| atp_eventOnly_train_trainHyp | 1.000/25.648 | 0.450/-0.892 | 0.500/-1.835 | -1.097 |
| atp_trainEvent_nonState_trainHyp | 1.000/24.175 | 0.494/-1.142 | 0.500/0.300 | -0.930 |
| atp_nonEvent_trainState_trainHyp | 0.463/0.919 | 0.562/0.441 | 0.600/-0.194 | -20.773 |

## After update arm Eflip_Rflip (update acc mean=1.000, base-anchor-after=0.001, seeds=1)

| eval | event acc/margin | focal acc/margin | secondary acc/margin | secondary margin delta |
|---|---:|---:|---:|---:|
| atp_trainTrain_trainHyp | 0.000/-13.456 | 0.000/-14.312 | 0.000/-14.497 | -35.169 |
| atp_trainTrain_heldHyp | 0.037/-10.727 | 0.250/-2.496 | 0.006/-3.760 | -22.728 |
| atp_heldHeld_trainHyp | 0.000/-12.981 | 0.512/0.760 | 0.500/0.787 | -2.177 |
| atp_dirDir_trainHyp | 0.000/-10.615 | 0.850/7.106 | 0.938/6.688 | 15.298 |
| atp_trainTrain_stateFirst_trainHyp | 0.000/-12.544 | 0.019/-13.665 | 0.000/-14.497 | -35.038 |
| atp_trainTrain_secondaryFirst_trainHyp | 0.000/-12.693 | 0.044/-12.950 | 0.000/-14.265 | -34.978 |
| atp_stateOnly_train_trainHyp | 0.500/0.857 | 0.000/-16.019 | 0.000/-15.787 | -36.750 |
| atp_eventOnly_train_trainHyp | 0.000/-16.684 | 0.463/-0.810 | 0.487/-0.885 | -0.146 |
| atp_trainEvent_nonState_trainHyp | 0.000/-12.199 | 0.406/-0.335 | 0.362/-0.477 | -1.707 |
| atp_nonEvent_trainState_trainHyp | 0.381/-0.755 | 0.025/-12.667 | 0.000/-14.305 | -34.885 |


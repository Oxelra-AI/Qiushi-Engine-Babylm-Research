# secondary retention and orientation interference secondary retention without direct sparse secondary supervision

Base training uses anchor-wording event/focal/secondary labels. Sparse update rows omit secondary-state directional labels and replace them with neutral mention rows so update row counts stay matched. Positive contrastive margin means the true AB/BA fact has higher belief than the swapped alternative.

## Construction

- base rows: 2240 {'event_role': 960, 'focal_state': 640, 'untouched_state': 640} labels={'0': 1120, '1': 1120}
- update exposure: rows=224 queries={'mention_exposure': 224} labels={'0': 112, '1': 112}
- update Etrue_only: rows=224 queries={'event_role': 96, 'neutral_mention': 128} labels={'0': 112, '1': 112}
- update Eflip_only: rows=224 queries={'event_role': 96, 'neutral_mention': 128} labels={'0': 112, '1': 112}
- update Rtrue_only: rows=224 queries={'focal_state': 64, 'mention_exposure': 32, 'neutral_mention': 128} labels={'0': 112, '1': 112}
- update Rflip_only: rows=224 queries={'focal_state': 64, 'mention_exposure': 32, 'neutral_mention': 128} labels={'0': 112, '1': 112}

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

## After update arm Etrue_only (update acc mean=1.000, base-anchor-after=0.998, seeds=1)

| eval | event acc/margin | focal acc/margin | secondary acc/margin | secondary margin delta |
|---|---:|---:|---:|---:|
| atp_trainTrain_trainHyp | 1.000/23.882 | 1.000/22.098 | 1.000/22.457 | 1.784 |
| atp_trainTrain_heldHyp | 0.975/21.197 | 0.812/9.280 | 1.000/14.188 | -4.780 |
| atp_heldHeld_trainHyp | 1.000/22.330 | 0.512/1.063 | 0.500/1.075 | -1.889 |
| atp_dirDir_trainHyp | 1.000/21.822 | 0.131/-10.348 | 0.144/-9.678 | -1.068 |
| atp_trainTrain_stateFirst_trainHyp | 1.000/23.849 | 0.988/21.495 | 1.000/21.970 | 1.430 |
| atp_trainTrain_secondaryFirst_trainHyp | 1.000/23.790 | 0.988/21.806 | 1.000/21.909 | 1.196 |
| atp_stateOnly_train_trainHyp | 0.450/-0.049 | 1.000/22.576 | 1.000/22.849 | 1.886 |
| atp_eventOnly_train_trainHyp | 1.000/24.216 | 0.575/1.478 | 0.450/-0.610 | 0.128 |
| atp_trainEvent_nonState_trainHyp | 1.000/23.772 | 0.519/0.260 | 0.550/0.856 | -0.374 |
| atp_nonEvent_trainState_trainHyp | 0.494/0.717 | 0.988/20.872 | 1.000/21.843 | 1.264 |

## After update arm Eflip_only (update acc mean=1.000, base-anchor-after=0.001, seeds=1)

| eval | event acc/margin | focal acc/margin | secondary acc/margin | secondary margin delta |
|---|---:|---:|---:|---:|
| atp_trainTrain_trainHyp | 0.000/-19.621 | 0.000/-17.880 | 0.000/-18.303 | -38.976 |
| atp_trainTrain_heldHyp | 0.087/-15.219 | 0.237/-4.890 | 0.025/-6.763 | -25.732 |
| atp_heldHeld_trainHyp | 0.037/-14.923 | 0.481/-2.083 | 0.494/-2.264 | -5.228 |
| atp_dirDir_trainHyp | 0.037/-15.853 | 0.681/3.316 | 0.781/4.042 | 12.652 |
| atp_trainTrain_stateFirst_trainHyp | 0.000/-19.665 | 0.025/-17.215 | 0.000/-19.064 | -39.605 |
| atp_trainTrain_secondaryFirst_trainHyp | 0.006/-19.528 | 0.013/-16.752 | 0.000/-17.493 | -38.206 |
| atp_stateOnly_train_trainHyp | 0.487/-0.781 | 0.000/-18.963 | 0.000/-18.412 | -39.375 |
| atp_eventOnly_train_trainHyp | 0.000/-22.290 | 0.512/-0.019 | 0.550/0.977 | 1.715 |
| atp_trainEvent_nonState_trainHyp | 0.000/-19.301 | 0.525/0.437 | 0.444/-0.235 | -1.465 |
| atp_nonEvent_trainState_trainHyp | 0.569/0.923 | 0.000/-15.346 | 0.013/-18.313 | -38.892 |

## After update arm Rtrue_only (update acc mean=1.000, base-anchor-after=0.986, seeds=1)

| eval | event acc/margin | focal acc/margin | secondary acc/margin | secondary margin delta |
|---|---:|---:|---:|---:|
| atp_trainTrain_trainHyp | 1.000/15.365 | 1.000/19.350 | 1.000/19.759 | -0.914 |
| atp_trainTrain_heldHyp | 0.881/11.433 | 0.963/3.224 | 1.000/3.713 | -15.256 |
| atp_heldHeld_trainHyp | 0.762/7.626 | 0.512/0.747 | 0.512/1.320 | -1.644 |
| atp_dirDir_trainHyp | 0.925/9.313 | 0.156/-9.989 | 0.106/-10.260 | -1.651 |
| atp_trainTrain_stateFirst_trainHyp | 1.000/16.493 | 0.975/19.032 | 1.000/19.717 | -0.823 |
| atp_trainTrain_secondaryFirst_trainHyp | 0.994/16.331 | 0.975/18.991 | 1.000/19.571 | -1.141 |
| atp_stateOnly_train_trainHyp | 0.400/-1.142 | 1.000/19.335 | 1.000/19.738 | -1.225 |
| atp_eventOnly_train_trainHyp | 1.000/15.556 | 0.500/0.067 | 0.362/-1.788 | -1.050 |
| atp_trainEvent_nonState_trainHyp | 0.994/14.673 | 0.494/0.524 | 0.562/1.838 | 0.608 |
| atp_nonEvent_trainState_trainHyp | 0.506/0.159 | 1.000/19.307 | 1.000/19.266 | -1.313 |

## After update arm Rflip_only (update acc mean=0.996, base-anchor-after=0.328, seeds=1)

| eval | event acc/margin | focal acc/margin | secondary acc/margin | secondary margin delta |
|---|---:|---:|---:|---:|
| atp_trainTrain_trainHyp | 0.981/7.713 | 0.000/-14.357 | 0.000/-13.153 | -33.826 |
| atp_trainTrain_heldHyp | 0.787/3.536 | 0.169/-4.333 | 0.025/-6.288 | -25.256 |
| atp_heldHeld_trainHyp | 0.938/6.318 | 0.463/-1.497 | 0.481/-0.094 | -3.058 |
| atp_dirDir_trainHyp | 0.944/6.797 | 0.850/4.308 | 0.819/3.795 | 12.404 |
| atp_trainTrain_stateFirst_trainHyp | 0.950/8.270 | 0.025/-11.797 | 0.013/-13.006 | -33.546 |
| atp_trainTrain_secondaryFirst_trainHyp | 0.956/6.389 | 0.031/-12.037 | 0.000/-13.560 | -34.272 |
| atp_stateOnly_train_trainHyp | 0.388/-1.331 | 0.000/-15.242 | 0.000/-14.977 | -35.941 |
| atp_eventOnly_train_trainHyp | 0.988/9.956 | 0.463/-0.782 | 0.475/-0.263 | 0.475 |
| atp_trainEvent_nonState_trainHyp | 1.000/6.973 | 0.388/-1.254 | 0.487/0.017 | -1.213 |
| atp_nonEvent_trainState_trainHyp | 0.456/0.390 | 0.019/-13.384 | 0.000/-13.678 | -34.257 |


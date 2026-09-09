# independent orientation cross and context interface independent event/ranking orientation cross

Event and focal-ranking sparse labels vary independently; secondary ranking labels remain true. Evaluation labels are always true facts.

## Arm exposure (train acc mean=0.996, std=0.000, seeds=1)

| eval_set | con_event | con_focal | con_untouched | focal_conflict | untouched_conflict |
|---|---:|---:|---:|---:|---:|
| atp_trainTrain_trainHyp | 0.688 | 0.044 | 0.037 | 0.031 | 0.031 |
| atp_trainTrain_heldHyp | 0.456 | 0.087 | 0.050 | 0.062 | 0.062 |
| atp_heldHeld_trainHyp | 0.713 | 0.037 | 0.013 | 0.000 | 0.000 |
| atp_heldHeld_heldHyp | 0.487 | 0.100 | 0.062 | 0.062 | 0.094 |
| atp_dirDir_trainHyp | 0.938 | 0.106 | 0.069 | 0.078 | 0.000 |
| atp_trainEvent_dirState_trainHyp | 0.600 | 0.113 | 0.025 | 0.109 | 0.031 |
| atp_dirEvent_trainState_trainHyp | 0.944 | 0.013 | 0.044 | 0.000 | 0.109 |
| atp_trainEvent_nonState_trainHyp | 0.619 | 0.456 | 0.537 | 0.453 | 0.500 |
| atp_nonEvent_trainState_trainHyp | 0.475 | 0.125 | 0.081 | 0.125 | 0.062 |

## Arm Etrue_Rtrue (train acc mean=0.869, std=0.000, seeds=1)

| eval_set | con_event | con_focal | con_untouched | focal_conflict | untouched_conflict |
|---|---:|---:|---:|---:|---:|
| atp_trainTrain_trainHyp | 0.931 | 0.637 | 0.606 | 0.312 | 0.625 |
| atp_trainTrain_heldHyp | 0.812 | 0.619 | 0.519 | 0.250 | 0.500 |
| atp_heldHeld_trainHyp | 0.481 | 0.087 | 0.050 | 0.031 | 0.062 |
| atp_heldHeld_heldHyp | 0.463 | 0.106 | 0.025 | 0.062 | 0.031 |
| atp_dirDir_trainHyp | 0.581 | 0.144 | 0.019 | 0.016 | 0.016 |
| atp_trainEvent_dirState_trainHyp | 0.681 | 0.294 | 0.025 | 0.031 | 0.031 |
| atp_dirEvent_trainState_trainHyp | 0.781 | 0.619 | 0.556 | 0.391 | 0.609 |
| atp_trainEvent_nonState_trainHyp | 0.975 | 0.606 | 0.544 | 0.078 | 0.531 |
| atp_nonEvent_trainState_trainHyp | 0.500 | 0.613 | 0.556 | 0.641 | 0.484 |

## Arm Eflip_Rtrue (train acc mean=1.000, std=0.000, seeds=1)

| eval_set | con_event | con_focal | con_untouched | focal_conflict | untouched_conflict |
|---|---:|---:|---:|---:|---:|
| atp_trainTrain_trainHyp | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| atp_trainTrain_heldHyp | 0.275 | 0.850 | 0.994 | 1.000 | 1.000 |
| atp_heldHeld_trainHyp | 0.713 | 0.106 | 0.119 | 0.094 | 0.172 |
| atp_heldHeld_heldHyp | 0.481 | 0.225 | 0.094 | 0.141 | 0.156 |
| atp_dirDir_trainHyp | 0.944 | 0.250 | 0.069 | 0.031 | 0.062 |
| atp_trainEvent_dirState_trainHyp | 0.000 | 0.075 | 0.025 | 0.141 | 0.031 |
| atp_dirEvent_trainState_trainHyp | 0.969 | 1.000 | 1.000 | 1.000 | 1.000 |
| atp_trainEvent_nonState_trainHyp | 0.013 | 0.438 | 0.450 | 0.609 | 0.469 |
| atp_nonEvent_trainState_trainHyp | 0.531 | 0.994 | 1.000 | 0.984 | 1.000 |

## Arm Etrue_Rflip (train acc mean=0.844, std=0.000, seeds=1)

| eval_set | con_event | con_focal | con_untouched | focal_conflict | untouched_conflict |
|---|---:|---:|---:|---:|---:|
| atp_trainTrain_trainHyp | 0.675 | 0.537 | 0.775 | 0.359 | 0.812 |
| atp_trainTrain_heldHyp | 0.719 | 0.594 | 0.800 | 0.359 | 0.828 |
| atp_heldHeld_trainHyp | 0.425 | 0.056 | 0.013 | 0.031 | 0.031 |
| atp_heldHeld_heldHyp | 0.512 | 0.150 | 0.013 | 0.031 | 0.031 |
| atp_dirDir_trainHyp | 0.500 | 0.125 | 0.031 | 0.000 | 0.047 |
| atp_trainEvent_dirState_trainHyp | 0.694 | 0.206 | 0.013 | 0.000 | 0.031 |
| atp_dirEvent_trainState_trainHyp | 0.669 | 0.656 | 0.731 | 0.516 | 0.766 |
| atp_trainEvent_nonState_trainHyp | 0.863 | 0.575 | 0.475 | 0.188 | 0.484 |
| atp_nonEvent_trainState_trainHyp | 0.469 | 0.581 | 0.713 | 0.594 | 0.766 |

## Arm Eflip_Rflip (train acc mean=0.984, std=0.000, seeds=1)

| eval_set | con_event | con_focal | con_untouched | focal_conflict | untouched_conflict |
|---|---:|---:|---:|---:|---:|
| atp_trainTrain_trainHyp | 0.000 | 0.450 | 0.525 | 0.484 | 0.531 |
| atp_trainTrain_heldHyp | 0.056 | 0.444 | 0.600 | 0.891 | 0.531 |
| atp_heldHeld_trainHyp | 0.625 | 0.250 | 0.212 | 0.250 | 0.250 |
| atp_heldHeld_heldHyp | 0.519 | 0.319 | 0.294 | 0.281 | 0.266 |
| atp_dirDir_trainHyp | 0.919 | 0.331 | 0.275 | 0.203 | 0.266 |
| atp_trainEvent_dirState_trainHyp | 0.044 | 0.219 | 0.225 | 0.266 | 0.188 |
| atp_dirEvent_trainState_trainHyp | 0.925 | 0.494 | 0.562 | 0.469 | 0.562 |
| atp_trainEvent_nonState_trainHyp | 0.000 | 0.400 | 0.512 | 0.469 | 0.516 |
| atp_nonEvent_trainState_trainHyp | 0.544 | 0.531 | 0.494 | 0.516 | 0.438 |

## Selective movement relative to Etrue_Rtrue

Positive values mean accuracy against true facts dropped when the named sparse orientation was flipped.

| eval_set | event flip on event | event flip on focal | event flip on untouched | rank flip on event | rank flip on focal | rank flip on untouched |
|---|---:|---:|---:|---:|---:|---:|
| atp_trainTrain_trainHyp | 0.931 | -0.363 | -0.394 | 0.256 | 0.100 | -0.169 |
| atp_trainTrain_heldHyp | 0.537 | -0.231 | -0.475 | 0.094 | 0.025 | -0.281 |
| atp_heldHeld_trainHyp | -0.231 | -0.019 | -0.069 | 0.056 | 0.031 | 0.038 |
| atp_heldHeld_heldHyp | -0.019 | -0.119 | -0.069 | -0.050 | -0.044 | 0.013 |
| atp_dirDir_trainHyp | -0.362 | -0.106 | -0.050 | 0.081 | 0.019 | -0.013 |
| atp_trainEvent_dirState_trainHyp | 0.681 | 0.219 | 0.000 | -0.012 | 0.088 | 0.013 |
| atp_dirEvent_trainState_trainHyp | -0.188 | -0.381 | -0.444 | 0.113 | -0.037 | -0.175 |
| atp_trainEvent_nonState_trainHyp | 0.963 | 0.169 | 0.094 | 0.112 | 0.031 | 0.069 |
| atp_nonEvent_trainState_trainHyp | -0.031 | -0.381 | -0.444 | 0.031 | 0.031 | -0.156 |

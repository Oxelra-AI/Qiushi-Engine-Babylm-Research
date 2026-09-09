# independent orientation cross and context interface independent event/ranking orientation cross

Event and focal-ranking sparse labels vary independently; secondary ranking labels remain true. Evaluation labels are always true facts.

## Arm exposure (train acc mean=1.000, std=0.000, seeds=1)

| eval_set | con_event | con_focal | con_untouched | focal_conflict | untouched_conflict |
|---|---:|---:|---:|---:|---:|
| atp_trainTrain_trainHyp | 0.537 | 0.283 | 0.221 | 0.259 | 0.241 |
| atp_trainTrain_heldHyp | 0.446 | 0.279 | 0.300 | 0.315 | 0.278 |
| atp_heldHeld_trainHyp | 0.858 | 0.208 | 0.129 | 0.130 | 0.194 |
| atp_heldHeld_heldHyp | 0.608 | 0.333 | 0.163 | 0.194 | 0.102 |
| atp_dirDir_trainHyp | 0.963 | 0.117 | 0.100 | 0.083 | 0.065 |
| atp_trainEvent_dirState_trainHyp | 0.554 | 0.083 | 0.113 | 0.093 | 0.111 |
| atp_dirEvent_trainState_trainHyp | 0.992 | 0.333 | 0.312 | 0.287 | 0.324 |
| atp_trainEvent_nonState_trainHyp | 0.608 | 0.504 | 0.479 | 0.481 | 0.454 |
| atp_nonEvent_trainState_trainHyp | 0.504 | 0.375 | 0.192 | 0.343 | 0.148 |

## Arm Etrue_Rtrue (train acc mean=1.000, std=0.000, seeds=1)

| eval_set | con_event | con_focal | con_untouched | focal_conflict | untouched_conflict |
|---|---:|---:|---:|---:|---:|
| atp_trainTrain_trainHyp | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| atp_trainTrain_heldHyp | 0.775 | 0.850 | 1.000 | 0.667 | 1.000 |
| atp_heldHeld_trainHyp | 0.988 | 0.008 | 0.017 | 0.000 | 0.019 |
| atp_heldHeld_heldHyp | 0.667 | 0.175 | 0.058 | 0.019 | 0.037 |
| atp_dirDir_trainHyp | 0.929 | 0.025 | 0.008 | 0.046 | 0.019 |
| atp_trainEvent_dirState_trainHyp | 1.000 | 0.021 | 0.013 | 0.000 | 0.000 |
| atp_dirEvent_trainState_trainHyp | 0.967 | 1.000 | 1.000 | 1.000 | 1.000 |
| atp_trainEvent_nonState_trainHyp | 1.000 | 0.529 | 0.508 | 0.370 | 0.528 |
| atp_nonEvent_trainState_trainHyp | 0.521 | 0.996 | 1.000 | 0.991 | 1.000 |

## Arm Eflip_Rtrue (train acc mean=1.000, std=0.000, seeds=1)

| eval_set | con_event | con_focal | con_untouched | focal_conflict | untouched_conflict |
|---|---:|---:|---:|---:|---:|
| atp_trainTrain_trainHyp | 0.004 | 1.000 | 0.992 | 1.000 | 1.000 |
| atp_trainTrain_heldHyp | 0.442 | 0.804 | 0.992 | 1.000 | 0.981 |
| atp_heldHeld_trainHyp | 0.787 | 0.104 | 0.008 | 0.065 | 0.000 |
| atp_heldHeld_heldHyp | 0.467 | 0.163 | 0.021 | 0.102 | 0.009 |
| atp_dirDir_trainHyp | 0.996 | 0.104 | 0.121 | 0.028 | 0.102 |
| atp_trainEvent_dirState_trainHyp | 0.025 | 0.100 | 0.108 | 0.120 | 0.111 |
| atp_dirEvent_trainState_trainHyp | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| atp_trainEvent_nonState_trainHyp | 0.000 | 0.529 | 0.512 | 0.546 | 0.481 |
| atp_nonEvent_trainState_trainHyp | 0.504 | 0.992 | 1.000 | 0.981 | 1.000 |

## Arm Etrue_Rflip (train acc mean=1.000, std=0.000, seeds=1)

| eval_set | con_event | con_focal | con_untouched | focal_conflict | untouched_conflict |
|---|---:|---:|---:|---:|---:|
| atp_trainTrain_trainHyp | 1.000 | 0.196 | 0.988 | 0.028 | 0.991 |
| atp_trainTrain_heldHyp | 0.996 | 0.533 | 0.958 | 0.000 | 0.972 |
| atp_heldHeld_trainHyp | 0.967 | 0.000 | 0.008 | 0.000 | 0.019 |
| atp_heldHeld_heldHyp | 0.562 | 0.071 | 0.033 | 0.000 | 0.019 |
| atp_dirDir_trainHyp | 0.938 | 0.087 | 0.029 | 0.065 | 0.056 |
| atp_trainEvent_dirState_trainHyp | 0.983 | 0.183 | 0.025 | 0.074 | 0.009 |
| atp_dirEvent_trainState_trainHyp | 0.992 | 0.392 | 0.892 | 0.259 | 0.917 |
| atp_trainEvent_nonState_trainHyp | 1.000 | 0.446 | 0.517 | 0.324 | 0.500 |
| atp_nonEvent_trainState_trainHyp | 0.487 | 0.421 | 0.925 | 0.435 | 0.907 |

## Arm Eflip_Rflip (train acc mean=1.000, std=0.000, seeds=1)

| eval_set | con_event | con_focal | con_untouched | focal_conflict | untouched_conflict |
|---|---:|---:|---:|---:|---:|
| atp_trainTrain_trainHyp | 0.004 | 0.113 | 0.967 | 0.111 | 0.981 |
| atp_trainTrain_heldHyp | 0.017 | 0.279 | 0.954 | 0.574 | 0.898 |
| atp_heldHeld_trainHyp | 0.858 | 0.121 | 0.442 | 0.074 | 0.444 |
| atp_heldHeld_heldHyp | 0.733 | 0.142 | 0.217 | 0.074 | 0.167 |
| atp_dirDir_trainHyp | 0.996 | 0.242 | 0.192 | 0.194 | 0.213 |
| atp_trainEvent_dirState_trainHyp | 0.017 | 0.254 | 0.250 | 0.296 | 0.259 |
| atp_dirEvent_trainState_trainHyp | 0.983 | 0.321 | 0.762 | 0.352 | 0.843 |
| atp_trainEvent_nonState_trainHyp | 0.000 | 0.458 | 0.512 | 0.491 | 0.407 |
| atp_nonEvent_trainState_trainHyp | 0.521 | 0.362 | 0.725 | 0.361 | 0.741 |

## Selective movement relative to Etrue_Rtrue

Positive values mean accuracy against true facts dropped when the named sparse orientation was flipped.

| eval_set | event flip on event | event flip on focal | event flip on untouched | rank flip on event | rank flip on focal | rank flip on untouched |
|---|---:|---:|---:|---:|---:|---:|
| atp_trainTrain_trainHyp | 0.996 | 0.000 | 0.008 | 0.000 | 0.804 | 0.012 |
| atp_trainTrain_heldHyp | 0.333 | 0.046 | 0.008 | -0.221 | 0.317 | 0.042 |
| atp_heldHeld_trainHyp | 0.200 | -0.096 | 0.008 | 0.021 | 0.008 | 0.008 |
| atp_heldHeld_heldHyp | 0.200 | 0.012 | 0.038 | 0.104 | 0.104 | 0.025 |
| atp_dirDir_trainHyp | -0.067 | -0.079 | -0.113 | -0.008 | -0.062 | -0.021 |
| atp_trainEvent_dirState_trainHyp | 0.975 | -0.079 | -0.096 | 0.017 | -0.162 | -0.013 |
| atp_dirEvent_trainState_trainHyp | -0.033 | 0.000 | 0.000 | -0.025 | 0.608 | 0.108 |
| atp_trainEvent_nonState_trainHyp | 1.000 | 0.000 | -0.004 | 0.000 | 0.083 | -0.008 |
| atp_nonEvent_trainState_trainHyp | 0.017 | 0.004 | 0.000 | 0.033 | 0.575 | 0.075 |

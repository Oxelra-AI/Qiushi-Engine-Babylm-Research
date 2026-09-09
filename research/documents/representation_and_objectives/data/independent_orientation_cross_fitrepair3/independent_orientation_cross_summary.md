# independent orientation cross and context interface independent event/ranking orientation cross

Event and focal-ranking sparse labels vary independently; secondary ranking labels remain true. Evaluation labels are always true facts.

## Arm exposure (train acc mean=1.000, std=0.000, seeds=3)

| eval_set | con_event | con_focal | con_untouched | focal_conflict | untouched_conflict |
|---|---:|---:|---:|---:|---:|
| atp_trainTrain_trainHyp | 0.572 | 0.471 | 0.468 | 0.463 | 0.497 |
| atp_trainTrain_heldHyp | 0.542 | 0.479 | 0.522 | 0.438 | 0.466 |
| atp_heldHeld_trainHyp | 0.862 | 0.250 | 0.265 | 0.207 | 0.272 |
| atp_heldHeld_heldHyp | 0.618 | 0.322 | 0.235 | 0.170 | 0.191 |
| atp_dirDir_trainHyp | 0.967 | 0.137 | 0.169 | 0.139 | 0.154 |
| atp_trainEvent_dirState_trainHyp | 0.565 | 0.101 | 0.153 | 0.096 | 0.154 |
| atp_dirEvent_trainState_trainHyp | 0.987 | 0.503 | 0.500 | 0.478 | 0.463 |
| atp_trainEvent_nonState_trainHyp | 0.614 | 0.497 | 0.486 | 0.463 | 0.460 |
| atp_nonEvent_trainState_trainHyp | 0.492 | 0.522 | 0.449 | 0.503 | 0.423 |

## Arm Etrue_Rtrue (train acc mean=1.000, std=0.000, seeds=3)

| eval_set | con_event | con_focal | con_untouched | focal_conflict | untouched_conflict |
|---|---:|---:|---:|---:|---:|
| atp_trainTrain_trainHyp | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| atp_trainTrain_heldHyp | 0.744 | 0.829 | 1.000 | 0.620 | 1.000 |
| atp_heldHeld_trainHyp | 0.924 | 0.147 | 0.169 | 0.139 | 0.176 |
| atp_heldHeld_heldHyp | 0.608 | 0.215 | 0.190 | 0.102 | 0.185 |
| atp_dirDir_trainHyp | 0.854 | 0.031 | 0.031 | 0.031 | 0.019 |
| atp_trainEvent_dirState_trainHyp | 0.901 | 0.040 | 0.024 | 0.012 | 0.025 |
| atp_dirEvent_trainState_trainHyp | 0.978 | 1.000 | 1.000 | 1.000 | 1.000 |
| atp_trainEvent_nonState_trainHyp | 1.000 | 0.479 | 0.500 | 0.364 | 0.522 |
| atp_nonEvent_trainState_trainHyp | 0.532 | 0.994 | 1.000 | 0.994 | 1.000 |

## Arm Eflip_Rtrue (train acc mean=1.000, std=0.000, seeds=3)

| eval_set | con_event | con_focal | con_untouched | focal_conflict | untouched_conflict |
|---|---:|---:|---:|---:|---:|
| atp_trainTrain_trainHyp | 0.001 | 0.997 | 0.997 | 1.000 | 1.000 |
| atp_trainTrain_heldHyp | 0.203 | 0.711 | 0.997 | 1.000 | 0.994 |
| atp_heldHeld_trainHyp | 0.625 | 0.229 | 0.229 | 0.194 | 0.293 |
| atp_heldHeld_heldHyp | 0.519 | 0.178 | 0.076 | 0.123 | 0.093 |
| atp_dirDir_trainHyp | 0.978 | 0.158 | 0.175 | 0.096 | 0.157 |
| atp_trainEvent_dirState_trainHyp | 0.008 | 0.153 | 0.124 | 0.201 | 0.099 |
| atp_dirEvent_trainState_trainHyp | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| atp_trainEvent_nonState_trainHyp | 0.000 | 0.478 | 0.493 | 0.534 | 0.485 |
| atp_nonEvent_trainState_trainHyp | 0.482 | 0.993 | 1.000 | 0.991 | 1.000 |

## Arm Etrue_Rflip (train acc mean=1.000, std=0.000, seeds=3)

| eval_set | con_event | con_focal | con_untouched | focal_conflict | untouched_conflict |
|---|---:|---:|---:|---:|---:|
| atp_trainTrain_trainHyp | 1.000 | 0.267 | 0.960 | 0.120 | 0.966 |
| atp_trainTrain_heldHyp | 0.874 | 0.438 | 0.965 | 0.025 | 0.951 |
| atp_heldHeld_trainHyp | 0.953 | 0.131 | 0.236 | 0.111 | 0.231 |
| atp_heldHeld_heldHyp | 0.569 | 0.193 | 0.185 | 0.102 | 0.179 |
| atp_dirDir_trainHyp | 0.838 | 0.151 | 0.094 | 0.133 | 0.123 |
| atp_trainEvent_dirState_trainHyp | 0.994 | 0.217 | 0.104 | 0.093 | 0.099 |
| atp_dirEvent_trainState_trainHyp | 0.964 | 0.539 | 0.890 | 0.451 | 0.892 |
| atp_trainEvent_nonState_trainHyp | 1.000 | 0.474 | 0.493 | 0.346 | 0.444 |
| atp_nonEvent_trainState_trainHyp | 0.487 | 0.519 | 0.874 | 0.537 | 0.840 |

## Arm Eflip_Rflip (train acc mean=0.833, std=0.236, seeds=3)

| eval_set | con_event | con_focal | con_untouched | focal_conflict | untouched_conflict |
|---|---:|---:|---:|---:|---:|
| atp_trainTrain_trainHyp | 0.175 | 0.303 | 0.749 | 0.287 | 0.719 |
| atp_trainTrain_heldHyp | 0.283 | 0.375 | 0.743 | 0.525 | 0.704 |
| atp_heldHeld_trainHyp | 0.699 | 0.310 | 0.524 | 0.269 | 0.512 |
| atp_heldHeld_heldHyp | 0.581 | 0.249 | 0.343 | 0.247 | 0.315 |
| atp_dirDir_trainHyp | 0.822 | 0.390 | 0.353 | 0.380 | 0.340 |
| atp_trainEvent_dirState_trainHyp | 0.183 | 0.375 | 0.321 | 0.401 | 0.321 |
| atp_dirEvent_trainState_trainHyp | 0.815 | 0.406 | 0.665 | 0.414 | 0.701 |
| atp_trainEvent_nonState_trainHyp | 0.163 | 0.486 | 0.483 | 0.509 | 0.392 |
| atp_nonEvent_trainState_trainHyp | 0.481 | 0.429 | 0.621 | 0.429 | 0.642 |

## Selective movement relative to Etrue_Rtrue

Positive values mean accuracy against true facts dropped when the named sparse orientation was flipped.

| eval_set | event flip on event | event flip on focal | event flip on untouched | rank flip on event | rank flip on focal | rank flip on untouched |
|---|---:|---:|---:|---:|---:|---:|
| atp_trainTrain_trainHyp | 0.999 | 0.003 | 0.003 | 0.000 | 0.733 | 0.040 |
| atp_trainTrain_heldHyp | 0.542 | 0.118 | 0.003 | -0.129 | 0.392 | 0.035 |
| atp_heldHeld_trainHyp | 0.299 | -0.082 | -0.060 | -0.029 | 0.017 | -0.067 |
| atp_heldHeld_heldHyp | 0.089 | 0.038 | 0.114 | 0.039 | 0.022 | 0.006 |
| atp_dirDir_trainHyp | -0.124 | -0.128 | -0.144 | 0.017 | -0.121 | -0.064 |
| atp_trainEvent_dirState_trainHyp | 0.893 | -0.113 | -0.100 | -0.093 | -0.176 | -0.081 |
| atp_dirEvent_trainState_trainHyp | -0.022 | 0.000 | 0.000 | 0.014 | 0.461 | 0.110 |
| atp_trainEvent_nonState_trainHyp | 1.000 | 0.001 | 0.007 | 0.000 | 0.006 | 0.007 |
| atp_nonEvent_trainState_trainHyp | 0.050 | 0.001 | 0.000 | 0.044 | 0.475 | 0.126 |

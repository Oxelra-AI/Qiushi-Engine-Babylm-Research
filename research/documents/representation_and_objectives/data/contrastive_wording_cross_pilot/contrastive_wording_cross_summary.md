# contrastive wording cross and grounding contrastive wording-cross probe
## Arm: aligned (train acc 1.000, 1 seeds)

| eval_set | std_acc | std_event | std_focal | std_untouched | con_acc | con_event | con_focal | con_untouched |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| atp_anchorCtx_trainHyp | 0.999 | 1.000 | 0.998 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| atp_heldCtx_heldHyp | 0.365 | 0.650 | 0.298 | 0.148 | 0.258 | 0.642 | 0.133 | 0.000 |
| atp_heldCtx_trainHyp | 0.342 | 0.975 | 0.023 | 0.027 | 0.332 | 0.983 | 0.004 | 0.008 |
| atp_trainCtx_heldHyp | 0.858 | 0.854 | 0.740 | 0.979 | 0.872 | 0.817 | 0.800 | 1.000 |
| atp_trainCtx_trainHyp | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| pair_heldCtx_heldHyp | 1.000 | 1.000 | nan | nan | 1.000 | 1.000 | nan | nan |
| pair_heldCtx_trainHyp | 1.000 | 1.000 | nan | nan | 1.000 | 1.000 | nan | nan |
| pair_trainCtx_heldHyp | 1.000 | 1.000 | nan | nan | 1.000 | 1.000 | nan | nan |
| pair_trainCtx_trainHyp | 1.000 | 1.000 | nan | nan | 1.000 | 1.000 | nan | nan |

## Arm: exposure (train acc 1.000, 1 seeds)

| eval_set | std_acc | std_event | std_focal | std_untouched | con_acc | con_event | con_focal | con_untouched |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| atp_anchorCtx_trainHyp | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| atp_heldCtx_heldHyp | 0.292 | 0.517 | 0.190 | 0.169 | 0.221 | 0.508 | 0.150 | 0.004 |
| atp_heldCtx_trainHyp | 0.333 | 0.794 | 0.079 | 0.127 | 0.329 | 0.925 | 0.025 | 0.037 |
| atp_trainCtx_heldHyp | 0.490 | 0.512 | 0.463 | 0.494 | 0.496 | 0.529 | 0.467 | 0.492 |
| atp_trainCtx_trainHyp | 0.516 | 0.619 | 0.460 | 0.469 | 0.514 | 0.608 | 0.467 | 0.467 |
| pair_heldCtx_heldHyp | 0.966 | 0.966 | nan | nan | 0.981 | 0.981 | nan | nan |
| pair_heldCtx_trainHyp | 0.972 | 0.972 | nan | nan | 0.988 | 0.988 | nan | nan |
| pair_trainCtx_heldHyp | 0.725 | 0.725 | nan | nan | 0.738 | 0.738 | nan | nan |
| pair_trainCtx_trainHyp | 0.722 | 0.722 | nan | nan | 0.738 | 0.738 | nan | nan |

## Arm: flipped (train acc 1.000, 1 seeds)

| eval_set | std_acc | std_event | std_focal | std_untouched | con_acc | con_event | con_focal | con_untouched |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| atp_anchorCtx_trainHyp | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| atp_heldCtx_heldHyp | 0.433 | 0.487 | 0.398 | 0.412 | 0.464 | 0.467 | 0.458 | 0.467 |
| atp_heldCtx_trainHyp | 0.442 | 0.621 | 0.340 | 0.365 | 0.529 | 0.762 | 0.371 | 0.454 |
| atp_trainCtx_heldHyp | 0.085 | 0.125 | 0.113 | 0.017 | 0.053 | 0.071 | 0.087 | 0.000 |
| atp_trainCtx_trainHyp | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| pair_heldCtx_heldHyp | 0.628 | 0.628 | nan | nan | 0.594 | 0.594 | nan | nan |
| pair_heldCtx_trainHyp | 0.731 | 0.731 | nan | nan | 0.719 | 0.719 | nan | nan |
| pair_trainCtx_heldHyp | 0.000 | 0.000 | nan | nan | 0.000 | 0.000 | nan | nan |
| pair_trainCtx_trainHyp | 0.000 | 0.000 | nan | nan | 0.000 | 0.000 | nan | nan |


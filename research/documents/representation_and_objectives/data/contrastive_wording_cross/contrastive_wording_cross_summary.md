# contrastive wording cross and grounding contrastive wording-cross probe
## Arm: aligned (train acc 1.000, 3 seeds)

| eval_set | std_acc | std_event | std_focal | std_untouched | con_acc | con_event | con_focal | con_untouched |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| atp_anchorCtx_trainHyp | 0.994 | 1.000 | 0.992 | 0.990 | 0.994 | 1.000 | 0.994 | 0.989 |
| atp_heldCtx_heldHyp | 0.416 | 0.582 | 0.332 | 0.333 | 0.346 | 0.560 | 0.262 | 0.217 |
| atp_heldCtx_trainHyp | 0.480 | 0.958 | 0.243 | 0.238 | 0.451 | 0.974 | 0.203 | 0.176 |
| atp_trainCtx_heldHyp | 0.838 | 0.665 | 0.855 | 0.993 | 0.834 | 0.642 | 0.862 | 0.997 |
| atp_trainCtx_trainHyp | 0.994 | 1.000 | 0.995 | 0.987 | 0.994 | 1.000 | 0.994 | 0.989 |
| pair_heldCtx_heldHyp | 1.000 | 1.000 | nan | nan | 1.000 | 1.000 | nan | nan |
| pair_heldCtx_trainHyp | 0.996 | 0.996 | nan | nan | 1.000 | 1.000 | nan | nan |
| pair_trainCtx_heldHyp | 1.000 | 1.000 | nan | nan | 1.000 | 1.000 | nan | nan |
| pair_trainCtx_trainHyp | 1.000 | 1.000 | nan | nan | 1.000 | 1.000 | nan | nan |

## Arm: exposure (train acc 1.000, 3 seeds)

| eval_set | std_acc | std_event | std_focal | std_untouched | con_acc | con_event | con_focal | con_untouched |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| atp_anchorCtx_trainHyp | 0.994 | 1.000 | 0.990 | 0.994 | 0.994 | 1.000 | 0.991 | 0.992 |
| atp_heldCtx_heldHyp | 0.408 | 0.659 | 0.339 | 0.226 | 0.383 | 0.661 | 0.298 | 0.190 |
| atp_heldCtx_trainHyp | 0.456 | 0.838 | 0.273 | 0.257 | 0.434 | 0.840 | 0.231 | 0.233 |
| atp_trainCtx_heldHyp | 0.429 | 0.514 | 0.394 | 0.380 | 0.415 | 0.544 | 0.351 | 0.350 |
| atp_trainCtx_trainHyp | 0.411 | 0.517 | 0.351 | 0.366 | 0.420 | 0.568 | 0.342 | 0.351 |
| pair_heldCtx_heldHyp | 0.855 | 0.855 | nan | nan | 0.855 | 0.855 | nan | nan |
| pair_heldCtx_trainHyp | 0.825 | 0.825 | nan | nan | 0.826 | 0.826 | nan | nan |
| pair_trainCtx_heldHyp | 0.647 | 0.647 | nan | nan | 0.655 | 0.655 | nan | nan |
| pair_trainCtx_trainHyp | 0.644 | 0.644 | nan | nan | 0.652 | 0.652 | nan | nan |

## Arm: flipped (train acc 1.000, 3 seeds)

| eval_set | std_acc | std_event | std_focal | std_untouched | con_acc | con_event | con_focal | con_untouched |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| atp_anchorCtx_trainHyp | 0.994 | 1.000 | 0.992 | 0.992 | 0.994 | 1.000 | 0.990 | 0.993 |
| atp_heldCtx_heldHyp | 0.390 | 0.557 | 0.315 | 0.299 | 0.365 | 0.532 | 0.280 | 0.284 |
| atp_heldCtx_trainHyp | 0.420 | 0.686 | 0.268 | 0.307 | 0.415 | 0.682 | 0.258 | 0.305 |
| atp_trainCtx_heldHyp | 0.163 | 0.198 | 0.213 | 0.079 | 0.146 | 0.222 | 0.209 | 0.008 |
| atp_trainCtx_trainHyp | 0.007 | 0.005 | 0.008 | 0.008 | 0.006 | 0.000 | 0.008 | 0.008 |
| pair_heldCtx_heldHyp | 0.687 | 0.687 | nan | nan | 0.655 | 0.655 | nan | nan |
| pair_heldCtx_trainHyp | 0.693 | 0.693 | nan | nan | 0.646 | 0.646 | nan | nan |
| pair_trainCtx_heldHyp | 0.000 | 0.000 | nan | nan | 0.000 | 0.000 | nan | nan |
| pair_trainCtx_trainHyp | 0.000 | 0.000 | nan | nan | 0.000 | 0.000 | nan | nan |


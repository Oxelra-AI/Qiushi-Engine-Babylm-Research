# Step019b result: embedding-role decomposition of held-symbol binding loss

This note is generated from `data/revision_019b_embedding_role_decomposition/results.json`. It uses query-first answer-only prepared checkpoints and continues with the full next-token objective while manipulating held entity input rows and tied/untied output roles.

## Branch-level final summary

| branch | train strong | abs held strong | final train4 | final held4 | final heldB | final heldSel | blocked train4 | held input L2 | be500 held4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tied_carry_full | 3/3 | 0/3 | 1.000 | 0.389 | 3.119 | 0.188 | 0.242 | 0.691 | 0.388 |
| tied_carry_freeze_held_shared | 3/3 | 0/3 | 1.000 | 0.367 | 2.745 | 0.175 | 0.244 | 0.000 | 0.371 |
| tied_reset_full | 2/3 | 0/3 | 0.921 | 0.389 | 3.003 | 0.217 | 0.255 | 0.779 | 0.406 |
| tied_reset_freeze_held_shared | 2/3 | 0/3 | 0.922 | 0.370 | 2.633 | 0.191 | 0.264 | 0.000 | 0.406 |
| untied_reset_full | 2/3 | 0/3 | 0.921 | 0.395 | 3.047 | 0.217 | 0.257 | 0.112 | 0.418 |
| untied_reset_freeze_held_input | 2/3 | 0/3 | 0.921 | 0.378 | 2.938 | 0.207 | 0.257 | 0.000 | 0.408 |

## Per-seed final summary

| seed | branch | prep held4 | final train4 | final held4 | Δheld4 | final heldB | ΔheldB | final heldSel | ΔheldSel | held input L2 | blocked train4 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | tied_carry_full | 0.484 | 1.000 | 0.473 | -0.012 | +5.300 | +1.467 | +0.268 | -0.097 | 0.993 | 0.275 |
| 42 | tied_carry_freeze_held_shared | 0.484 | 1.000 | 0.449 | -0.035 | +4.412 | +0.579 | +0.246 | -0.119 | 0.000 | 0.275 |
| 42 | tied_reset_full | 0.484 | 1.000 | 0.473 | -0.012 | +4.706 | +0.874 | +0.308 | -0.057 | 0.514 | 0.271 |
| 42 | tied_reset_freeze_held_shared | 0.484 | 1.000 | 0.465 | -0.020 | +4.503 | +0.670 | +0.284 | -0.081 | 0.000 | 0.277 |
| 42 | untied_reset_full | 0.484 | 1.000 | 0.465 | -0.020 | +4.512 | +0.679 | +0.273 | -0.092 | 0.134 | 0.275 |
| 42 | untied_reset_freeze_held_input | 0.484 | 1.000 | 0.465 | -0.020 | +4.431 | +0.598 | +0.270 | -0.095 | 0.000 | 0.275 |
| 43 | tied_carry_full | 0.820 | 1.000 | 0.457 | -0.363 | +3.022 | -2.481 | +0.325 | -0.401 | 0.470 | 0.215 |
| 43 | tied_carry_freeze_held_shared | 0.820 | 1.000 | 0.414 | -0.406 | +2.907 | -2.596 | +0.307 | -0.419 | 0.000 | 0.207 |
| 43 | tied_reset_full | 0.820 | 0.764 | 0.387 | -0.434 | +2.142 | -3.362 | +0.286 | -0.440 | 0.230 | 0.248 |
| 43 | tied_reset_freeze_held_shared | 0.820 | 0.766 | 0.371 | -0.449 | +2.071 | -3.432 | +0.275 | -0.451 | 0.000 | 0.250 |
| 43 | untied_reset_full | 0.820 | 0.764 | 0.406 | -0.414 | +2.291 | -3.213 | +0.301 | -0.425 | 0.090 | 0.248 |
| 43 | untied_reset_freeze_held_input | 0.820 | 0.764 | 0.383 | -0.438 | +2.206 | -3.298 | +0.291 | -0.436 | 0.000 | 0.248 |
| 100 | tied_carry_full | 0.953 | 1.000 | 0.238 | -0.715 | +1.035 | -6.739 | -0.029 | -0.942 | 0.609 | 0.236 |
| 100 | tied_carry_freeze_held_shared | 0.953 | 1.000 | 0.238 | -0.715 | +0.916 | -6.858 | -0.029 | -0.943 | 0.000 | 0.250 |
| 100 | tied_reset_full | 0.953 | 1.000 | 0.309 | -0.645 | +2.161 | -5.613 | +0.058 | -0.856 | 1.592 | 0.244 |
| 100 | tied_reset_freeze_held_shared | 0.953 | 1.000 | 0.273 | -0.680 | +1.326 | -6.448 | +0.014 | -0.900 | 0.000 | 0.264 |
| 100 | untied_reset_full | 0.953 | 1.000 | 0.312 | -0.641 | +2.338 | -5.435 | +0.078 | -0.836 | 0.112 | 0.246 |
| 100 | untied_reset_freeze_held_input | 0.953 | 1.000 | 0.285 | -0.668 | +2.178 | -5.596 | +0.061 | -0.853 | 0.000 | 0.246 |

## Tied-branch zero-training hybrids

`final_FF` is an untied copy of the final tied model. `final_PF` restores only held input rows to preparation values while leaving output rows and the rest of the model final. `final_FP` restores held output rows only. `final_PP` restores both held input and output rows. `prep_FP` inserts final held input rows into the preparation network.

| seed | branch | prep held4 | final_FF held4 | final_PF held4 | final_FP held4 | final_PP held4 | prep_FP held4 | PF gap-closed held4 | PF gap-closed heldB |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | tied_carry_full | 0.484 | 0.473 | 0.465 | 0.473 | 0.465 | 0.484 | -0.667 | 0.588 |
| 42 | tied_carry_freeze_held_shared | 0.484 | 0.449 | 0.449 | 0.449 | 0.449 | 0.484 | 0.000 | -0.000 |
| 42 | tied_reset_full | 0.484 | 0.473 | 0.461 | 0.473 | 0.461 | 0.484 | -1.000 | 0.498 |
| 42 | tied_reset_freeze_held_shared | 0.484 | 0.465 | 0.465 | 0.465 | 0.465 | 0.484 | 0.000 | -0.000 |
| 43 | tied_carry_full | 0.820 | 0.457 | 0.426 | 0.457 | 0.426 | 0.840 | -0.086 | -0.030 |
| 43 | tied_carry_freeze_held_shared | 0.820 | 0.414 | 0.414 | 0.414 | 0.414 | 0.820 | 0.000 | 0.000 |
| 43 | tied_reset_full | 0.820 | 0.387 | 0.371 | 0.387 | 0.371 | 0.836 | -0.036 | -0.024 |
| 43 | tied_reset_freeze_held_shared | 0.820 | 0.371 | 0.371 | 0.371 | 0.371 | 0.820 | 0.000 | 0.000 |
| 100 | tied_carry_full | 0.953 | 0.238 | 0.238 | 0.238 | 0.238 | 0.965 | 0.000 | -0.008 |
| 100 | tied_carry_freeze_held_shared | 0.953 | 0.238 | 0.238 | 0.238 | 0.238 | 0.953 | 0.000 | 0.000 |
| 100 | tied_reset_full | 0.953 | 0.309 | 0.297 | 0.309 | 0.297 | 0.965 | -0.018 | -0.019 |
| 100 | tied_reset_freeze_held_shared | 0.953 | 0.273 | 0.273 | 0.273 | 0.273 | 0.953 | 0.000 | 0.000 |

## Scientific reading

Input-row drift is supported only if `final_PF` selectively restores held behavior, or if inserting final held input rows into the preparation network destroys held behavior. A null `final_PF` with strong familiar-symbol binding points to contextual network/readout specialization or to a more distributed geometry change. The tied shared-row freeze is not input-only; it anchors the same row in both input and output roles. Untied branches test global removal of weight sharing but also change optimization for the entire output classifier.

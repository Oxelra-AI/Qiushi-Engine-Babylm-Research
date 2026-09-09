# entity leg reconstruction and second basin plan Entity leg reconstruction and second-basin readout plan

This note freezes the quantitative reading before the second-basin MAX and breadth score tables arrive. It uses only existing stable-family score files and the CPU-safe MAX-repeat 100M EWoK partial row; no model inference, GlobalPIQA, SuperGLUE, AoA, upload, or leaderboard action was involved.

## First-basin Entity leg

Source: `research/documents/frontier_consolidation/data/entity_leg_reconstruction/entity_leg_reconstruction_summary.md`.

On the common 10M--80M window, the semantic leg `V-R` for Entity is:

| dose | rho | Entity V-R | cheap6 contribution | cheap5 contribution |
|---:|---:|---:|---:|---:|
| 1x | 0.042352 | +0.55875 | +0.09313 | +0.11175 |
| 1.82x | 0.077120 | +1.06625 | +0.17771 | +0.21325 |
| MAX / 2.64x | 0.111872 | +2.06250 | +0.34375 | +0.41250 |

The three-point ordinary fit has slope 21.63 Entity score-points per rho and R^2 0.966; a through-origin fit gives 16.62 score-points per rho. The finite slope mildly accelerates from 14.60 score-points/rho between 1x and 1.82x to 28.67 between 1.82x and MAX. The important practical fact is not exact linearity but that the 1x family signal is diluted to ~+0.09 cheap6, the same scale as the 1x trajectory-mean cheap6 seed difference (+0.1133) and far below pointwise seed movement. This reconciles the earlier minimum-dose GPT2/RoBERTa/extractive/common-copy near-nulls: as six-family averages at the smallest dose they were not capable of resolving this Entity carrier.

## EWoK reading before hardening a trade frame

EWoK relative `V-R` is negative in the common 10M--80M table, but absolute scores hover near 50, and MAX recovers late:

- 1x common 10M--80M mean EWoK V-R: -0.8825; view mean 49.3825, repeat mean 50.265.
- 1.82x common mean EWoK V-R: -1.1650; view mean 48.7875, repeat mean 49.9525.
- MAX common mean EWoK V-R: -0.60625; view mean 49.6525, repeat mean 50.25875.
- MAX available 10M--100M mean EWoK V-R after CPU-safe 100M recovery: -0.4110; view mean 49.782, repeat mean 50.193; 80% of paired checkpoints have both scores in [49,51].
- MAX late endpoints are not EWoK-down: 90M V-R = +0.47, 100M V-R = +0.27.

Therefore the current first-basin result should be phrased as a dose-scaled Entity/state-tracking gain under source-conditioned re-expression, not yet as a hardened Entity-up/EWoK-down trade. EWoK remains a near-chance context and possible allocation signal unless the pending full ladder or continuous-margin jobs show a stable movement away from 50.

## Frozen prediction for incoming tables

The second-basin MAX view-minus-repeat Entity leg should be near +2 score points on the same stable-family readout, much closer to the first-basin MAX value (+2.0625) than to the 1x value (+0.55875). If it is near zero or close to the 1x scale, the dose-scaled Entity mechanism weakens substantially and should be revised.

MAX breadth uses the same companion-word budget for additional same-population sentences. If source-conditioned re-expression is the carrier, breadth should fall well short of the compact-view Entity lift. If generic non-duplicate same-population experience is sufficient, breadth should approach the compact-view Entity level.

address-coordinate result names a compatible object from another side: repeated address/state mentions can create collisions for changed records, while useful coordinates transfer under extra tag-free filler. The shared mechanism candidate is record addressability under a scarce fixed budget: re-expression can create useful alternative access to the same record, whereas exact duplicate recurrence can waste or collide with record coordinates.

## Execution state

Both second-basin MAX DeBERTa trainings are now complete and checkpointed:

- repeat: `experiments/archive/frontier_consolidation/training/runs/full_p2c_c2p_abs_repeat_dose2p64x_matched_rowholdout_deberta100M_seed43122`, exact 100M words, 2552 updates, final loss 2.348378896713257, ten checkpoints.
- view: `experiments/archive/frontier_consolidation/training/runs/full_p2c_c2p_abs_view_dose2p64x_matched_rowholdout_deberta100M_seed43122`, exact 100M words, 2552 updates, final loss 2.3948090076446533, ten checkpoints.

I created `experiments/archive/frontier_consolidation/scripts/cpu_safe_arm_ladder_eval.py`, a thin driver around the binding content trade predeclared predictions CPU-safe worker, and plan-checked both second-basin EWoK+Entity ladders. I submitted the two CPU-safe EWoK+Entity ladder jobs:

- second-basin MAX repeat EWoK+Entity, output root `experiments/archive/frontier_consolidation/data/second_basin_entity_ewok_eval/repeat_eval`.
- second-basin MAX view EWoK+Entity, output root `experiments/archive/frontier_consolidation/data/second_basin_entity_ewok_eval/view_eval`.

I also created and AST-checked `experiments/archive/frontier_consolidation/scripts/second_basin_entity_ewok_readout.py`. Once both jobs complete, run it without `--require-complete` first to merge view/repeat and produce `experiments/archive/frontier_consolidation/data/second_basin_entity_ewok_readout`; if complete, compare the common 10M--80M Entity mean against the frozen +2.0625 prediction and inspect EWoK absolute levels before deciding whether to launch full stable-family scoring for the second basin.

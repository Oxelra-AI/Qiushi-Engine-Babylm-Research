# earlier analysis — scale1.75 family trajectory readout

Readout from saved item-flip JSONs only; no model inference.

## cheap7 path

| exposure | base cheap7 | scale1.75 cheap7 | delta | discrete mean delta | net items |
|---|---:|---:|---:|---:|---:|
| 20M | 39.663571 | 40.305714 | +0.642143 | +0.745810 | +40 |
| 50M | 41.972857 | 42.384286 | +0.411429 | +0.518314 | +1951 |
| 70M | 42.608571 | 42.670000 | +0.061429 | +0.188927 | +1445 |
| 80M | 42.948571 | 43.812143 | +0.863571 | +1.008917 | +1087 |
| 100M | 43.005714 | 43.542143 | +0.536429 | +0.596729 | +1707 |

## Selected group net percentages (20M, 50M, 70M, 80M, 100M)

| column | group | path |
|---|---|---:|
| BLiMP | existential_there_quantifiers_2 | +3.51, +10.10, +17.78, +21.95, +21.30 |
| BLiMP | wh_questions_object_gap | -2.56, +21.07, +15.37, +15.48, +14.20 |
| BLiMP | principle_A_reconstruction | +11.38, +8.17, +3.62, +11.89, +10.96 |
| BLiMP | matrix_question_npi_licensor_present | -0.86, +3.12, +3.55, +3.44, +10.76 |
| BLiMP | only_npi_licensor_present | -11.00, +11.00, +0.57, +11.79, +14.29 |
| BLiMP | npi_present_1 | +9.24, -5.06, +10.34, +5.06, +8.80 |
| BLiMP | npi_present_2 | +5.47, -13.68, +8.21, +0.00, +1.75 |
| BLiMP | left_branch_island_echo_question | -1.27, -10.35, +2.01, +0.84, +0.95 |
| BLiMP | adjunct_island | +6.36, -9.91, -2.48, -0.75, +2.69 |
| BLiMP | wh_vs_that_with_gap | -1.52, -8.27, -5.22, -4.68, -4.24 |
| BLiMP | existential_there_object_raising | +0.00, -2.71, -5.42, -4.68, -3.69 |
| Supplement | subject_aux_inversion | -0.34, -3.90, -2.64, -2.64, -4.16 |
| Supplement | qa_congruence_tricky | +0.61, +7.27, -1.82, +4.24, +7.27 |
| Supplement | qa_congruence_easy | +0.00, +0.00, +9.38, +10.94, +6.25 |
| Supplement | hypernym | +0.95, -2.49, +0.71, -3.09, -1.07 |
| Supplement | turn_taking | +2.86, -1.07, +0.00, +0.36, +0.36 |
| EWoK | material-properties | +6.47, -9.41, -6.47, -10.59, -11.18 |
| EWoK | material-dynamics | -5.19, -4.03, +2.34, -1.82, +1.04 |
| EWoK | social-interactions | -9.52, -5.44, -9.86, -5.78, -8.50 |
| EWoK | social-relations | +2.26, -0.52, -1.74, -1.49, -1.49 |
| EWoK | spatial-relations | +0.82, +0.20, -4.90, -3.67, -5.31 |
| EWoK | quantitative-properties | +4.14, -6.05, -2.23, -2.87, -1.27 |
| EWoK | physical-dynamics | +12.50, +15.83, -1.67, +6.67, +7.50 |
| EWoK | physical-relations | +0.24, +0.00, +4.03, -1.59, +0.86 |
| EWoK | social-properties | +0.61, +1.52, -1.22, +4.57, +5.49 |
| Entity | regular_0_ops | +0.77, +8.51, -4.84, -1.74, -0.97 |
| Entity | regular_1_ops | +1.96, -3.91, -2.44, +0.98, -1.96 |
| Entity | regular_2_ops | +1.73, +6.67, -0.49, +2.72, +1.73 |
| Entity | regular_3_ops | +0.47, -0.24, +4.94, +7.06, +5.18 |
| Entity | regular_4_ops | +2.84, +8.76, +3.61, +3.61, +0.77 |
| Entity | regular_5_ops | -8.51, +0.00, +3.19, -7.45, -8.51 |
| Entity | move_contents_0_ops | +2.13, +5.43, -0.78, +0.78, +1.74 |
| Entity | move_contents_5_ops | -0.86, +12.07, +2.59, +4.31, +1.72 |
| Entity | ambiref_3_ops | -0.73, -0.49, -1.22, -0.24, -0.73 |
| Entity | ambiref_4_ops | -0.46, +3.46, +2.53, +2.53, +1.61 |
| Entity | ambiref_5_ops | -8.13, +4.07, +2.44, +0.81, -0.81 |
| COMPS | base | +0.46, +1.33, +0.47, -0.27, +0.21 |
| COMPS | wugs | -0.38, +1.17, +0.58, +0.94, +0.56 |
| COMPS | wugs_dist_before | -0.48, -1.67, -12.77, -9.99, -9.77 |
| COMPS | wugs_dist_in_between | -0.74, +2.09, +12.74, +10.03, +10.18 |
| GlobalPIQA | GlobalPIQA_parallel | +2.91, -1.94, -8.74, -1.94, -2.91 |
| GlobalPIQA | GlobalPIQA_nonparallel | +3.00, -1.00, +7.00, +7.00, +3.00 |

## Scientific reading

- The 100M endpoint keeps a large BLiMP rotation and small gains in several cheap columns, but the cheap surface is below the 80M high point and needs SuperGLUE+AoA ≈ 71.405 to reach Overall 41.8.
- The EWoK material/social/spatial relation losses persist at 100M even though material-dynamics itself is no longer the dominant loss; GlobalPIQA is neutral by net item count.
- Entity high-operation gains seen at 50M are not preserved as a strong 100M advantage; regular_5_ops is negative.

JSON: `experiments/archive/frontier_consolidation/data/scale1p75_family_trajectory_readout/scale1p75_family_trajectory_readout.json`
CSV: `experiments/archive/frontier_consolidation/data/scale1p75_family_trajectory_readout/selected_group_trajectory.csv`

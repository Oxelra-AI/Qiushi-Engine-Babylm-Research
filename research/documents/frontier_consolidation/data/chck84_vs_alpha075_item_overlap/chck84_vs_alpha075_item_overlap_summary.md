# chck84 vs alpha075 endpoint overlap synthesis `chck_84M` versus coherent86 alpha0.75 cheap-task overlap

CPU/file-only comparison of two existing legal endpoint functions. `chck_84M` is an ordinary checkpoint on the trained scale1.75 trajectory; coherent86 alpha0.75 is a frozen-anchor private-pathway interpolation. This analysis does not evaluate SuperGLUE, does not train, and does not submit.

## Aggregate cheap columns

| endpoint | cheap7 | cheap6 no GlobalPIQA | cheap5 no GlobalPIQA/Reading | EWoK+Entity | BLiMP+COMPS |
|---|---:|---:|---:|---:|---:|
| chck82 | 43.959634 | 45.023294 | 52.397953 | 39.184748 | 60.341230 |
| chck84 | 44.123626 | 45.124004 | 52.517804 | 39.324289 | 60.228378 |
| alpha0.75 | 44.181208 | 45.117558 | 52.508070 | 39.170918 | 60.281573 |

Deltas alpha0.75 minus chck84: cheap7 +0.057582, cheap6-no-GP -0.006446, cheap5-no-GP/Reading -0.009735, EWoK+Entity -0.153371, BLiMP+COMPS +0.053195.

## Per-column scores

| column | chck82 | chck84 | alpha0.75 | 84-82 | alpha-82 | alpha-84 |
|---|---:|---:|---:|---:|---:|---:|
| BLiMP | 68.4913 | 68.2512 | 68.5160 | -0.2401 | +0.0247 | +0.2648 |
| Supplement | 62.9378 | 63.4837 | 63.6354 | +0.5459 | +0.6976 | +0.1517 |
| EWoK | 50.0555 | 50.0735 | 50.0196 | +0.0180 | -0.0358 | -0.0539 |
| Entity | 28.3140 | 28.5751 | 28.3222 | +0.2611 | +0.0082 | -0.2529 |
| COMPS | 52.1912 | 52.2056 | 52.0472 | +0.0144 | -0.1440 | -0.1584 |
| GlobalPIQA | 37.5777 | 38.1214 | 38.5631 | +0.5437 | +0.9854 | +0.4417 |
| Reading | 8.1500 | 8.1550 | 8.1650 | +0.0050 | +0.0150 | +0.0100 |

## Raw item overlap relative to chck82

| column | gain84 | gain_alpha | shared gains | gain Jaccard | loss84 | loss_alpha | shared losses | alpha correct / 84 wrong | 84 correct / alpha wrong | net alpha-84 items |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BLiMP | 1015 | 565 | 311 | 0.2451 | 1154 | 545 | 319 | 1089 | 930 | 159 |
| Supplement | 51 | 27 | 8 | 0.1143 | 61 | 37 | 16 | 64 | 64 | 0 |
| EWoK | 237 | 117 | 70 | 0.2465 | 240 | 113 | 54 | 233 | 226 | 7 |
| Entity | 134 | 50 | 24 | 0.1500 | 123 | 54 | 19 | 130 | 145 | -15 |
| COMPS | 3001 | 1570 | 862 | 0.2324 | 2919 | 1668 | 869 | 2758 | 2938 | -180 |
| GlobalPIQA | 5 | 3 | 1 | 0.1429 | 4 | 1 | 0 | 6 | 5 | 1 |
| ALL | 4443 | 2332 | 1276 | 0.2320 | 4501 | 2418 | 1277 | 4280 | 4308 | -28 |

## Interpretation

The alpha0.75 endpoint has a small cheap7 advantage over chck84, but that advantage disappears when GlobalPIQA is removed. Relation/state columns favor the trained chck84 checkpoint rather than the interpolated private endpoint. Thus the two endpoint functions are not a strict dominance relation: alpha0.75 remains the numerically stronger local cheap7/Overall(AoA0) carrier before chck84 SuperGLUE is known, while chck84 is the cleaner ordinary-training endpoint and preserves more relation/state signal. This is endpoint evidence, not a new authorization for alpha tuning or eval-set-driven combination.

## Largest subtask differences, alpha0.75 minus chck84

### BLiMP
Alpha0.75 above chck84:
- only_npi_licensor_present: +5.442 points (n=882, net=48)
- principle_A_c_command: +3.277 points (n=946, net=31)
- only_npi_scope: +3.106 points (n=837, net=26)
- tough_vs_raising_1: +2.532 points (n=948, net=24)
- irregular_plural_subject_verb_agreement_2: +2.242 points (n=892, net=20)
Chck84 above alpha0.75:
- existential_there_quantifiers_2: -6.696 points (n=911, net=-61)
- left_branch_island_echo_question: -2.957 points (n=947, net=-28)
- wh_vs_that_with_gap_long_distance: -1.758 points (n=910, net=-16)
- principle_A_domain_2: -1.749 points (n=915, net=-16)
- animate_subject_trans: -1.733 points (n=923, net=-16)

### Supplement
Alpha0.75 above chck84:
- qa_congruence_easy: +1.562 points (n=64, net=1)
- hypernym: +0.594 points (n=842, net=5)
- subject_aux_inversion: -0.078 points (n=3867, net=-3)
- qa_congruence_tricky: -0.606 points (n=165, net=-1)
- turn_taking: -0.714 points (n=280, net=-2)
Chck84 above alpha0.75:
- turn_taking: -0.714 points (n=280, net=-2)
- qa_congruence_tricky: -0.606 points (n=165, net=-1)
- subject_aux_inversion: -0.078 points (n=3867, net=-3)
- hypernym: +0.594 points (n=842, net=5)
- qa_congruence_easy: +1.562 points (n=64, net=1)

### EWoK
Alpha0.75 above chck84:
- agent-properties: +0.995 points (n=2210, net=22)
- physical-dynamics: +0.833 points (n=120, net=1)
- material-properties: +0.588 points (n=170, net=1)
- social-properties: +0.305 points (n=328, net=1)
- social-relations: +0.194 points (n=1548, net=3)
Chck84 above alpha0.75:
- material-dynamics: -1.818 points (n=770, net=-14)
- quantitative-properties: -0.955 points (n=314, net=-3)
- spatial-relations: -0.612 points (n=490, net=-3)
- physical-relations: -0.122 points (n=818, net=-1)
- physical-interactions: +0.000 points (n=556, net=0)

### Entity
Alpha0.75 above chck84:
- regular_3_ops: +1.882 points (n=425, net=8)
- move_contents_1_ops: +0.915 points (n=437, net=4)
- move_contents_5_ops: +0.862 points (n=116, net=1)
- ambiref_1_ops: +0.701 points (n=428, net=3)
- ambiref_2_ops: +0.484 points (n=413, net=2)
Chck84 above alpha0.75:
- move_contents_3_ops: -2.709 points (n=406, net=-11)
- regular_2_ops: -1.481 points (n=405, net=-6)
- regular_5_ops: -1.064 points (n=94, net=-1)
- ambiref_5_ops: -0.813 points (n=123, net=-1)
- move_contents_0_ops: -0.775 points (n=516, net=-4)

### COMPS
Alpha0.75 above chck84:
- wugs: -0.022 points (n=13896, net=-3)
- wugs_dist_in_between: -0.094 points (n=13896, net=-13)
- wugs_dist_before: -0.259 points (n=13896, net=-36)
- base: -0.259 points (n=49340, net=-128)
Chck84 above alpha0.75:
- base: -0.259 points (n=49340, net=-128)
- wugs_dist_before: -0.259 points (n=13896, net=-36)
- wugs_dist_in_between: -0.094 points (n=13896, net=-13)
- wugs: -0.022 points (n=13896, net=-3)

### GlobalPIQA
Alpha0.75 above chck84:
- global_piqa_parallel: +3.883 points (n=103, net=4)
- global_piqa_nonparallel: -3.000 points (n=100, net=-3)
Chck84 above alpha0.75:
- global_piqa_nonparallel: -3.000 points (n=100, net=-3)
- global_piqa_parallel: +3.883 points (n=103, net=4)

## Files

- `experiments/archive/frontier_consolidation/data/chck84_vs_alpha075_item_overlap/classification_item_records.csv` (67472053 bytes, sha256 `2ff11bb2617aa0e7…`)
- `experiments/archive/frontier_consolidation/data/chck84_vs_alpha075_item_overlap/chck84_alpha075_disagreements.csv` (3377617 bytes, sha256 `8acf85f76db46e28…`)
- `experiments/archive/frontier_consolidation/data/chck84_vs_alpha075_item_overlap/subtask_overlap_and_movement.csv` (18094 bytes, sha256 `8fccb152582ba519…`)
- `experiments/archive/frontier_consolidation/data/chck84_vs_alpha075_item_overlap/subgroup_overlap_and_movement.csv` (7268 bytes, sha256 `26df4053af8fcdec…`)
- `experiments/archive/frontier_consolidation/data/chck84_vs_alpha075_item_overlap/fine_group_overlap_and_movement.csv` (29852 bytes, sha256 `9b502881b7875ada…`)
- JSON summary: `experiments/archive/frontier_consolidation/data/chck84_vs_alpha075_item_overlap/chck84_vs_alpha075_item_overlap_summary.json`

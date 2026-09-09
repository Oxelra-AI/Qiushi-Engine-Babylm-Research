# chck84 item movement synthesis `chck_84M` cheap-task item movement analysis
CPU/file-only analysis of already-produced official-compatible predictions for the scale1.75 seed43022 reference trajectory. It does not evaluate SuperGLUE and does not authorize any new training.
## Endpoint-level cheap columns
| column | score82 | score84 | score100 | Δ84-82 | Δ100-84 | raw gains 82→84 | raw losses 82→84 |
|---|---:|---:|---:|---:|---:|---:|---:|
| BLiMP | 68.4913 | 68.2512 | 68.6318 | -0.2401 | +0.3806 | 1015 | 1154 |
| Supplement | 62.9378 | 63.4837 | 62.8952 | +0.5459 | -0.5885 | 51 | 61 |
| EWoK | 50.0555 | 50.0735 | 49.0801 | +0.0180 | -0.9934 | 237 | 240 |
| Entity | 28.3140 | 28.5751 | 27.4645 | +0.2611 | -1.1106 | 134 | 123 |
| COMPS | 52.1912 | 52.2056 | 52.3039 | +0.0144 | +0.0983 | 3001 | 2919 |
| GlobalPIQA | 37.5777 | 38.1214 | 36.1068 | +0.5437 | -2.0146 | 5 | 4 |

Reading is regression-style; selected-grid score moved 8.150→8.155→8.320. The 84M cheap7 lift is therefore driven by classification columns rather than Reading.

## Main interpretation
`chck_84M` improves 5/6 classification cheap columns relative to `chck_82M` (COMPS, EWoK, Entity, GlobalPIQA, Supplement positive; BLiMP negative). This supports treating 84M as a real endpoint branch rather than a single GlobalPIQA/Reading accident, but it remains a single seed on the same trajectory and therefore cannot support the transferable residual-capacity story without the pending seed43122 grid.
Raw item churn remains large relative to the net movement, showing competence allocation rather than monotone accumulation: many 82M-correct decisions flip off even when the weighted column score improves.

## Largest subtask movements from 82M to 84M

### BLiMP
Positive subtask deltas:
- existential_there_quantifiers_2: +7.574 points (n=911, net=69)
- left_branch_island_echo_question: +3.801 points (n=947, net=36)
- wh_vs_that_with_gap_long_distance: +1.978 points (n=910, net=18)
- superlative_quantifiers_2: +1.826 points (n=986, net=18)
- distractor_agreement_relative_clause: +1.722 points (n=871, net=15)
Negative subtask deltas:
- npi_present_2: -5.689 points (n=914, net=-52)
- only_npi_licensor_present: -5.102 points (n=882, net=-45)
- npi_present_1: -4.180 points (n=909, net=-38)
- irregular_plural_subject_verb_agreement_2: -2.803 points (n=892, net=-25)
- only_npi_scope: -2.748 points (n=837, net=-23)

### Supplement
Positive subtask deltas:
- qa_congruence_easy: +1.562 points (n=64, net=1)
- qa_congruence_tricky: +1.212 points (n=165, net=2)
- turn_taking: +0.714 points (n=280, net=2)
- subject_aux_inversion: -0.284 points (n=3867, net=-11)
- hypernym: -0.475 points (n=842, net=-4)
Negative subtask deltas:
- hypernym: -0.475 points (n=842, net=-4)
- subject_aux_inversion: -0.284 points (n=3867, net=-11)
- turn_taking: +0.714 points (n=280, net=2)
- qa_congruence_tricky: +1.212 points (n=165, net=2)
- qa_congruence_easy: +1.562 points (n=64, net=1)

### EWoK
Positive subtask deltas:
- material-dynamics: +1.688 points (n=770, net=13)
- social-interactions: +0.680 points (n=294, net=2)
- quantitative-properties: +0.637 points (n=314, net=2)
- material-properties: +0.588 points (n=170, net=1)
- social-relations: +0.388 points (n=1548, net=6)
Negative subtask deltas:
- physical-dynamics: -1.667 points (n=120, net=-2)
- social-properties: -1.524 points (n=328, net=-5)
- agent-properties: -1.041 points (n=2210, net=-23)
- physical-interactions: +0.000 points (n=556, net=0)
- spatial-relations: +0.204 points (n=490, net=1)

### Entity
Positive subtask deltas:
- regular_5_ops: +2.128 points (n=94, net=2)
- move_contents_3_ops: +1.478 points (n=406, net=6)
- move_contents_2_ops: +1.253 points (n=399, net=5)
- move_contents_0_ops: +1.163 points (n=516, net=6)
- move_contents_4_ops: +1.133 points (n=353, net=4)
Negative subtask deltas:
- regular_3_ops: -1.176 points (n=425, net=-5)
- ambiref_3_ops: -0.978 points (n=409, net=-4)
- ambiref_1_ops: -0.935 points (n=428, net=-4)
- ambiref_0_ops: -0.787 points (n=508, net=-4)
- move_contents_1_ops: -0.686 points (n=437, net=-3)

### COMPS
Positive subtask deltas:
- wugs_dist_in_between: +0.259 points (n=13896, net=36)
- base: +0.209 points (n=49340, net=103)
- wugs: -0.137 points (n=13896, net=-19)
- wugs_dist_before: -0.273 points (n=13896, net=-38)
Negative subtask deltas:
- wugs_dist_before: -0.273 points (n=13896, net=-38)
- wugs: -0.137 points (n=13896, net=-19)
- base: +0.209 points (n=49340, net=103)
- wugs_dist_in_between: +0.259 points (n=13896, net=36)

### GlobalPIQA
Positive subtask deltas:
- global_piqa_nonparallel: +4.000 points (n=100, net=4)
- global_piqa_parallel: -2.913 points (n=103, net=-3)
Negative subtask deltas:
- global_piqa_parallel: -2.913 points (n=103, net=-3)
- global_piqa_nonparallel: +4.000 points (n=100, net=4)

## Files
- `experiments/archive/frontier_consolidation/data/chck84_item_movement_analysis/classification_item_records.csv` (67467031 bytes, sha256 `c82a255c4b4df5b3…`)
- `experiments/archive/frontier_consolidation/data/chck84_item_movement_analysis/classification_item_flips.csv` (5667417 bytes, sha256 `52fe700ac004d603…`)
- `experiments/archive/frontier_consolidation/data/chck84_item_movement_analysis/subtask_movement.csv` (17194 bytes, sha256 `862e280ca4f2bf3b…`)
- `experiments/archive/frontier_consolidation/data/chck84_item_movement_analysis/subgroup_movement.csv` (6774 bytes, sha256 `e30dd0719a782e87…`)
- `experiments/archive/frontier_consolidation/data/chck84_item_movement_analysis/fine_group_movement.csv` (28859 bytes, sha256 `d187f23e13144fa4…`)
- `experiments/archive/frontier_consolidation/data/chck84_item_movement_analysis/reading_token_deltas_top200.csv` (49762 bytes, sha256 `b4807c732d08c4bb…`)
- `experiments/archive/frontier_consolidation/data/chck84_item_movement_analysis/reading_token_deltas_all.csv` (472192 bytes, sha256 `0410298e79d8b74f…`)
- JSON summary: `experiments/archive/frontier_consolidation/data/chck84_item_movement_analysis/chck84_item_movement_summary.json`

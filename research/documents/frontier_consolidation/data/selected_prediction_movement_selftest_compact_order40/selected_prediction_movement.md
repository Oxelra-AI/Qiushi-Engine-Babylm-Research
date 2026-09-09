# earlier analysis selected-prediction movement: ordered minus scrambled

Created: `2026-09-02T03:19:26Z`

This is a CPU/file-only reading of existing predictions; it does not train, evaluate, upload, or submit. Item accuracies are direct prediction-vs-gold correctness; official scores are quoted separately because several BabyLM columns average by subtask/UID rather than by raw item count.

## Inputs
- LEFT `scrambled`: `experiments/archive/frontier_consolidation/data/compact_order_selected_eval/scrambled/chck_40M/eval/per_target/compact_order_scrambled_chck_40M.json`
- RIGHT `ordered`: `experiments/archive/frontier_consolidation/data/compact_order_selected_eval/ordered/chck_40M/eval/per_target/compact_order_ordered_chck_40M.json`

## Column movement
| column | n | official LEFT | official RIGHT | official Δ | item LEFT | item RIGHT | item Δ | net | flips | Jaccard | pred agree |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BLiMP | 59875 | 63.140 | 62.470 | -0.670 | 63.038 | 62.340 | -0.698 | -418 | 10642 | 0.7517 | 82.23 |
| Supplement | 5218 | 54.880 | 53.840 | -1.040 | 72.001 | 70.889 | -1.112 | -58 | 790 | 0.8084 | 84.86 |
| EWoK | 7618 | 50.040 | 49.510 | -0.530 | 49.068 | 49.514 | 0.446 | 34 | 2184 | 0.5494 | 71.33 |
| Entity | 6780 | 22.700 | 23.190 | 0.490 | 22.507 | 22.581 | 0.074 | 5 | 1235 | 0.4245 | 57.17 |
| COMPS | 91028 | 50.880 | 51.140 | 0.260 | 51.540 | 51.810 | 0.270 | 246 | 35188 | 0.4556 | 61.34 |
| GlobalPIQA_parallel | 103 | 28.160 | 25.240 | -2.920 | 28.155 | 25.243 | -2.913 | -3 | 17 | 0.5278 | 66.02 |
| GlobalPIQA_nonparallel | 100 | 53.000 | 50.000 | -3.000 | 53.000 | 50.000 | -3.000 | -3 | 23 | 0.6349 | 77.00 |
| GlobalPIQA | 203 | 40.580 | 37.620 | -2.960 | 40.394 | 37.438 | -2.956 | -6 | 40 | 0.5960 | 71.43 |
| stable_five_item_pool | 170519 | NA | NA | NA | 54.939 | 54.827 | -0.112 | -191 | 50039 | 0.5781 | 69.68 |

## Largest subtask movements (item accuracy, RIGHT minus LEFT)
### Worst for RIGHT
- BLiMP / npi_present_1: n=909, Δ=-16.722, net=-152, flips=206, agree=77.34%
- BLiMP / superlative_quantifiers_1: n=979, Δ=-16.139, net=-158, flips=312, agree=68.13%
- BLiMP / principle_A_reconstruction: n=967, Δ=-16.029, net=-155, flips=221, agree=77.15%
- BLiMP / npi_present_2: n=914, Δ=-14.004, net=-128, flips=216, agree=76.37%
- BLiMP / coordinate_structure_constraint_object_extraction: n=949, Δ=-12.645, net=-120, flips=222, agree=76.61%
- EWoK / material-properties: n=170, Δ=-11.765, net=-20, flips=68, agree=60.00%
- EWoK / physical-dynamics: n=120, Δ=-10.833, net=-13, flips=43, agree=64.17%
- BLiMP / existential_there_quantifiers_2: n=911, Δ=-9.440, net=-86, flips=222, agree=75.63%
- BLiMP / left_branch_island_simple_question: n=951, Δ=-9.043, net=-86, flips=250, agree=73.71%
- BLiMP / coordinate_structure_constraint_complex_left_branch: n=906, Δ=-6.623, net=-60, flips=216, agree=76.16%
- BLiMP / sentential_subject_island: n=961, Δ=-6.556, net=-63, flips=297, agree=69.09%
- BLiMP / principle_A_domain_1: n=914, Δ=-6.455, net=-59, flips=219, agree=76.04%

### Best for RIGHT
- BLiMP / superlative_quantifiers_2: n=986, Δ=25.254, net=249, flips=367, agree=62.78%
- EWoK / social-properties: n=328, Δ=15.549, net=51, flips=141, agree=57.01%
- BLiMP / principle_A_case_2: n=915, Δ=11.803, net=108, flips=166, agree=81.86%
- Entity / regular_5_ops: n=94, Δ=8.511, net=8, flips=22, agree=53.19%
- BLiMP / regular_plural_subject_verb_agreement_1: n=890, Δ=7.303, net=65, flips=173, agree=80.56%
- BLiMP / only_npi_scope: n=837, Δ=7.288, net=61, flips=227, agree=72.88%
- BLiMP / complex_NP_island: n=846, Δ=6.619, net=56, flips=212, agree=74.94%
- BLiMP / matrix_question_npi_licensor_present: n=929, Δ=5.597, net=52, flips=170, agree=81.70%
- BLiMP / wh_vs_that_no_gap: n=861, Δ=5.459, net=47, flips=123, agree=85.71%
- BLiMP / wh_questions_subject_gap: n=898, Δ=5.345, net=48, flips=190, agree=78.84%
- Entity / ambiref_0_ops: n=508, Δ=5.315, net=27, flips=89, agree=61.81%
- BLiMP / left_branch_island_echo_question: n=947, Δ=5.280, net=50, flips=178, agree=81.20%

## Coverage
- common items `170722`, left-only `0`, right-only `0`.
- loader warnings `0`; skipped columns: left {'Reading': 'continuous reading correlations, not item correctness flips'}, right {'Reading': 'continuous reading correlations, not item correctness flips'}.

## Files
- JSON: `experiments/archive/frontier_consolidation/data/selected_prediction_movement_selftest_compact_order40/selected_prediction_movement.json`
- per-item CSV: `experiments/archive/frontier_consolidation/data/selected_prediction_movement_selftest_compact_order40/per_item_movement.csv`

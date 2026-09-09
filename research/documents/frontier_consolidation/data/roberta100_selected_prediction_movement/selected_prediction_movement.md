# earlier analysis selected-prediction movement: compact minus repeat

Created: `2026-09-02T04:41:24Z`

This is a CPU/file-only reading of existing predictions; it does not train, evaluate, upload, or submit. Item accuracies are direct prediction-vs-gold correctness; official scores are quoted separately because several BabyLM columns average by subtask/UID rather than by raw item count.

## Inputs
- LEFT `repeat`: `experiments/archive/frontier_consolidation/data/roberta_minimal_selected_eval/repeat/chck_100M/eval/per_target/roberta_repeat_chck_100M.json`
- RIGHT `compact`: `experiments/archive/frontier_consolidation/data/roberta_minimal_selected_eval/compact/chck_100M/eval/per_target/roberta_compact_chck_100M.json`

## Column movement
| column | n | official LEFT | official RIGHT | official Δ | item LEFT | item RIGHT | item Δ | net | flips | Jaccard | pred agree |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BLiMP | 59875 | 55.600 | 56.090 | 0.490 | 55.597 | 56.033 | 0.436 | 261 | 9347 | 0.7546 | 84.39 |
| Supplement | 5218 | 53.540 | 51.330 | -2.210 | 63.204 | 63.377 | 0.172 | 9 | 809 | 0.7818 | 84.50 |
| EWoK | 7618 | 50.740 | 51.790 | 1.050 | 50.643 | 51.181 | 0.538 | 41 | 2585 | 0.5001 | 66.07 |
| Entity | 6780 | 20.050 | 19.640 | -0.410 | 18.864 | 18.702 | -0.162 | -11 | 847 | 0.5009 | 68.10 |
| COMPS | 91028 | 51.620 | 51.730 | 0.110 | 52.311 | 52.297 | -0.014 | -13 | 26457 | 0.5651 | 70.94 |
| GlobalPIQA_parallel | 103 | 23.300 | 24.270 | 0.970 | 23.301 | 24.272 | 0.971 | 1 | 15 | 0.5312 | 75.73 |
| GlobalPIQA_nonparallel | 100 | 46.000 | 47.000 | 1.000 | 46.000 | 47.000 | 1.000 | 1 | 13 | 0.7547 | 87.00 |
| GlobalPIQA | 203 | 34.650 | 35.635 | 0.985 | 34.483 | 35.468 | 0.985 | 2 | 28 | 0.6706 | 81.28 |
| stable_five_item_pool | 170519 | NA | NA | NA | 52.394 | 52.562 | 0.168 | 287 | 40045 | 0.6343 | 75.74 |

## Largest subtask movements (item accuracy, RIGHT minus LEFT)
### Worst for RIGHT
- BLiMP / superlative_quantifiers_2: n=986, Δ=-16.531, net=-163, flips=237, agree=75.96%
- BLiMP / adjunct_island: n=928, Δ=-14.547, net=-135, flips=243, agree=73.81%
- BLiMP / only_npi_licensor_present: n=882, Δ=-11.224, net=-99, flips=169, agree=80.84%
- BLiMP / principle_A_domain_1: n=914, Δ=-8.753, net=-80, flips=136, agree=85.12%
- BLiMP / npi_present_1: n=909, Δ=-7.041, net=-64, flips=140, agree=84.60%
- BLiMP / distractor_agreement_relational_noun: n=788, Δ=-5.964, net=-47, flips=133, agree=83.12%
- BLiMP / distractor_agreement_relative_clause: n=871, Δ=-5.741, net=-50, flips=200, agree=77.04%
- BLiMP / principle_A_domain_2: n=915, Δ=-5.464, net=-50, flips=266, agree=70.93%
- BLiMP / existential_there_subject_raising: n=924, Δ=-5.303, net=-49, flips=125, agree=86.47%
- BLiMP / superlative_quantifiers_1: n=979, Δ=-4.903, net=-48, flips=100, agree=89.79%
- Supplement / qa_congruence_easy: n=64, Δ=-4.688, net=-3, flips=5, agree=92.19%
- Supplement / turn_taking: n=280, Δ=-4.643, net=-13, flips=17, agree=93.93%

### Best for RIGHT
- BLiMP / only_npi_scope: n=837, Δ=16.487, net=138, flips=196, agree=76.58%
- BLiMP / coordinate_structure_constraint_object_extraction: n=949, Δ=13.593, net=129, flips=243, agree=74.39%
- BLiMP / regular_plural_subject_verb_agreement_1: n=890, Δ=11.461, net=102, flips=180, agree=79.78%
- BLiMP / principle_A_c_command: n=946, Δ=8.985, net=85, flips=239, agree=74.74%
- BLiMP / determiner_noun_agreement_irregular_1: n=681, Δ=8.811, net=60, flips=146, agree=78.56%
- BLiMP / wh_island: n=960, Δ=8.646, net=83, flips=237, agree=75.31%
- BLiMP / determiner_noun_agreement_2: n=931, Δ=7.304, net=68, flips=172, agree=81.53%
- BLiMP / irregular_past_participle_adjectives: n=961, Δ=6.556, net=63, flips=197, agree=79.50%
- BLiMP / determiner_noun_agreement_with_adj_irregular_2: n=840, Δ=6.190, net=52, flips=144, agree=82.86%
- BLiMP / irregular_plural_subject_verb_agreement_1: n=804, Δ=5.597, net=45, flips=177, agree=77.99%
- BLiMP / determiner_noun_agreement_with_adj_irregular_1: n=718, Δ=5.571, net=40, flips=160, agree=77.72%
- BLiMP / determiner_noun_agreement_with_adj_2: n=941, Δ=4.995, net=47, flips=143, agree=84.80%

## Coverage
- common items `170722`, left-only `0`, right-only `0`.
- loader warnings `0`; skipped columns: left {'Reading': 'continuous reading correlations, not item correctness flips'}, right {'Reading': 'continuous reading correlations, not item correctness flips'}.

## Files
- JSON: `experiments/archive/frontier_consolidation/data/roberta100_selected_prediction_movement/selected_prediction_movement.json`
- per-item CSV: `experiments/archive/frontier_consolidation/data/roberta100_selected_prediction_movement/per_item_movement.csv`

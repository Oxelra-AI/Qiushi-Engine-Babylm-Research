# semantic repair frontier and refill plan — exact sparse DiD with fixed BLiMP union

The exact clean sparse temporal control has delivered. This file augments the prepared DiD with a fixed 18-subtask BLiMP union, because the named worst/control BLiMP groups were outcome-selected.

## BLiMP union trajectory
- 1M: reinvest gap -0.361; clean gap -1.333; excess +0.972
- 10M: reinvest gap +1.944; clean gap -5.833; excess +7.778
- 40M: reinvest gap -0.250; clean gap +1.639; excess -1.889
- 100M: reinvest gap -3.333; clean gap -1.278; excess -2.056

## Exact task-group excess summary
- blimp_control: excess_by_exposure={'1': -0.8125, '10': 5.1875, '40': -2.125, '100': 7.125}; mean=+2.344; final=+7.125
- blimp_worst: excess_by_exposure={'1': 2.3999999999999986, '10': 9.850000000000001, '40': -1.7000000000000028, '100': -9.399999999999999}; mean=+0.287; final=-9.400
- entity_full: excess_by_exposure={'1': 0.1357205904810037, '10': -2.505772550823787, '40': 1.4979622190831208, '100': -0.9581975887792709}; mean=-0.458; final=-0.958
- ewok_all: excess_by_exposure={'1': 0.36363636363636687, '10': -3.363636363636367, '40': -3.818181818181813, '100': -5.18181818181818}; mean=-3.000; final=-5.182
- supplement_all: excess_by_exposure={'1': -0.7999999999999972, '10': -2.3999999999999986, '40': 7.200000000000003, '100': -4.0}; mean=+0.000; final=-4.000
- blimp_union_18_subtasks: excess_by_exposure={'1': +0.972, '10': +7.778, '40': -1.889, '100': -2.056}; final=-2.056

## Late excess change, 100M - 40M
- supplement_all: -11.200
- blimp_worst: -7.700
- entity_full: -2.456
- ewok_all: -1.364
- blimp_union_18_subtasks: -0.167
- blimp_control: +9.250

## 100M subtask excess highlights
### supplement_all
Most negative: hypernym -10.00; qa_congruence_tricky -4.00; turn_taking -4.00; qa_congruence_easy -2.00; subject_aux_inversion +0.00
Most positive: subject_aux_inversion +0.00; qa_congruence_easy -2.00; qa_congruence_tricky -4.00; turn_taking -4.00; hypernym -10.00
### ewok_all
Most negative: spatial-relations -21.00; physical-relations -15.00; material-dynamics -13.00; social-properties -12.00; physical-interactions -10.00
Most positive: quantitative-properties +12.00; social-interactions +8.00; material-properties +7.00; social-relations +0.00; agent-properties -5.00
### entity_full
Most negative: ambiref_5_ops -7.32; regular_5_ops -6.38; regular_4_ops -4.90; move_contents_2_ops -3.51; regular_1_ops -3.42
Most positive: regular_0_ops +5.80; move_contents_0_ops +5.04; ambiref_0_ops +2.76; move_contents_5_ops +1.72; move_contents_4_ops +1.70
### blimp_worst
Most negative: superlative_quantifiers_1 -38.50; left_branch_island_simple_question -18.00; distractor_agreement_relative_clause -12.50; sentential_subject_island -11.00; principle_A_c_command -6.50
Most positive: matrix_question_npi_licensor_present +10.50; sentential_negation_npi_scope -3.00; wh_questions_object_gap -4.00; principle_A_case_2 -4.50; principle_A_c_command -6.50
### blimp_control
Most negative: only_npi_licensor_present -6.00; wh_island +0.00; wh_vs_that_with_gap +4.00; principle_A_domain_2 +7.50; superlative_quantifiers_2 +7.50
Most positive: left_branch_island_echo_question +21.00; principle_A_domain_1 +14.50; drop_argument +8.50; principle_A_domain_2 +7.50; superlative_quantifiers_2 +7.50

## Scientific read
- Fixed 18-subtask BLiMP union has modest final excess, so selected worst/control groups should not drive the route.
- Supplement excess is nonmonotonic and reverses sharply between 40M and 100M, pointing to late dynamics or repeated-exposure interaction rather than a simple early acquisition defect.
- Fast EWoK excess is already negative by 10M and becomes more negative by 100M; current-official full EWoK confirms positive but much smaller seed43122 treatment effect.

Machine-readable output: `experiments/archive/frontier_consolidation/data/exact_sparse_did_augmented/exact_sparse_did_augmented.json`

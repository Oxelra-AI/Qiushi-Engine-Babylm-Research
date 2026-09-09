# semantic repair frontier and refill plan — sparse temporal treatment effects

Treatment effect is compact_view_reinvest minus clean-Qwen within the same seed. DiD is TE43122 - TE43022.

## blimp_control
- final TE43022 -1.188; final TE43122 +5.938; final DiD +7.125
- late change TE43022 -6.625; TE43122 +2.625; DiD +9.250
- TE by exposure: 1M: seed43022 -2.62, seed43122 -3.44, DiD -0.81; 10M: seed43022 -3.62, seed43122 +1.56, DiD +5.19; 40M: seed43022 +5.44, seed43122 +3.31, DiD -2.12; 100M: seed43022 -1.19, seed43122 +5.94, DiD +7.12

## blimp_worst
- final TE43022 +1.750; final TE43122 -7.650; final DiD -9.400
- late change TE43022 +0.800; TE43122 -6.900; DiD -7.700
- TE by exposure: 1M: seed43022 -0.55, seed43122 +1.85, DiD +2.40; 10M: seed43022 -2.60, seed43122 +7.25, DiD +9.85; 40M: seed43022 +0.95, seed43122 -0.75, DiD -1.70; 100M: seed43022 +1.75, seed43122 -7.65, DiD -9.40

## entity_full
- final TE43022 +1.985; final TE43122 +1.027; final DiD -0.958
- late change TE43022 +2.633; TE43122 +0.177; DiD -2.456
- TE by exposure: 1M: seed43022 +0.24, seed43122 +0.37, DiD +0.14; 10M: seed43022 +1.51, seed43122 -1.00, DiD -2.51; 40M: seed43022 -0.65, seed43122 +0.85, DiD +1.50; 100M: seed43022 +1.99, seed43122 +1.03, DiD -0.96

## ewok_all
- final TE43022 +4.364; final TE43122 -0.818; final DiD -5.182
- late change TE43022 +0.727; TE43122 -0.636; DiD -1.364
- TE by exposure: 1M: seed43022 -0.18, seed43122 +0.18, DiD +0.36; 10M: seed43022 +1.09, seed43122 -2.27, DiD -3.36; 40M: seed43022 +3.64, seed43122 -0.18, DiD -3.82; 100M: seed43022 +4.36, seed43122 -0.82, DiD -5.18

## supplement_all
- final TE43022 +2.000; final TE43122 -2.000; final DiD -4.000
- late change TE43022 +6.000; TE43122 -5.200; DiD -11.200
- TE by exposure: 1M: seed43022 +0.40, seed43122 -0.40, DiD -0.80; 10M: seed43022 +2.00, seed43122 -0.40, DiD -2.40; 40M: seed43022 -4.00, seed43122 +3.20, DiD +7.20; 100M: seed43022 +2.00, seed43122 -2.00, DiD -4.00

## blimp_union_18_subtasks
- final TE43022 +0.444; final TE43122 -1.611; final DiD -2.056
- late change TE43022 -2.500; TE43122 -2.667; DiD -0.167
- TE by exposure: 1M: seed43022 -1.47, seed43122 -0.50, DiD +0.97; 10M: seed43022 -3.06, seed43122 +4.72, DiD +7.78; 40M: seed43022 +2.94, seed43122 +1.06, DiD -1.89; 100M: seed43022 +0.44, seed43122 -1.61, DiD -2.06

## 100M subtask interaction highlights
### supplement_all
Most negative: hypernym -10.00; qa_congruence_tricky -4.00; turn_taking -4.00; qa_congruence_easy -2.00; subject_aux_inversion +0.00
Most positive: subject_aux_inversion +0.00; qa_congruence_easy -2.00; qa_congruence_tricky -4.00; turn_taking -4.00; hypernym -10.00
### ewok_all
Most negative: spatial-relations -21.00; physical-relations -15.00; material-dynamics -13.00; social-properties -12.00; physical-interactions -10.00; physical-dynamics -8.00
Most positive: quantitative-properties +12.00; social-interactions +8.00; material-properties +7.00; social-relations +0.00; agent-properties -5.00; physical-dynamics -8.00
### entity_full
Most negative: ambiref_5_ops -7.32; regular_5_ops -6.38; regular_4_ops -4.90; move_contents_2_ops -3.51; regular_1_ops -3.42; regular_2_ops -2.96
Most positive: regular_0_ops +5.80; move_contents_0_ops +5.04; ambiref_0_ops +2.76; move_contents_5_ops +1.72; move_contents_4_ops +1.70; regular_3_ops +0.24
### blimp_union_18_subtasks
Most negative: superlative_quantifiers_1 -38.50; left_branch_island_simple_question -18.00; distractor_agreement_relative_clause -12.50; sentential_subject_island -11.00; principle_A_c_command -6.50; principle_A_reconstruction -6.50
Most positive: left_branch_island_echo_question +21.00; principle_A_domain_1 +14.50; matrix_question_npi_licensor_present +10.50; drop_argument +8.50; principle_A_domain_2 +7.50; superlative_quantifiers_2 +7.50

## Scientific read
- Final Supplement DiD is negative after a positive 40M DiD, consistent with a late treatment-by-seed reversal rather than only early acquisition failure.
- Fast EWoK DiD is persistently negative after 10M, while current-official full EWoK confirms positive but small seed43122 treatment effect; EWoK remains the strongest semantic/domain instability signal.
- The fixed BLiMP union has much smaller final and late DiD than the selected groups, so broad BLiMP repair is not the main target.

Machine-readable output: `experiments/archive/frontier_consolidation/data/temporal_treatment_effects/temporal_treatment_effects.json`

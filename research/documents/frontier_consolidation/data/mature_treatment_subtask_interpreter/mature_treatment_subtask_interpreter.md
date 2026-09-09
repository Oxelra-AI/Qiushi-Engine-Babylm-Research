# legal deficit representation vs candidate alignment mature legal-tokenizer treatment subtask interpreter

cheap official-compatible zero-shot/reading columns plus BLiMP/Supplement/EWoK UID reports; no SuperGLUE or AoA; clean-Qwen uses fixed reinvest tokenizer and is a scientific control only

## Payload status
- `reinvest_20M` exists=True: `experiments/archive/frontier_consolidation/data/legal20m_treatment_effect_eval/per_target/complianttok_reinvest_seed43022_20M.json`
- `clean_20M` exists=True: `experiments/archive/frontier_consolidation/data/legal_mature_clean_control_eval/per_target/complianttok_cleanqwen_seed43022_20M.json`
- `reinvest_70M` exists=True: `experiments/archive/frontier_consolidation/data/legal_mature_treatment_effect_eval/per_target/complianttok_reinvest_seed43022_70M.json`
- `clean_70M` exists=True: `experiments/archive/frontier_consolidation/data/legal_mature_clean_control_eval/per_target/complianttok_cleanqwen_seed43022_70M.json`
- `reinvest_80M` exists=True: `experiments/archive/frontier_consolidation/data/legal_mature_treatment_effect_eval/per_target/complianttok_reinvest_seed43022_80M.json`
- `clean_80M` exists=True: `experiments/archive/frontier_consolidation/data/legal_mature_clean_control_eval/per_target/complianttok_cleanqwen_seed43022_80M.json`

## Column deltas
| exposure | complete | reinvest mean7 | clean mean7 | Δ mean7 | BLiMP | Supp | EWoK | Entity | COMPS | GPIQA | Reading |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | yes | 39.6636 | 40.4914 | -0.8279 | 0.9500 | -1.7600 | 1.5900 | -0.9400 | -0.7400 | -4.9400 | 0.0450 |
| 70 | yes | 42.6086 | 41.3164 | 1.2921 | 0.9800 | 1.0900 | 1.5300 | 2.3900 | 0.6900 | 0.9300 | 1.4350 |
| 80 | yes | 42.9486 | 41.6021 | 1.3464 | 1.5300 | 2.0600 | 2.0300 | 2.3300 | 0.6500 | -0.0250 | 0.8500 |

## Mature route-reading status
Mature 70M/80M subtask pairs are present; read column deltas together with UID deltas and the static-prior alignment.

## 20M UID summaries
### BLiMP
- n_uid: 67; mean Δ reinvest-clean: 0.9469
- Pearson Δ vs relation_only pressure: 0.054026; Spearman: 0.108189
- largest reinvest losses: superlative_quantifiers_2 (-15.11), irregular_past_participle_adjectives (-11.97), anaphor_gender_agreement (-11.53), sentential_negation_npi_scope (-10.11), npi_present_2 (-9.41), anaphor_number_agreement (-8.91), principle_A_domain_1 (-7.88), principle_A_domain_2 (-6.56)
- largest reinvest gains: superlative_quantifiers_1 (39.42), only_npi_licensor_present (32.76), wh_island (20.41), existential_there_quantifiers_2 (13.4), principle_A_c_command (12.58), ellipsis_n_bar_1 (8.86), complex_NP_island (8.63), wh_questions_object_gap (7.1)
### Supplement
- n_uid: 5; mean Δ reinvest-clean: -1.756
- Pearson Δ vs relation_only pressure: 0.002203; Spearman: -0.2
- largest reinvest losses: qa_congruence_tricky (-4.24), turn_taking (-1.79), subject_aux_inversion (-1.66), qa_congruence_easy (-1.57), hypernym (0.48)
- largest reinvest gains: hypernym (0.48), qa_congruence_easy (-1.57), subject_aux_inversion (-1.66), turn_taking (-1.79), qa_congruence_tricky (-4.24)
### EWoK
- n_uid: 11; mean Δ reinvest-clean: 1.5909
- EWoK relation mean Δ: 2.0571; property mean Δ: 0.775
- Pearson Δ vs relation_only pressure: -0.301047; Spearman: -0.445455
- largest reinvest losses: physical-interactions (-4.14), material-properties (-2.94), social-relations (-2.9), agent-properties (-1.13), spatial-relations (-0.61), physical-relations (2.69), physical-dynamics (3.34), social-properties (3.35)
- largest reinvest gains: social-interactions (11.22), material-dynamics (4.8), quantitative-properties (3.82), social-properties (3.35), physical-dynamics (3.34), physical-relations (2.69), spatial-relations (-0.61), agent-properties (-1.13)

## 70M UID summaries
### BLiMP
- n_uid: 67; mean Δ reinvest-clean: 0.9799
- Pearson Δ vs relation_only pressure: 0.105707; Spearman: 0.338797
- largest reinvest losses: anaphor_gender_agreement (-16.68), tough_vs_raising_1 (-11.29), regular_plural_subject_verb_agreement_1 (-8.54), animate_subject_trans (-6.93), principle_A_reconstruction (-5.69), existential_there_quantifiers_2 (-5.37), wh_vs_that_no_gap_long_distance (-4.69), anaphor_number_agreement (-4.4)
- largest reinvest gains: wh_island (23.86), wh_vs_that_with_gap (14.59), superlative_quantifiers_1 (13.79), superlative_quantifiers_2 (10.75), adjunct_island (9.59), coordinate_structure_constraint_complex_left_branch (9.05), distractor_agreement_relational_noun (8.88), only_npi_scope (8.48)
### Supplement
- n_uid: 5; mean Δ reinvest-clean: 1.082
- Pearson Δ vs relation_only pressure: 0.600991; Spearman: 0.4
- largest reinvest losses: qa_congruence_easy (-1.56), hypernym (-1.42), turn_taking (2.15), qa_congruence_tricky (2.42), subject_aux_inversion (3.82)
- largest reinvest gains: subject_aux_inversion (3.82), qa_congruence_tricky (2.42), turn_taking (2.15), hypernym (-1.42), qa_congruence_easy (-1.56)
### EWoK
- n_uid: 11; mean Δ reinvest-clean: 1.5345
- EWoK relation mean Δ: 2.0657; property mean Δ: 0.605
- Pearson Δ vs relation_only pressure: -0.019293; Spearman: -0.036364
- largest reinvest losses: social-properties (-6.7), physical-relations (-5.86), physical-interactions (-2.34), agent-properties (1.45), social-relations (2.26), material-dynamics (3.38), material-properties (3.53), spatial-relations (4.08)
- largest reinvest gains: physical-dynamics (7.5), social-interactions (5.44), quantitative-properties (4.14), spatial-relations (4.08), material-properties (3.53), material-dynamics (3.38), social-relations (2.26), agent-properties (1.45)

## 80M UID summaries
### BLiMP
- n_uid: 67; mean Δ reinvest-clean: 1.5246
- Pearson Δ vs relation_only pressure: 0.055854; Spearman: 0.203871
- largest reinvest losses: anaphor_gender_agreement (-12.05), principle_A_reconstruction (-8.48), tough_vs_raising_1 (-6.96), animate_subject_trans (-6.61), left_branch_island_echo_question (-5.49), regular_plural_subject_verb_agreement_1 (-5.28), existential_there_quantifiers_2 (-4.83), only_npi_licensor_present (-4.42)
- largest reinvest gains: wh_island (16.15), ellipsis_n_bar_1 (13.84), distractor_agreement_relational_noun (12.95), superlative_quantifiers_1 (12.56), wh_vs_that_with_gap (10.99), left_branch_island_simple_question (10.3), existential_there_object_raising (9.24), superlative_quantifiers_2 (9.13)
### Supplement
- n_uid: 5; mean Δ reinvest-clean: 2.06
- Pearson Δ vs relation_only pressure: 0.514739; Spearman: 0.3
- largest reinvest losses: qa_congruence_easy (0.0), hypernym (1.07), qa_congruence_tricky (2.42), turn_taking (3.22), subject_aux_inversion (3.59)
- largest reinvest gains: subject_aux_inversion (3.59), turn_taking (3.22), qa_congruence_tricky (2.42), hypernym (1.07), qa_congruence_easy (0.0)
### EWoK
- n_uid: 11; mean Δ reinvest-clean: 2.0255
- EWoK relation mean Δ: 2.0057; property mean Δ: 2.06
- Pearson Δ vs relation_only pressure: -0.141437; Spearman: -0.190909
- largest reinvest losses: social-properties (-7.92), physical-relations (-5.13), physical-interactions (-1.98), agent-properties (0.27), material-dynamics (2.6), spatial-relations (2.66), social-relations (3.29), social-interactions (5.1)
- largest reinvest gains: material-properties (8.24), quantitative-properties (7.65), physical-dynamics (7.5), social-interactions (5.1), social-relations (3.29), spatial-relations (2.66), material-dynamics (2.6), agent-properties (0.27)

Full JSON: `experiments/archive/frontier_consolidation/data/mature_treatment_subtask_interpreter/mature_treatment_subtask_interpreter.json`

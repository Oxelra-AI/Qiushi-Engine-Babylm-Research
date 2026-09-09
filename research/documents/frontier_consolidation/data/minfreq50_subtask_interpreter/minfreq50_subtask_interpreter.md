# expensive work score thresholds minfreq50 subtask interpreter

minfreq50 init-matched support-floor vs spatial repair route status legal16k compact-view reinvest at cheap official-compatible columns and BLiMP/Supplement/EWoK UID layer; no SuperGLUE or AoA.

Read with experiments/archive/frontier_consolidation/data/supportfloor_substrate_alignment/supportfloor_substrate_alignment.json: support-floor is a broad representation/support/segmentation package, not a targeted UID repair.

## Payload status
- `70M_minfreq50` exists=True: `experiments/archive/frontier_consolidation/data/minfreq50_initmatched_70_80M_eval/per_target/minfreq50_initmatched_seed43022_70M.json`
- `70M_step35_reinvest` exists=True: `experiments/archive/frontier_consolidation/data/legal_mature_treatment_effect_eval/per_target/complianttok_reinvest_seed43022_70M.json`
- `70M_clean_control` exists=True: `experiments/archive/frontier_consolidation/data/legal_mature_clean_control_eval/per_target/complianttok_cleanqwen_seed43022_70M.json`
- `80M_minfreq50` exists=True: `experiments/archive/frontier_consolidation/data/minfreq50_initmatched_70_80M_eval/per_target/minfreq50_initmatched_seed43022_80M.json`
- `80M_step35_reinvest` exists=True: `experiments/archive/frontier_consolidation/data/legal_mature_treatment_effect_eval/per_target/complianttok_reinvest_seed43022_80M.json`
- `80M_clean_control` exists=True: `experiments/archive/frontier_consolidation/data/legal_mature_clean_control_eval/per_target/complianttok_cleanqwen_seed43022_80M.json`

## Column deltas
| exposure | complete | minfreq50 mean7 | spatial repair route status mean7 | clean mean7 | Δ minfreq-spatial repair route status | Δ minfreq-clean | BLiMP | Supp | EWoK | Entity | COMPS | GPIQA | Reading |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 70 | yes | 42.5564 | 42.6086 | 41.3164 | -0.0521 | 1.2400 | 0.1100 | -0.6600 | -0.9500 | -0.4400 | 0.2000 | 1.5700 | -0.1950 |
| 80 | yes | 42.7807 | 42.9486 | 41.6021 | -0.1679 | 1.1786 | -0.4200 | -2.2900 | -1.4400 | 0.2500 | -0.0300 | 2.5400 | 0.2150 |

## UID summaries

### 70M
#### BLiMP
- n_uid: 67; mean Δ minfreq50-spatial repair route status: 0.112985
- Pearson Δ vs prior legal deficit magnitude: 0.388377; Spearman: 0.450645
- Pearson Δ vs minfreq50 token reduction: 0.154591; Spearman: 0.132658
- largest minfreq50 losses: only_npi_licensor_present (-28.12), wh_island (-18.86), wh_vs_that_with_gap (-11.76), tough_vs_raising_2 (-9.89), irregular_past_participle_verbs (-9.34), wh_vs_that_with_gap_long_distance (-8.24), superlative_quantifiers_1 (-7.45), distractor_agreement_relational_noun (-7.36)
- largest minfreq50 gains: matrix_question_npi_licensor_present (35.63), anaphor_gender_agreement (19.77), existential_there_quantifiers_2 (17.45), regular_plural_subject_verb_agreement_1 (11.35), animate_subject_trans (10.29), tough_vs_raising_1 (10.03), sentential_subject_island (8.95), anaphor_number_agreement (8.06)
#### Supplement
- n_uid: 5; mean Δ minfreq50-spatial repair route status: -0.654
- Pearson Δ vs prior legal deficit magnitude: 0.057473; Spearman: 0.447214
- Pearson Δ vs minfreq50 token reduction: -0.530841; Spearman: -0.3
- largest minfreq50 losses: subject_aux_inversion (-5.09), turn_taking (-1.79), qa_congruence_easy (-1.57), hypernym (1.54), qa_congruence_tricky (3.64)
- largest minfreq50 gains: qa_congruence_tricky (3.64), hypernym (1.54), qa_congruence_easy (-1.57), turn_taking (-1.79), subject_aux_inversion (-5.09)
#### EWoK
- n_uid: 11; mean Δ minfreq50-spatial repair route status: -0.950909
- EWoK relation/dynamics mean Δ: -1.177143; property mean Δ: -0.555
- Pearson Δ vs prior legal deficit magnitude: 0.640027; Spearman: 0.600681
- Pearson Δ vs minfreq50 token reduction: 0.389402; Spearman: 0.445455
- largest minfreq50 losses: material-dynamics (-5.2), spatial-relations (-4.89), material-properties (-3.53), agent-properties (-2.72), social-interactions (-1.7), social-relations (-1.35), physical-interactions (-0.36), physical-relations (1.1)
- largest minfreq50 gains: physical-dynamics (4.16), social-properties (2.44), quantitative-properties (1.59), physical-relations (1.1), physical-interactions (-0.36), social-relations (-1.35), social-interactions (-1.7), agent-properties (-2.72)

### 80M
#### BLiMP
- n_uid: 67; mean Δ minfreq50-spatial repair route status: -0.412537
- Pearson Δ vs prior legal deficit magnitude: 0.374836; Spearman: 0.404494
- Pearson Δ vs minfreq50 token reduction: 0.212724; Spearman: 0.191723
- largest minfreq50 losses: only_npi_licensor_present (-25.4), ellipsis_n_bar_1 (-12.09), wh_island (-11.25), irregular_past_participle_verbs (-10.83), distractor_agreement_relational_noun (-10.03), superlative_quantifiers_1 (-9.19), principle_A_domain_1 (-6.78), wh_vs_that_with_gap (-6.64)
- largest minfreq50 gains: matrix_question_npi_licensor_present (30.89), existential_there_quantifiers_2 (13.94), anaphor_gender_agreement (12.26), animate_subject_trans (10.51), sentential_subject_island (8.63), regular_plural_subject_verb_agreement_1 (8.31), causative (5.87), tough_vs_raising_1 (4.85)
#### Supplement
- n_uid: 5; mean Δ minfreq50-spatial repair route status: -2.296
- Pearson Δ vs prior legal deficit magnitude: -0.139178; Spearman: 0.111803
- Pearson Δ vs minfreq50 token reduction: -0.456235; Spearman: -0.5
- largest minfreq50 losses: subject_aux_inversion (-5.89), qa_congruence_easy (-4.69), turn_taking (-2.86), hypernym (-1.07), qa_congruence_tricky (3.03)
- largest minfreq50 gains: qa_congruence_tricky (3.03), hypernym (-1.07), turn_taking (-2.86), qa_congruence_easy (-4.69), subject_aux_inversion (-5.89)
#### EWoK
- n_uid: 11; mean Δ minfreq50-spatial repair route status: -1.437273
- EWoK relation/dynamics mean Δ: -0.49; property mean Δ: -3.095
- Pearson Δ vs prior legal deficit magnitude: 0.522012; Spearman: 0.276504
- Pearson Δ vs minfreq50 token reduction: 0.24174; Spearman: 0.1
- largest minfreq50 losses: material-properties (-6.48), spatial-relations (-4.09), quantitative-properties (-3.83), material-dynamics (-2.99), agent-properties (-1.76), physical-interactions (-1.62), social-properties (-0.31), social-relations (-0.06)
- largest minfreq50 gains: physical-dynamics (4.16), social-interactions (0.68), physical-relations (0.49), social-relations (-0.06), social-properties (-0.31), physical-interactions (-1.62), agent-properties (-1.76), material-dynamics (-2.99)

Full JSON: `experiments/archive/frontier_consolidation/data/minfreq50_subtask_interpreter/minfreq50_subtask_interpreter.json`

# legal deficit decomposition and prior mismatch legal-tokenizer deficit: UID-level decomposition

All endpoints share the identical 100M reinvest stream (SHA `3dd19f...`), architecture, seeds, and recipe. Only the tokenizer differs. `old_ref_42033` is the NON-SUBMITTABLE inherited-tokenizer reference (Overall 42.0331).

## Column averages

| column | old_ref_42033 | legal_step35 | legal_bytealpha | legal_a01_ss16k |
|---|---:|---:|---:|---:|
| BLiMP | 66.87 | 65.87 | 66.33 | 66.33 |
| Supplement | 63.28 | 61.17 | 59.28 | 59.28 |
| EWoK | 53.67 | 50.39 | 49.84 | — |

## EWoK relation vs property (legal_step35 minus old_ref)

- mean relation-domain delta: **-3.1586**
- mean property-domain delta: **-3.495**

Relation-domain deltas:
  - physical-dynamics: -19.5
  - material-dynamics: -8.83
  - physical-interactions: -3.31
  - physical-relations: -1.83
  - social-relations: 2.52
  - spatial-relations: 3.06
  - social-interactions: 5.78

Property-domain deltas:
  - social-properties: -13.75
  - quantitative-properties: -5.8
  - agent-properties: 0.6
  - material-properties: 4.97

## Cross-tokenizer agreement (two independent legal 16k tokenizers)

If the two independently built legal 16k tokenizers (spatial repair route status vs ss16k) lose on the SAME UID subtasks, the deficit is a representation property of the legal budget, not tokenizer noise.

### BLiMP
- n_uid compared: 67
- mean delta spatial repair route status: -0.9964, ss16k: -0.5381
- Pearson(UID deltas): **0.6526**, sign agreement: **0.597**

### Supplement
- n_uid compared: 5
- mean delta spatial repair route status: -2.114, ss16k: -3.998
- Pearson(UID deltas): **0.8235**, sign agreement: **1.0**

## Biggest legal_step35 UID losses/gains vs old_ref

### BLiMP
Losses: principle_A_reconstruction (-20.89), wh_questions_object_gap (-17.34), animate_subject_trans (-11.91), regular_plural_subject_verb_agreement_1 (-11.13), tough_vs_raising_1 (-10.97), anaphor_gender_agreement (-10.71), matrix_question_npi_licensor_present (-9.47), anaphor_number_agreement (-7.41)
Gains: only_npi_licensor_present (24.26), left_branch_island_echo_question (19.96), tough_vs_raising_2 (12.39), superlative_quantifiers_2 (9.44), ellipsis_n_bar_1 (7.48), adjunct_island (7.22), complex_NP_island (6.97), existential_there_subject_raising (5.3)

### Supplement
Losses: qa_congruence_easy (-12.5), qa_congruence_tricky (-3.64), hypernym (1.3), subject_aux_inversion (1.42), turn_taking (2.85)
Gains: turn_taking (2.85), subject_aux_inversion (1.42), hypernym (1.3), qa_congruence_tricky (-3.64), qa_congruence_easy (-12.5)

### EWoK
Losses: physical-dynamics (-19.5), social-properties (-13.75), material-dynamics (-8.83), quantitative-properties (-5.8), physical-interactions (-3.31), physical-relations (-1.83), agent-properties (0.6), social-relations (2.52)
Gains: social-interactions (5.78), material-properties (4.97), spatial-relations (3.06), social-relations (2.52), agent-properties (0.6), physical-relations (-1.83), physical-interactions (-3.31), quantitative-properties (-5.8)


Full JSON: `experiments/archive/frontier_consolidation/data/legal_deficit_subtask_decomposition/legal_deficit_subtask_decomposition.json`

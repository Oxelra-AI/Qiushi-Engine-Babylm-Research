# compact density subtask delta compact-density subtask deltas

This CPU-only pass parses finished fast no-AoA reports for `compact_repeat_core`, `compact_view_core`, and `compact_view_reinvest`. It does not read or modify the running compact reinvest full eval summary full-evaluation or seed43122 outputs.

## Fast component surface

| target | BLiMP | Supplement | EWoK | Entity | Entity_full | COMPS | GlobalPIQA_mean | Reading | equal7_mean | equal7_full_entity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| compact_repeat_core | 67.090 | 58.000 | 48.640 | 25.110 | 25.420 | 51.090 | 34.105 | 8.020 | 41.722 | 41.766 |
| compact_view_core | 66.930 | 65.600 | 51.550 | 27.300 | 27.850 | 52.180 | 35.135 | 8.250 | 43.849 | 43.928 |
| compact_view_reinvest | 66.630 | 66.400 | 53.090 | 28.070 | 27.750 | 51.970 | 35.620 | 8.240 | 44.289 | 44.243 |

## Component deltas

- `compact_view_core_minus_repeat_core`: {"BLiMP": -0.16, "Supplement": 7.6, "EWoK": 2.91, "Entity": 2.19, "Entity_full": 2.43, "COMPS": 1.09, "GlobalPIQA_parallel": -1.94, "GlobalPIQA_nonparallel": 4.0, "GlobalPIQA_mean": 1.03, "Reading": 0.23, "Reading_eye": 0.25, "Reading_self_paced": 0.21, "equal7_mean": 2.127143, "equal7_full_entity": 2.161429}
- `compact_view_reinvest_minus_view_core`: {"BLiMP": -0.3, "Supplement": 0.8, "EWoK": 1.54, "Entity": 0.77, "Entity_full": -0.1, "COMPS": -0.21, "GlobalPIQA_parallel": 0.97, "GlobalPIQA_nonparallel": 0.0, "GlobalPIQA_mean": 0.485, "Reading": -0.01, "Reading_eye": -0.35, "Reading_self_paced": 0.33, "equal7_mean": 0.439286, "equal7_full_entity": 0.315}
- `compact_view_reinvest_minus_repeat_core`: {"BLiMP": -0.46, "Supplement": 8.4, "EWoK": 4.45, "Entity": 2.96, "Entity_full": 2.33, "COMPS": 0.88, "GlobalPIQA_parallel": -0.97, "GlobalPIQA_nonparallel": 4.0, "GlobalPIQA_mean": 1.515, "Reading": 0.22, "Reading_eye": -0.1, "Reading_self_paced": 0.54, "equal7_mean": 2.566429, "equal7_full_entity": 2.476429}
- `compact_view_reinvest_minus_visible_leader_fast`: {"BLiMP": -0.57, "Supplement": 10.39, "EWoK": -2.98, "Entity": -0.38, "COMPS": -1.6, "GlobalPIQA_mean": -4.05, "Reading": 2.82}
- `compact_view_reinvest_minus_compact_experience_clean_fast`: {"BLiMP": -0.21, "Supplement": 3.56, "EWoK": 2.9, "Entity": 2.31, "COMPS": 0.19, "GlobalPIQA_mean": -1.0, "Reading": 0.48}

## SOTA arithmetic from the fast reinvest surface

{
  "fast_seven_sum": 310.02,
  "superglue_plus_aoa_needed_for_41p8": 66.18,
  "superglue_plus_aoa_needed_for_42p0": 67.98
}

## Largest subtask movements

### Core compact view minus same-source repeat

**BLiMP / LINGUISTICS_TERM ACCURACY**

Top positive:
- anaphor_agreement: 82.50 -> 89.25 (delta +6.75, n=None)
- island_effects: 47.94 -> 51.50 (delta +3.56, n=None)
- subject_verb_agreement: 60.67 -> 63.67 (delta +3.00, n=None)
- filler_gap_dependency: 66.79 -> 68.07 (delta +1.28, n=None)
- argument_structure: 63.36 -> 63.93 (delta +0.57, n=None)
Top negative:
- npi_licensing: 60.00 -> 54.14 (delta -5.86, n=None)
- quantifiers: 75.00 -> 70.62 (delta -4.38, n=None)
- irregular_forms: 88.00 -> 85.00 (delta -3.00, n=None)
- ellipsis: 86.00 -> 84.50 (delta -1.50, n=None)
- control_raising: 65.00 -> 63.70 (delta -1.30, n=None)

**BLiMP / UID ACCURACY**

Top positive:
- coordinate_structure_constraint_object_extraction: 73.50 -> 84.00 (delta +10.50, n=200)
- principle_A_reconstruction: 37.00 -> 45.50 (delta +8.50, n=200)
- sentential_negation_npi_scope: 70.00 -> 78.50 (delta +8.50, n=200)
- distractor_agreement_relational_noun: 36.50 -> 44.00 (delta +7.50, n=200)
- anaphor_gender_agreement: 81.00 -> 88.00 (delta +7.00, n=200)
Top negative:
- only_npi_licensor_present: 72.50 -> 28.00 (delta -44.50, n=200)
- principle_A_c_command: 54.00 -> 44.50 (delta -9.50, n=200)
- superlative_quantifiers_2: 71.50 -> 62.50 (delta -9.00, n=200)
- tough_vs_raising_2: 76.50 -> 69.50 (delta -7.00, n=200)
- irregular_past_participle_adjectives: 92.50 -> 87.00 (delta -5.50, n=200)

**Supplement / UID ACCURACY**

Top positive:
- subject_aux_inversion: 62.00 -> 86.00 (delta +24.00, n=50)
- turn_taking: 62.00 -> 74.00 (delta +12.00, n=50)
- hypernym: 48.00 -> 52.00 (delta +4.00, n=50)
- qa_congruence_tricky: 50.00 -> 50.00 (delta +0.00, n=50)
- qa_congruence_easy: 68.00 -> 66.00 (delta -2.00, n=50)
Top negative:
- qa_congruence_easy: 68.00 -> 66.00 (delta -2.00, n=50)
- qa_congruence_tricky: 50.00 -> 50.00 (delta +0.00, n=50)
- hypernym: 48.00 -> 52.00 (delta +4.00, n=50)
- turn_taking: 62.00 -> 74.00 (delta +12.00, n=50)
- subject_aux_inversion: 62.00 -> 86.00 (delta +24.00, n=50)

**EWoK / UID ACCURACY**

Top positive:
- social-interactions: 41.00 -> 49.00 (delta +8.00, n=100)
- physical-dynamics: 53.00 -> 60.00 (delta +7.00, n=100)
- physical-relations: 56.00 -> 61.00 (delta +5.00, n=100)
- social-properties: 47.00 -> 52.00 (delta +5.00, n=100)
- social-relations: 52.00 -> 56.00 (delta +4.00, n=100)
Top negative:
- quantitative-properties: 52.00 -> 49.00 (delta -3.00, n=100)
- spatial-relations: 39.00 -> 37.00 (delta -2.00, n=100)
- material-properties: 41.00 -> 42.00 (delta +1.00, n=100)
- physical-interactions: 57.00 -> 58.00 (delta +1.00, n=100)
- agent-properties: 50.00 -> 53.00 (delta +3.00, n=100)

**Entity_full / UID ACCURACY**

Top positive:
- move_contents_5_ops: 31.03 -> 39.66 (delta +8.63, n=116)
- move_contents_4_ops: 25.50 -> 33.71 (delta +8.21, n=353)
- ambiref_2_ops: 20.10 -> 27.12 (delta +7.02, n=413)
- regular_4_ops: 23.45 -> 29.64 (delta +6.19, n=388)
- regular_3_ops: 20.71 -> 26.35 (delta +5.64, n=425)
Top negative:
- move_contents_0_ops: 53.10 -> 46.90 (delta -6.20, n=516)
- regular_0_ops: 44.49 -> 40.81 (delta -3.68, n=517)
- ambiref_1_ops: 15.65 -> 13.79 (delta -1.86, n=428)
- ambiref_0_ops: 27.56 -> 25.79 (delta -1.77, n=508)
- move_contents_3_ops: 22.41 -> 22.17 (delta -0.24, n=406)

**COMPS / UID ACCURACY**

Top positive:
- wugs_dist_before: 40.65 -> 43.92 (delta +3.27, n=13896)
- wugs: 51.01 -> 52.73 (delta +1.72, n=13896)
- base: 53.00 -> 53.79 (delta +0.79, n=49340)
- wugs_dist_in_between: 59.68 -> 58.30 (delta -1.38, n=13896)
Top negative:
- wugs_dist_in_between: 59.68 -> 58.30 (delta -1.38, n=13896)
- base: 53.00 -> 53.79 (delta +0.79, n=49340)
- wugs: 51.01 -> 52.73 (delta +1.72, n=13896)
- wugs_dist_before: 40.65 -> 43.92 (delta +3.27, n=13896)

**GlobalPIQA_parallel UID movement counts**: {"count_items": 103, "positive_items": 9, "negative_items": 11, "unchanged_items": 83, "zero_to_hundred": 9, "hundred_to_zero": 11}
**GlobalPIQA_nonparallel UID movement counts**: {"count_items": 100, "positive_items": 13, "negative_items": 9, "unchanged_items": 78, "zero_to_hundred": 13, "hundred_to_zero": 9}
**Reading subscores**: [{"key": "Reading_eye", "target_value": 11.5, "baseline_value": 11.25, "delta": 0.25}, {"key": "Reading_self_paced", "target_value": 5.0, "baseline_value": 4.79, "delta": 0.21}, {"key": "Reading", "target_value": 8.25, "baseline_value": 8.02, "delta": 0.23}]

### Reinvest minus core view

**BLiMP / LINGUISTICS_TERM ACCURACY**

Top positive:
- s-selection: 66.25 -> 69.50 (delta +3.25, n=None)
- binding: 64.36 -> 67.00 (delta +2.64, n=None)
- filler_gap_dependency: 68.07 -> 69.36 (delta +1.29, n=None)
- determiner_noun_agreement: 85.69 -> 86.44 (delta +0.75, n=None)
- control_raising: 63.70 -> 64.40 (delta +0.70, n=None)
Top negative:
- island_effects: 51.50 -> 47.56 (delta -3.94, n=None)
- ellipsis: 84.50 -> 80.75 (delta -3.75, n=None)
- quantifiers: 70.62 -> 68.50 (delta -2.12, n=None)
- npi_licensing: 54.14 -> 52.86 (delta -1.28, n=None)
- argument_structure: 63.93 -> 62.71 (delta -1.22, n=None)

**BLiMP / UID ACCURACY**

Top positive:
- matrix_question_npi_licensor_present: 29.50 -> 40.50 (delta +11.00, n=200)
- principle_A_c_command: 44.50 -> 53.50 (delta +9.00, n=200)
- left_branch_island_simple_question: 52.50 -> 59.50 (delta +7.00, n=200)
- wh_island: 54.50 -> 61.50 (delta +7.00, n=200)
- principle_A_reconstruction: 45.50 -> 51.50 (delta +6.00, n=200)
Top negative:
- left_branch_island_echo_question: 38.50 -> 15.50 (delta -23.00, n=200)
- npi_present_2: 40.50 -> 31.00 (delta -9.50, n=200)
- sentential_subject_island: 43.50 -> 35.00 (delta -8.50, n=200)
- npi_present_1: 35.50 -> 28.00 (delta -7.50, n=200)
- adjunct_island: 76.50 -> 69.50 (delta -7.00, n=200)

**Supplement / UID ACCURACY**

Top positive:
- qa_congruence_easy: 66.00 -> 74.00 (delta +8.00, n=50)
- hypernym: 52.00 -> 56.00 (delta +4.00, n=50)
- qa_congruence_tricky: 50.00 -> 54.00 (delta +4.00, n=50)
- subject_aux_inversion: 86.00 -> 80.00 (delta -6.00, n=50)
- turn_taking: 74.00 -> 68.00 (delta -6.00, n=50)
Top negative:
- subject_aux_inversion: 86.00 -> 80.00 (delta -6.00, n=50)
- turn_taking: 74.00 -> 68.00 (delta -6.00, n=50)
- hypernym: 52.00 -> 56.00 (delta +4.00, n=50)
- qa_congruence_tricky: 50.00 -> 54.00 (delta +4.00, n=50)
- qa_congruence_easy: 66.00 -> 74.00 (delta +8.00, n=50)

**EWoK / UID ACCURACY**

Top positive:
- material-properties: 42.00 -> 53.00 (delta +11.00, n=100)
- spatial-relations: 37.00 -> 43.00 (delta +6.00, n=100)
- material-dynamics: 50.00 -> 55.00 (delta +5.00, n=100)
- social-interactions: 49.00 -> 52.00 (delta +3.00, n=100)
- physical-dynamics: 60.00 -> 62.00 (delta +2.00, n=100)
Top negative:
- social-relations: 56.00 -> 49.00 (delta -7.00, n=100)
- agent-properties: 53.00 -> 48.00 (delta -5.00, n=100)
- physical-interactions: 58.00 -> 57.00 (delta -1.00, n=100)
- quantitative-properties: 49.00 -> 49.00 (delta +0.00, n=100)
- physical-relations: 61.00 -> 62.00 (delta +1.00, n=100)

**Entity_full / UID ACCURACY**

Top positive:
- move_contents_3_ops: 22.17 -> 29.56 (delta +7.39, n=406)
- ambiref_5_ops: 29.27 -> 34.15 (delta +4.88, n=123)
- regular_5_ops: 27.66 -> 31.91 (delta +4.25, n=94)
- regular_1_ops: 16.14 -> 19.32 (delta +3.18, n=409)
- regular_3_ops: 26.35 -> 27.76 (delta +1.41, n=425)
Top negative:
- move_contents_0_ops: 46.90 -> 40.12 (delta -6.78, n=516)
- regular_0_ops: 40.81 -> 36.56 (delta -4.25, n=517)
- move_contents_4_ops: 33.71 -> 30.03 (delta -3.68, n=353)
- ambiref_2_ops: 27.12 -> 23.97 (delta -3.15, n=413)
- move_contents_5_ops: 39.66 -> 37.93 (delta -1.73, n=116)

**COMPS / UID ACCURACY**

Top positive:
- wugs_dist_in_between: 58.30 -> 59.77 (delta +1.47, n=13896)
- base: 53.79 -> 54.30 (delta +0.51, n=49340)
- wugs: 52.73 -> 52.17 (delta -0.56, n=13896)
- wugs_dist_before: 43.92 -> 41.64 (delta -2.28, n=13896)
Top negative:
- wugs_dist_before: 43.92 -> 41.64 (delta -2.28, n=13896)
- wugs: 52.73 -> 52.17 (delta -0.56, n=13896)
- base: 53.79 -> 54.30 (delta +0.51, n=49340)
- wugs_dist_in_between: 58.30 -> 59.77 (delta +1.47, n=13896)

**GlobalPIQA_parallel UID movement counts**: {"count_items": 103, "positive_items": 7, "negative_items": 6, "unchanged_items": 90, "zero_to_hundred": 7, "hundred_to_zero": 6}
**GlobalPIQA_nonparallel UID movement counts**: {"count_items": 100, "positive_items": 13, "negative_items": 13, "unchanged_items": 74, "zero_to_hundred": 13, "hundred_to_zero": 13}
**Reading subscores**: [{"key": "Reading_eye", "target_value": 11.15, "baseline_value": 11.5, "delta": -0.35}, {"key": "Reading_self_paced", "target_value": 5.33, "baseline_value": 5.0, "delta": 0.33}, {"key": "Reading", "target_value": 8.24, "baseline_value": 8.25, "delta": -0.01}]

## Mechanistic reading

- Core compact views improve the same-source repeat endpoint mainly through Supplement, EWoK, Entity, COMPS, GlobalPIQA_mean, and Reading while slightly reducing BLiMP; the fast evidence supports a real source-own-compact-view learning effect, not only endpoint noise, but it does not separate semantic correspondence from lexical-token geometry.
- Reinvestment turns the saved compact-view words into more source-view pairs and raises EWoK, Entity, Supplement, and GlobalPIQA_mean over compact_view_core while slightly reducing BLiMP, COMPS, and Reading; the future full result should be read as a trade between stronger lexical/world-relation exposure and possible syntactic/procedural dilution.
- The weak area remains GlobalPIQA: reinvest improves the fast mean over core but is still about four points under the public 41.8 leader on that column. Any future data repair should increase practical affordance and causal-procedural coverage without sacrificing the high Supplement/Entity/EWoK surface or SuperGLUE/AoA.

JSON: `experiments/archive/representation_and_objectives/data/compact_density_subtask_delta/compact_density_subtask_delta.json`

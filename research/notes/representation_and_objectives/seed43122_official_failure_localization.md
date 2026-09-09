# fast full seed43122 reconciliation — seed43122 official-coordinate failure localization

Seed43122 staged official Overall: `41.24823958912208`; margin over 41.8: `-0.5517604108779182`.
Seed43122 minus seed43022 Overall: `-0.7848952009527181`; NLP average delta: `-1.0982144510021001`.

Column deltas (seed43122 - seed43022) show broad weakening except Reading; AoA remains equal at 0.0.

Worst family summaries:
- EWoK: mean_delta `-1.6448575776036476`; worst [('material-dynamics', -11.558441558441558), ('spatial-relations', -4.285714285714292), ('physical-interactions', -3.237410071942449)]
- Entity: mean_delta `-1.4604021386814054`; worst [('ambiref_5_ops', -8.13008130081301), ('move_contents_3_ops', -8.128078817733993), ('regular_4_ops', -7.989690721649485)]
- Supplement: mean_delta `-1.3234112442271595`; worst [('qa_congruence_easy', -6.25), ('qa_congruence_tricky', -3.6363636363636402), ('hypernym', 0.4750593824227991)]
- BLiMP: mean_delta `-1.2079230106099135`; worst [('principle_A_reconstruction', -31.64426059979317), ('superlative_quantifiers_1', -23.799795709908068), ('wh_questions_object_gap', -16.06519208381839)]
- SuperGLUE: mean_delta `-1.1358364022491452`; worst [('wsc', -9.615384615384606), ('rte', -1.4388489208633075), ('mnli', -0.0407497962510206)]
- global_piqa_parallel: mean_delta `-0.9708737864077666`; worst [('global_piqa_parallel', -0.9708737864077666)]
- COMPS: mean_delta `-0.42963389043958244`; worst [('wugs_dist_in_between', -5.958549222797927), ('base', 0.18848804215646453), ('wugs', 0.223085780080595)]
- global_piqa_nonparallel: mean_delta `0.0`; worst [('global_piqa_nonparallel', 0.0)]

Largest raw subtask losses:
- BLiMP/principle_A_reconstruction: delta `-31.64426059979317`, net_correct_delta `-306`
- BLiMP/superlative_quantifiers_1: delta `-23.799795709908068`, net_correct_delta `-233`
- BLiMP/wh_questions_object_gap: delta `-16.06519208381839`, net_correct_delta `-138`
- BLiMP/sentential_subject_island: delta `-12.59105098855359`, net_correct_delta `-121`
- EWoK/material-dynamics: delta `-11.558441558441558`, net_correct_delta `-89`
- BLiMP/distractor_agreement_relative_clause: delta `-11.25143513203215`, net_correct_delta `-98`
- BLiMP/principle_A_c_command: delta `-10.570824524312897`, net_correct_delta `-100`
- BLiMP/left_branch_island_simple_question: delta `-10.515247108307044`, net_correct_delta `-100`
- SuperGLUE/wsc: delta `-9.615384615384606`, net_correct_delta `None`
- BLiMP/matrix_question_npi_licensor_present: delta `-8.934337997847148`, net_correct_delta `-83`
- Entity/ambiref_5_ops: delta `-8.13008130081301`, net_correct_delta `-10`
- Entity/move_contents_3_ops: delta `-8.128078817733993`, net_correct_delta `-33`

Reading details (positive for seed43122) are preserved separately because they do not offset the broad NLP-column loss enough to clear the visible leader.
CSV: `experiments/archive/representation_and_objectives/data/seed43122_official_failure_localization/seed43122_minus_seed43022_official_subtask_deltas.csv`
JSON: `experiments/archive/representation_and_objectives/data/seed43122_official_failure_localization/seed43122_official_failure_localization.json`

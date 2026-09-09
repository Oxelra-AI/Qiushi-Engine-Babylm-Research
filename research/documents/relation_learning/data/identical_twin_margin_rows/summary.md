# earlier analysis identical-context twin row family

earlier analysis was rejected because its A/B contexts differed in observable features. This family uses the assignment-reversal maps so A and B twins differ only in whether the assignment update names the queried entity or its co-introduced partner.

Raw maps: train=90, heldout=30; kept after equal-update-token filter: train=21, heldout=6.

## earlier analysis observable failure
Old heldout_frame_seen pairs: 720
A has one fewer update than B: 0.250
A shared-new alternative absent before answer: 1.000
Sentence-count majority rule accuracy on rows: 0.667 vs always-source 0.556

## train_frame_seen
Rows 1260; pairs 630; row words 96645; bad pairs 435; span errors 0.
Candidate presence: {'rows_with_query_source_present_before_answer': 1260, 'rows_with_shared_new_present_before_answer': 1260, 'rows_with_all_candidate_values': 1260}
Context token delta counts A-B: {'0': 240, '-1': 120, '2': 45, '1': 120, '3': 30, '-2': 60, '-3': 15}
By distractor count:
- d=0: n=252, P(source)=0.500, roles={'source_state': 126, 'new_state': 126}
- d=1: n=252, P(source)=0.500, roles={'source_state': 126, 'new_state': 126}
- d=2: n=252, P(source)=0.500, roles={'source_state': 126, 'new_state': 126}
- d=3: n=252, P(source)=0.500, roles={'source_state': 126, 'new_state': 126}
- d=4: n=252, P(source)=0.500, roles={'source_state': 126, 'new_state': 126}
Observable rule accuracies:
- always_source: 0.500
- operation_count_majority: 0.500
- context_words_majority: 0.548
- context_tokens_majority: 0.569
- query_entity_in_update_sentence: 1.000 (desired identity cue, not a nuisance balance target)

## heldout_frame_all
Rows 480; pairs 240; row words 37604; bad pairs 220; span errors 0.
Candidate presence: {'rows_with_query_source_present_before_answer': 480, 'rows_with_shared_new_present_before_answer': 480, 'rows_with_all_candidate_values': 480}
Context token delta counts A-B: {'-1': 40, '0': 40, '-3': 20, '3': 20, '1': 100, '2': 20}
By distractor count:
- d=0: n=96, P(source)=0.500, roles={'source_state': 48, 'new_state': 48}
- d=1: n=96, P(source)=0.500, roles={'source_state': 48, 'new_state': 48}
- d=2: n=96, P(source)=0.500, roles={'source_state': 48, 'new_state': 48}
- d=3: n=96, P(source)=0.500, roles={'source_state': 48, 'new_state': 48}
- d=4: n=96, P(source)=0.500, roles={'source_state': 48, 'new_state': 48}
Observable rule accuracies:
- always_source: 0.500
- operation_count_majority: 0.500
- context_words_majority: 0.600
- context_tokens_majority: 0.681
- query_entity_in_update_sentence: 1.000 (desired identity cue, not a nuisance balance target)

# relational xspan compact v4 filter — compact relational XSpan v4 filter

Input: `experiments/archive/initial_model_studies/data/xspan_revision_117/relational_xspan_v3_seed117_target200000_actual.jsonl`
Output JSONL: `experiments/archive/initial_model_studies/data/xspan_revision_118/relational_xspan_compact_v4_from_v3_seed117_target200000_actual.jsonl`
Summary: `experiments/archive/initial_model_studies/data/xspan_revision_118/relational_xspan_compact_v4_from_v3_seed117_target200000_actual.summary.json`

Kept rows: **4235** / 7477
True-context counted words: **101777**
Kept by target type: {'definition_property_complement': 1136, 'location_spatial_phrase': 1754, 'action_object_result_phrase': 509, 'semantic_content_continuation': 836}
Rejection reasons: {'length_not_2_to_7_words': 2551, 'punctuation_or_list_tail': 1578, 'hidden_sentence_boundary': 14}

**Policy:** this is the final compact data-boundary repair before likelihood testing; no model scores were used for selection. If same-target true/wrong/no-s1 likelihood signal is weak, change target representation rather than stacking more filters.

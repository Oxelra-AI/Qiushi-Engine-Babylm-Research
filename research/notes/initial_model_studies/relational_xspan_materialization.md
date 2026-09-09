# relational xspan materialization — model-independent relational XSpan materialization

JSONL: `experiments/archive/initial_model_studies/data/xspan_revision_115/relational_xspan_rule_seed115_target200000_actual.jsonl`
Summary: `experiments/archive/initial_model_studies/data/xspan_revision_115/relational_xspan_rule_seed115_target200000_actual.summary.json`

Rows: **7611**
True-context counted words: **199989**
By target type: {'definition_property_complement': 1989, 'semantic_content_continuation': 1181, 'location_spatial_phrase': 4108, 'action_object_result_phrase': 333}

**Policy:** no model scores were used to choose rows. counterfactual propagation closed protected-model scores only informed the rule family: do not target initial dependent tokens; target semantic s2 content phrases in definition/description continuations.

Each row stores true-s1, same-source length-matched wrong-s1, and no-s1 context forms for mechanism probes and controls. A trainer can use `text`/`target_span_text` for true XSpan and `text_wrong_s1`/`target_span_wrong_s1` for wrong-context control.

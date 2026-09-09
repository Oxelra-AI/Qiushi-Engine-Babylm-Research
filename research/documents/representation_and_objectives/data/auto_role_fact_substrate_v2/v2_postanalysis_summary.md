# v2 role substrate result and route v2 post-analysis

## Main finding

The v2 prompt produced many nominal counterfactual families and retained more structural variety than earlier analysis independent facts, but strict source-bridge retention is only 11/42 structurally valid families (0.262) and 11/53 original A/B cases (0.208); full-context retention is 7/42. Most row failures are counterfactual negatives accepted by at least one teacher, especially Llama, not parser noise. The retained core spans several role types and maps heuristically onto correctness transition analysis relational EWoK domains, so it is useful as seed/evaluation material, but it is too sparse for distillation from the present collection alone.

## Counts

- candidate_cases: 53
- malformed_json_cases: 9
- model_empty_reject_cases: 2
- structurally_bad_cases: 0
- ok_counterfactual_family_cases: 42
- retained_source_bridge_families: 11
- retained_full_context_families: 7
- retained_source_bridge_over_candidates: 0.20754716981132076
- retained_source_bridge_over_ok: 0.2619047619047619
- retained_full_context_over_candidates: 0.1320754716981132
- retained_full_context_over_ok: 0.16666666666666666
- distinct_retained_role_types: 7
- mapped_step211_relational_overlap_domains: ['physical-dynamics', 'physical-relations', 'social-properties', 'spatial-relations']
- finding: real_diverse_seed_core_but_sparse

## Row failure categories

- positive_rejected_by_at_least_one_teacher: 14
- negative_overaccepted_by_at_least_one_teacher: 109
- negative_accepted_by_both_teachers: 20

## Teacher failure patterns

- qwen_not_expected: 16
- llama_not_expected: 86
- both_same_wrong: 21

## role substrate review and repair plan/v2 role substrate result and route retained-case relation

- strict_case_ids: ['B010', 'B012', 'B017', 'B019', 'B022', 'B024', 'B051', 'B060', 'B061', 'B072', 'B075', 'B096', 'B098']
- source_bridge_retained_case_ids: ['B010', 'B017', 'B019', 'B022', 'B023', 'B024', 'B037', 'B060', 'B061', 'B075', 'B098']
- full_context_retained_case_ids: ['B017', 'B022', 'B023', 'B024', 'B061', 'B075', 'B098']
- intersection_step248_step249_source_bridge: ['B010', 'B017', 'B019', 'B022', 'B024', 'B060', 'B061', 'B075', 'B098']
- lost_in_step249_source_bridge: ['B012', 'B051', 'B072', 'B096']
- new_vs_step248_source_bridge: ['B023', 'B037']
- intersection_rate_of_step248_strict: 0.6923076923076923

## Consequence

Use retained families as probes and either construct a broader source-attested transformation source or search existing corpora/resources for naturally paired role-exchange transformations; do not train a student or BabyLM model from the current 11-family core.

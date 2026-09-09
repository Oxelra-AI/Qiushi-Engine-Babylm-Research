# relation frame retention audit relation-frame retention audit

This CPU-only audit inspects existing Qwen compact source/rewrite pairs. It does not score models and does not use benchmark rows to select training text.

## Core measurements

- Accepted compact pairs scanned: `18682`; selected reinvest pairs recovered: `12155`.
- Selected reinvest source-relation pair fraction: `0.8364`; unused accepted: `0.8214`.
- Selected mean source categories: `1.694`, rewrite categories: `1.338`; mean category retention on source-relational pairs: `0.6701`.
- Selected source-relational pairs that drop at least one source category: `0.5210`; drop majority: `0.2193`.
- Top selected dropped categories: `{'causal_temporal': 1547, 'quantity_measure': 1235, 'spatial_state': 1166, 'comparison_order': 851, 'negation_modality': 847, 'mental_state_social': 524, 'physical_interaction': 255, 'social_relation': 149, 'material_property': 117, 'physical_dynamics': 86}`.

## Repair-pool feasibility inside existing accepted pairs

- Bad selected definition: selected reinvest pair with >=2 source relation categories, category retention <0.5, content_recall>=0.50, entity/number recall>=0.99.
- Bad selected rows/pair-words: `862` / `32830`.
- Good unused rows/pair-words: `706` / `27414`.
- Same-primary-domain coverable bad pair-words: `27048`.

## Interpretation

- Relation-frame retention is measured from source/rewrite pairs, not from official score items. It is therefore suitable as a legal pretraining-data repair signal if future scores show relation weakness.
- A substantial fraction of selected source-relational compact pairs drop at least one broad relation category in the rewrite; compact density may therefore remove some relational framing even while preserving entities and numbers.
- There is an unused accepted-pair pool with high relation-category retention; a future CPU-only corpus dry run can attempt same-budget replacement before any new Qwen generation or GPU training is authorized.
- Selected-minus-unused mean category-retention difference is +0.0358; this indicates whether the current reinvest selector already favored or disfavored relation-preserving pairs.

## Files

- json: `experiments/archive/representation_and_objectives/data/relation_frame_retention_audit/relation_frame_retention_audit.json`
- selected_relation_drop_examples: `experiments/archive/representation_and_objectives/data/relation_frame_retention_audit/selected_relation_drop_examples.csv`
- unused_high_retention_candidates: `experiments/archive/representation_and_objectives/data/relation_frame_retention_audit/unused_high_retention_candidate_examples.csv`
- domain_category_retention_table: `experiments/archive/representation_and_objectives/data/relation_frame_retention_audit/domain_category_retention_table.csv`
- note: `research/notes/representation_and_objectives/relation_frame_retention_audit.md`

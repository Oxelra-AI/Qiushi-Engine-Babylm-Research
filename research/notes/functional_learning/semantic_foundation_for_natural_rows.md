# curated source grounded probe semantic foundation for natural recipient rows
## Why this audit was needed
The next natural-data construction cannot be made reliable by balancing entities, symmetric scoring, or stricter surface-copy filters alone. earlier analysis already enforced same source, assigned entities, a shared update frame, and a shared use frame, but the generated examples can still fail the intended semantics: the update may add a compatible object rather than replace the target's current state, the use frame may independently favor one candidate, and the validator can let a generated `target_source_state` or `distractor_source_state` help certify its own answer.
This audit measures these failure modes as proxies, not as entailment judgments. Raw lexical overlap is still not proof of source grounding; it is a lower-level check that generator metadata has not substituted for evidence in the source.
## Quantitative audit of the earlier accepted state-update pilot rows
Input: `experiments/archive/relation_learning/data/contrastive_binding_validated_strict/accepted_contrastive_binding_pairs_strict_pilot512.jsonl`; baseline scores from `experiments/archive/functional_learning/data/multitoken_scorer_pilot512/pair_scores.jsonl`. The earlier analysis validator accepted 55 pairs from 55 recorded accepted pairs at acceptance rate 0.1074.
- Both source answers were exact substrings of the raw source in 26/55 pairs (47.3%).
- Both source answers had overlap_min >= 0.34 with the raw source in 44/55 pairs (80.0%).
- At least one source answer had the self-certification pattern (weak raw-source overlap, but acceptable overlap after concatenating the generated state description) in 11/55 pairs (20.0%).
- Both generated source-state descriptions had raw-source overlap >= 0.34 in 46/55 pairs (83.6%).
- The update frame contained a strong replacement/change/assignment-like cue in 7/55 pairs (12.7%). Event-class proxy counts: `{'weak_or_unclear': 39, 'additive_or_descriptive': 9, 'strong_replacement_like': 7}`.
- Suspect entity surface forms (e.g. fragmentary roles, adjectives, dates, transcript tokens) appeared in 13/55 pairs (23.6%).

These numbers explain why the current 55-pair pilot is useful as a scorer stress test but is not yet a strong BabyLM training substrate. The problem is not merely that examples are too easy or too copy-based; many examples do not clearly entail the opposite answers from the texts a competent reader sees.
## Baseline capability should still be measured on semantically valid hard pairs
- `all`: n=55, joint=3/55, mean beta=-0.009, mean |alpha|=3.324, mean gamma=-3.333, median gamma=-2.569.
- `loose_raw_grounded_no_suspect_entity`: n=35, joint=3/35, mean beta=0.104, mean |alpha|=3.472, mean gamma=-3.368, median gamma=-2.569.
- `strict_automatic_proxy`: n=3, joint=0/3, mean beta=-0.336, mean |alpha|=2.363, mean gamma=-2.699, median gamma=-1.541.
- `rows_with_any_self_certification_gap`: n=11, joint=0/11, mean beta=-0.193, mean |alpha|=3.048, mean gamma=-3.241, median gamma=-2.692.
- `rows_without_self_certification_gap`: n=44, joint=3/44, mean beta=0.036, mean |alpha|=3.393, mean gamma=-3.356, median gamma=-2.451.
- `replacement_like`: n=7, joint=0/7, mean beta=-0.103, mean |alpha|=2.794, mean gamma=-2.897, median gamma=-1.541.
- `not_replacement_like`: n=48, joint=3/48, mean beta=0.004, mean |alpha|=3.401, mean gamma=-3.397, median gamma=-2.616.

Automatic flags do not decide scientific validity. Poor coherent86 performance on a semantically valid hard pair remains a useful missing-capability signal. The repair should reject ambiguous/non-entailing rows while preserving rows where both answer phrases are present and the learner still must select the entity to choose between competing states.
## Contract for the next tranche
A stronger construction should first establish independently source-grounded relations, then render recipient counterfactuals. The minimum object should be a source-grounded record with raw character spans or independently verified evidence for `(target_entity, relation, target_source_answer)` and `(distractor_entity, relation, distractor_source_answer)`, plus an explicit update event that changes the named entity's current value to `shared_new_answer`. The rendered pair should then be:
1. `TARGET_UPDATE`: raw source + identical replacement/update event applied to the target + identical use frame asking about the target, answer `shared_new_answer`.
2. `DISTRACTOR_UPDATE/RETAIN`: same raw source + the same update event applied to the distractor + the same use frame asking about the target, answer `target_source_answer`.
The answer strings may be copied from source/update; copying after selecting the correct entity is the desired computation when both competing phrases are available. Temporal or replacement language is also allowed when it makes the current state well-defined, provided it is shared across the two variants and cannot by itself identify whether the target or distractor was updated.
## Validator changes needed
- Do not test source-answer grounding against `source + generated_state_description`; test raw source spans and retain the span positions. If a paraphrase answer is desired, keep both an exact raw support span and a separately scored normalized answer, rather than letting the paraphrase certify itself.
- Add an independent reader/entailment pass, ideally not the same generator, over the fully rendered TARGET_UPDATE and RETAIN examples: the reader must answer the target use question with the intended candidate and mark whether the opposite answer is unsupported.
- Require explicit replacement/change/assignment semantics in the update event, but do not ban all temporal language. A shared phrase such as 'after the revision' or 'now listed as' can define current state; it is not a shortcut unless it differs between variants or points to one entity.
- Keep difficult valid pairs, including cases where both source and update candidate phrases are visible. The readout remains UPDATE, RETAIN, pair-level gamma=beta-|alpha|, and joint correctness; easy parent-solved rows are less informative for the BabyLM bridge.

## Files produced
- Structured summary: `experiments/archive/functional_learning/data/semantic_foundation_audit/semantic_audit_summary.json`
- Pair-level JSONL/CSV: `experiments/archive/functional_learning/data/semantic_foundation_audit/semantic_audit_pairs.jsonl`, `experiments/archive/functional_learning/data/semantic_foundation_audit/semantic_audit_pairs.csv`
- Manual queues: `research/documents/functional_learning/data/semantic_foundation_audit/manual_review_severe_auto_flags.md`, `research/documents/functional_learning/data/semantic_foundation_audit/manual_review_plausible_hard_rows.md`
- Figures: `experiments/archive/functional_learning/figures/semantic_flags_vs_gamma.png`, `experiments/archive/functional_learning/figures/raw_grounding_event_counts.png`

## Step038b entity-answer association addendum
Raw source overlap is still insufficient because an answer phrase can occur in the source but be associated with a different entity or event. A cheap exact-span proximity probe therefore measured whether each source answer is closer to its assigned entity than to the competing entity/source answer. This is only a proxy, but it catches association failures such as list/rank rows where the right phrase exists in the sentence but belongs to another item.
- Both source answers exact raw spans: 24/55 (43.6%).
- Both entities had their assigned source answer closer than the competing source answer under exact-span distance: 11/55 (20.0%).
- Rows with no entity-answer proximity flags: 10/55 (18.2%).
- Main entity-answer flags: `{'target_source_answer_no_exact_raw_span': 20, 'distractor_source_answer_no_exact_raw_span': 25, 'target_entity_closer_to_competing_source_answer': 5, 'target_answer_far_from_target_entity': 2, 'target_source_answer_closer_to_distractor_than_target': 8, 'distractor_entity_closer_to_competing_source_answer': 6, 'target_entity_no_exact_source_span': 2, 'distractor_source_answer_closer_to_target_than_distractor': 5, 'distractor_entity_no_exact_source_span': 3, 'distractor_answer_far_from_distractor_entity': 2}`.
This reinforces the construction change: source-grounded records need raw spans or independent evidence for the relation between each entity and answer, not merely phrase overlap with the source.
Files: `experiments/archive/functional_learning/data/entity_answer_grounding_audit/entity_answer_grounding_summary.json`, `experiments/archive/functional_learning/data/entity_answer_grounding_audit/entity_answer_grounding_pairs.jsonl`, `research/documents/functional_learning/data/entity_answer_grounding_audit/manual_review_entity_answer_suspicious.md`.

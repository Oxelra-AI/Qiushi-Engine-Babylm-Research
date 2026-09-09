# assignment reversal export synthesis assignment-reversal export synthesis

## Why this asset matters

The earlier state-update packet audit identified a real substrate problem: values were often descriptor-mediated rather than literal strings in the learner-visible context. The earlier literal-value export had the correct literal-value A/B assignment-reversal form but was too small and mostly `death_place`. The same construction was expanded into larger pools for controlled operation tests across relation and value types.

The construction remains a mechanism substrate, not evidence of broad BabyLM improvement by itself. Source sentences are exact BabyLM sentences with extracted literal values; update sentences and replacement values are controlled counterfactual additions. The right question is whether a learner can acquire the operation: given two source facts and one later update to one entity, answer the updated entity with the shared new value and the unchanged entity with its source value, across assignments, frames, relation/value types, and held source/entity content.

## Produced exports

### Larger mixed split export

- Manifest: `experiments/archive/functional_learning/data/larger_assignment_reversal_export/manifest.json`
- Operation maps: `experiments/archive/functional_learning/data/larger_assignment_reversal_export/a01_larger_assignment_reversal_operation_maps.jsonl`
- Rows: `train_frame_all` has 6,608 rows / 3,304 A/B pairs; `heldout_frame_all` has 1,648 rows / 824 A/B pairs.
- Base maps: 516 total, split train 413 / heldout 103.
- Relation counts: `birth_year` 130, `birthplace` 150, `death_place` 160, `founded_year` 21, `located_in` 20, `nationality` 18, `occupation` 17.
- Static contract: zero duplicate row IDs, zero answer-span errors, zero malformed two-row groups.
- Important boundary: this export forms many pair combinations from 309 filtered triples and has train-held source/entity reuse. It is useful for scale and type variety, but not for interpreting heldout success as source/entity-disjoint transfer.
- Feature analysis: `experiments/archive/functional_learning/data/larger_assignment_reversal_export/export_audit.json` and `.../shortcut_feature_audit_train_frame_all.json`.

### Strict source/entity-disjoint export

- Manifest: `experiments/archive/functional_learning/data/strict_disjoint_assignment_reversal_export/manifest.json`
- Operation maps: `experiments/archive/functional_learning/data/strict_disjoint_assignment_reversal_export/a01_strict_disjoint_assignment_reversal_operation_maps.jsonl`
- Rows: `train_frame_all` has 7,376 rows / 3,688 A/B pairs; `heldout_frame_all` has 1,536 rows / 768 A/B pairs.
- Base maps: 557 total, split train 461 / heldout 96.
- Relation counts: `birth_year` 139, `birthplace` 152, `death_place` 190, `founded_year` 21, `located_in` 20, `nationality` 18, `occupation` 17.
- Split property: zero train-held source-id overlap and zero train-held entity overlap; source values can overlap because years and common places recur.
- Static contract: zero duplicate row IDs, zero answer-span errors, zero malformed two-row groups.
- Boundary: held relation coverage is `birth_year`, `birthplace`, `death_place`; small relation types are train-only type variety.

### High-confidence strict export for held-transfer experiments

- Manifest: `experiments/archive/functional_learning/data/highconfidence_strict_assignment_reversal_export/manifest.json`
- Operation maps: `experiments/archive/functional_learning/data/highconfidence_strict_assignment_reversal_export/a01_highconf_strict_assignment_reversal_operation_maps.jsonl`
- Train rows: `experiments/archive/functional_learning/data/highconfidence_strict_assignment_reversal_export/a01_highconf_strict_assignment_reversal_train_frame_all.jsonl`
- Held rows: `experiments/archive/functional_learning/data/highconfidence_strict_assignment_reversal_export/a01_highconf_strict_assignment_reversal_heldout_frame_all.jsonl`
- Binding pairs are in the same directory as `binding_pairs_train_frame_all.jsonl` and `binding_pairs_heldout_frame_all.jsonl`.
- Base maps: 556 total, split train 460 / heldout 96.
- Relation counts: `birth_year` 150, `birthplace` 154, `death_place` 190, `founded_year` 10, `located_in` 20, `nationality` 16, `occupation` 16.
- Rows: `train_frame_all` has 7,360 rows / 3,680 A/B pairs; `heldout_frame_all` has 1,536 rows / 768 A/B pairs.
- Split property: zero train-held source-id overlap and zero train-held entity overlap; source-value overlap remains and should be handled as shared literal-type exposure, not novelty.
- Static contract: zero duplicate row IDs, zero answer-span errors, zero malformed two-row groups.
- Quality filter: removed triples where the extracted entity was not a whole literal substring of the exact source sentence or was far from the value. This removed examples such as `Hare` extracted from `O'Hare`, `Cormack` from `McCormack`, and `Yehudi` from `HaBayit HaYehudi`.

## Feature regularities to preserve in interpretation

For both the larger and high-confidence strict exports, the row construction intentionally gives a perfect role cue at the single-row level:

- Updated rows: the final answer is the shared new value and appears in the update sentence; the query entity is named in the update sentence.
- Unchanged rows: the final answer is the original source value and appears in the source sentence; the query entity is not named in the update sentence.

This is not a flaw in the controlled operation: the desired operation is exactly to use the update recipient to decide whether to use the new value or retrieve the source value for the other entity. But in-format success alone would be compatible with a recipient/recency rule. Stronger interpretation needs wrong-recipient and no-update comparisons, source/entity-disjoint held rows, relation and value-type splits, and broad BabyLM preservation.

## Current best use

The preferred substrate is the high-confidence strict export when held transfer matters. Use the larger mixed split only when more total rows or small held examples for small relation types matter and train-held content reuse is acceptable for the analysis. These exports are not a direct v5 route; they are a cleaner natural-language mechanism substrate for testing whether relation-aligned, literal-value experience can form a reusable assignment-update operation without the semantic failures of descriptor packets.

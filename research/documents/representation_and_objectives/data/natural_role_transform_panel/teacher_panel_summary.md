# earlier analysis natural source-attested role-preservation teacher premise test

## Purpose

Before any new architecture or Strict-Small training, test whether the raw-text role-assignment signal exists in a cheap form: an approved teacher receives one natural/source-attested sentence at a time and judges role-sensitive hypotheses. The bridge atlas is used adversarially: source-attested bridge candidates must preserve roles under paraphrase and reject actor/query swaps, otherwise they are not a training premise.

## Aggregate result

- rows: 144, valid parsed labels: 144
- overall accuracy: 0.993
- true-hypothesis accuracy: 0.986
- false role-swap rejection accuracy: 1.000
- source context accuracy: 1.000
- source-attested bridge context accuracy: 1.000
- natural compact reference context accuracy: 0.979
- bridge sentence all-facts-correct rate: 1.000
- source↔bridge gold-consistent invariance: 1.000
- all-context gold-consistent invariance: 0.979
- decision: **POSITIVE_PREMISE_FOR_TEACHER_ROLE_SIGNAL**

## Context accuracy

| context | n | acc | invalid |
|---|---:|---:|---:|
| natural_compact_reference | 48 | 0.979 | 0 |
| source | 48 | 1.000 | 0 |
| source_attested_bridge | 48 | 1.000 | 0 |

## Category accuracy

| category | n | acc | invalid |
|---|---:|---:|---:|
| actor_swap_false | 36 | 1.000 | 0 |
| conservation_false | 36 | 1.000 | 0 |
| true_actor_action | 15 | 1.000 | 0 |
| true_actor_effect | 6 | 0.833 | 0 |
| true_cause | 9 | 1.000 | 0 |
| true_comparison | 3 | 1.000 | 0 |
| true_definition | 3 | 1.000 | 0 |
| true_effect | 3 | 1.000 | 0 |
| true_embedded_role | 3 | 1.000 | 0 |
| true_instrument | 3 | 1.000 | 0 |
| true_object | 3 | 1.000 | 0 |
| true_other_role | 3 | 1.000 | 0 |
| true_result | 3 | 1.000 | 0 |
| true_scope | 3 | 1.000 | 0 |
| true_speaker | 3 | 1.000 | 0 |
| true_state | 3 | 1.000 | 0 |
| true_theme | 9 | 1.000 | 0 |

## Failure sample

### B082_natural_compact_reference_f00_true_actor_effect
- context_type=natural_compact_reference fact_category=true_actor_effect gold=ENTAILED parsed=NOT_ENTAILED raw='NOT_ENTAILED'
- context: Excess mercury in the gut upsets natural microorganism balance, which may lead to candida.
- hypothesis: Excess mercury upsets the body's microorganism balance.

## Next scientific meaning

Construct a larger independent-sentence teacher-label substrate over source-attested transformations, with held-out paraphrase and role-swap controls, then test small-student distillation against the EWoK/Entity item signature.

## Files

- prompts: `experiments/archive/representation_and_objectives/training/data/natural_role_transform_panel/teacher_prompts_all.jsonl`
- labeled rows: `experiments/archive/representation_and_objectives/data/natural_role_transform_panel/teacher_labeled_rows.jsonl`
- invariance rows: `experiments/archive/representation_and_objectives/data/natural_role_transform_panel/source_bridge_invariance.jsonl`
- summary JSON: `experiments/archive/representation_and_objectives/data/natural_role_transform_panel/teacher_panel_summary.json`

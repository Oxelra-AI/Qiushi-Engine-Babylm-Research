# paired world stress teacher and route paired-world sequence shortcut stress

## Purpose

The paired world pilot result grouped bag-of-words result is not enough to say the corpus forces general role assignment. This stress test asks whether sequence features, participant order, template identity, and context--hypothesis alignment can solve the current score-ablated paired-world task before any student training.

## Data

- families: 500
- original score-ablated NLI rows: 2000
- held original rows: 400; held rows using held templates: 204

## Transparent template/order parser

- coverage on winner-first/loser-first templates: 0.678
- accuracy on covered rows: 1.000

## Learned baselines

### raw_word12

| split | accuracy | n_test |
|---|---:|---:|
| train_to_all_held_original | 0.517 | 400 |
| train_to_held_train_templates | 0.515 | 196 |
| train_to_held_held_templates | 0.520 | 204 |
| train_to_held_name_swapped_context | 0.492 | 400 |
| train_to_held_hypothesis_lost_to | 0.497 | 400 |
| train_to_held_hypothesis_passive | 0.497 | 400 |
| train_to_held_retemplate_opposite_order | 0.500 | 400 |
| group-family CV original mean | 0.500 +/- 0.004 | 2000 |

### raw_char35

| split | accuracy | n_test |
|---|---:|---:|
| train_to_all_held_original | 0.507 | 400 |
| train_to_held_train_templates | 0.500 | 196 |
| train_to_held_held_templates | 0.515 | 204 |
| train_to_held_name_swapped_context | 0.500 | 400 |
| train_to_held_hypothesis_lost_to | 0.520 | 400 |
| train_to_held_hypothesis_passive | 0.492 | 400 |
| train_to_held_retemplate_opposite_order | 0.507 | 400 |
| group-family CV original mean | 0.505 +/- 0.007 | 2000 |

### canonical_word12

| split | accuracy | n_test |
|---|---:|---:|
| train_to_all_held_original | 0.510 | 400 |
| train_to_held_train_templates | 0.505 | 196 |
| train_to_held_held_templates | 0.515 | 204 |
| train_to_held_name_swapped_context | 0.485 | 400 |
| train_to_held_hypothesis_lost_to | 0.515 | 400 |
| train_to_held_hypothesis_passive | 0.517 | 400 |
| train_to_held_retemplate_opposite_order | 0.505 | 400 |
| group-family CV original mean | 0.504 +/- 0.008 | 2000 |

### canonical_char35

| split | accuracy | n_test |
|---|---:|---:|
| train_to_all_held_original | 0.515 | 400 |
| train_to_held_train_templates | 0.520 | 196 |
| train_to_held_held_templates | 0.510 | 204 |
| train_to_held_name_swapped_context | 0.512 | 400 |
| train_to_held_hypothesis_lost_to | 0.502 | 400 |
| train_to_held_hypothesis_passive | 0.515 | 400 |
| train_to_held_retemplate_opposite_order | 0.522 | 400 |
| group-family CV original mean | 0.523 +/- 0.004 | 2000 |

### symbolic_order_text

| split | accuracy | n_test |
|---|---:|---:|
| train_to_all_held_original | 0.500 | 400 |
| train_to_held_train_templates | 0.500 | 196 |
| train_to_held_held_templates | 0.500 | 204 |
| train_to_held_name_swapped_context | 0.500 | 400 |
| train_to_held_hypothesis_lost_to | 0.500 | 400 |
| train_to_held_hypothesis_passive | 0.500 | 400 |
| train_to_held_retemplate_opposite_order | 0.500 | 400 |
| group-family CV original mean | 0.500 +/- 0.000 | 2000 |

## Scientific meaning

A high score for canonicalized sequence or symbolic-order baselines means the current sports outcome task is still a narrow role-parsing substrate. It can remain useful for language-realization verification and for designing harder transfer tests, but it should not trigger student training by itself. The next source object must add cross-predicate transfer and a distinct event-to-state family.

## Files

- summary JSON: `experiments/archive/representation_and_objectives/data/paired_world_sequence_teacher/sequence_shortcut_stress_summary.json`
- original examples: `experiments/archive/representation_and_objectives/data/paired_world_sequence_teacher/sequence_examples_original_ablated.jsonl`
- perturbation examples: `experiments/archive/representation_and_objectives/data/paired_world_sequence_teacher/sequence_examples_perturbations.jsonl`

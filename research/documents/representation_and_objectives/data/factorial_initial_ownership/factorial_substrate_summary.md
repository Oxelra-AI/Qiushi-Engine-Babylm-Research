# analysis framework for factorial probe factorial initial-ownership substrate

## Design
Two matched training conditions built from equivariant symmetry repair and macro context data:
- **Underdetermined**: all state pairs initial=opposite (anti-copy works)
- **Disambiguated**: alternating opposite/same (anti-copy fails on half)
- Comparison rows, unsupervised text, and common-seen train identical
- Eval includes both initial patterns for all state suites

## Condition: underdetermined

### Training arms

| arm | state pairs | opp | same | comp rows | anticopy changed | event-role changed |
|---|---:|---:|---:|---:|---:|---:|
| aligned_state_bridge | 32 | 32 | 0 | 192 | 1.000 | 1.000 |
| exposure_only | 0 | 0 | 0 | 0 | n/a | n/a |
| heldheld_only | 0 | 0 | 0 | 192 | n/a | n/a |
| inverted_state_bridge | 32 | 32 | 0 | 192 | 1.000 | 1.000 |
| mixed_event_bridge | 0 | 0 | 0 | 256 | n/a | n/a |
| neutral_decoupled | 128 | 128 | 0 | 192 | 1.000 | 1.000 |

### Eval baselines

**cross_template_state_readout**

| pattern | anticopy changed | event-role changed | anticopy pair-both | event-role pair-both |
|---|---:|---:|---:|---:|
| opposite | 1.000 | 1.000 | 1.000 | 1.000 |
| same | 0.000 | 1.000 | 0.000 | 1.000 |
| all | 0.500 | 1.000 | 0.500 | 1.000 |

**heldheld_unseen_edge_closure**

(no state rows or no initial-pattern annotation)

**mixed_held_seen_orientation**

(no state rows or no initial-pattern annotation)

**name_permutation_counterfactual**

| pattern | anticopy changed | event-role changed | anticopy pair-both | event-role pair-both |
|---|---:|---:|---:|---:|
| opposite | 1.000 | 1.000 | 1.000 | 1.000 |
| same | 0.000 | 1.000 | 0.000 | 1.000 |
| all | 0.500 | 1.000 | 0.500 | 1.000 |

**paired_state_conservation**

| pattern | anticopy changed | event-role changed | anticopy pair-both | event-role pair-both |
|---|---:|---:|---:|---:|
| opposite | 1.000 | 1.000 | 1.000 | 1.000 |
| same | 0.000 | 1.000 | 0.000 | 1.000 |
| all | 0.500 | 1.000 | 0.500 | 1.000 |

## Condition: disambiguated

### Training arms

| arm | state pairs | opp | same | comp rows | anticopy changed | event-role changed |
|---|---:|---:|---:|---:|---:|---:|
| aligned_state_bridge | 32 | 16 | 16 | 192 | 0.500 | 1.000 |
| exposure_only | 0 | 0 | 0 | 0 | n/a | n/a |
| heldheld_only | 0 | 0 | 0 | 192 | n/a | n/a |
| inverted_state_bridge | 32 | 16 | 16 | 192 | 0.500 | 1.000 |
| mixed_event_bridge | 0 | 0 | 0 | 256 | n/a | n/a |
| neutral_decoupled | 128 | 128 | 0 | 192 | 1.000 | 1.000 |

### Eval baselines

**cross_template_state_readout**

| pattern | anticopy changed | event-role changed | anticopy pair-both | event-role pair-both |
|---|---:|---:|---:|---:|
| opposite | 1.000 | 1.000 | 1.000 | 1.000 |
| same | 0.000 | 1.000 | 0.000 | 1.000 |
| all | 0.500 | 1.000 | 0.500 | 1.000 |

**heldheld_unseen_edge_closure**

(no state rows or no initial-pattern annotation)

**mixed_held_seen_orientation**

(no state rows or no initial-pattern annotation)

**name_permutation_counterfactual**

| pattern | anticopy changed | event-role changed | anticopy pair-both | event-role pair-both |
|---|---:|---:|---:|---:|
| opposite | 1.000 | 1.000 | 1.000 | 1.000 |
| same | 0.000 | 1.000 | 0.000 | 1.000 |
| all | 0.500 | 1.000 | 0.500 | 1.000 |

**paired_state_conservation**

| pattern | anticopy changed | event-role changed | anticopy pair-both | event-role pair-both |
|---|---:|---:|---:|---:|
| opposite | 1.000 | 1.000 | 1.000 | 1.000 |
| same | 0.000 | 1.000 | 0.000 | 1.000 |
| all | 0.500 | 1.000 | 0.500 | 1.000 |

## Predicted transition

If disambiguation forces event-role learning:
1. Disambiguated models should achieve high changed accuracy on BOTH initial patterns
2. Underdetermined models should fail on initial=same (anti-copy predicts wrong)
3. Mixed held-seen orientation should separate (aligned > inverted) in disambiguated
4. Pair-both on initial=same should transition from low to high

## Files
- Report JSON: `experiments/archive/representation_and_objectives/data/factorial_initial_ownership/factorial_report.json`
- Formal derivation: `notes/formal_derivation_factorial_disambiguation.md`
- equivariant symmetry repair and macro context base: `experiments/archive/representation_and_objectives/data/equivariant_symmetry_substrate`
- matched state and initial owner results confound: `notes/verified_initial_owner_shortcut_and_open_controls.md`

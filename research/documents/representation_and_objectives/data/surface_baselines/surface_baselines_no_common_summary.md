# equivariant symmetry repair and macro context transparent surface baselines (no_common)

Pattern-majority classifiers over visible features. `order_only` excludes relation IDs; `order_relid` is the strongest transparent lexical check.

## Critical comparison transfer

| arm | feature group | task fit | train acc | hh eval | mixed eval | mixed unseen-pattern |
|---|---|---|---:|---:|---:|---:|
| exposure_only | order_only | relation_comparison | nan | 0.500 | 0.500 | 1.000 |
| exposure_only | order_reltype | relation_comparison | nan | 0.500 | 0.500 | 1.000 |
| exposure_only | order_relid | relation_comparison | nan | 0.500 | 0.500 | 1.000 |
| heldheld_only | order_only | relation_comparison | 0.500 | 0.500 | 0.500 | 0.625 |
| heldheld_only | order_reltype | relation_comparison | 1.000 | 0.750 | 0.500 | 1.000 |
| heldheld_only | order_relid | relation_comparison | 1.000 | 0.750 | 0.500 | 1.000 |
| aligned_state_bridge | order_only | relation_comparison | 0.500 | 0.500 | 0.500 | 0.625 |
| aligned_state_bridge | order_reltype | relation_comparison | 1.000 | 0.750 | 0.500 | 1.000 |
| aligned_state_bridge | order_relid | relation_comparison | 1.000 | 0.750 | 0.500 | 1.000 |
| inverted_state_bridge | order_only | relation_comparison | 0.500 | 0.500 | 0.500 | 0.625 |
| inverted_state_bridge | order_reltype | relation_comparison | 1.000 | 0.750 | 0.500 | 1.000 |
| inverted_state_bridge | order_relid | relation_comparison | 1.000 | 0.750 | 0.500 | 1.000 |
| neutral_decoupled | order_only | relation_comparison | 0.500 | 0.500 | 0.500 | 0.625 |
| neutral_decoupled | order_reltype | relation_comparison | 1.000 | 0.750 | 0.500 | 1.000 |
| neutral_decoupled | order_relid | relation_comparison | 1.000 | 0.750 | 0.500 | 1.000 |
| mixed_event_bridge | order_only | relation_comparison | 0.500 | 0.500 | 0.500 | 0.500 |
| mixed_event_bridge | order_reltype | relation_comparison | 1.000 | 0.750 | 0.625 | 0.750 |
| mixed_event_bridge | order_relid | relation_comparison | 1.000 | 0.750 | 0.625 | 0.750 |

## Critical changed-state transfer

| arm | feature group | task fit | train acc | paired changed | paired unchanged | cross-template changed |
|---|---|---|---:|---:|---:|---:|
| exposure_only | order_only | state_query | nan | 0.500 | 0.500 | 0.500 |
| exposure_only | order_reltype | state_query | nan | 0.500 | 0.500 | 0.500 |
| exposure_only | order_relid | state_query | nan | 0.500 | 0.500 | 0.500 |
| heldheld_only | order_only | state_query | nan | 0.500 | 0.500 | 0.500 |
| heldheld_only | order_reltype | state_query | nan | 0.500 | 0.500 | 0.500 |
| heldheld_only | order_relid | state_query | nan | 0.500 | 0.500 | 0.500 |
| aligned_state_bridge | order_only | state_query | 0.500 | 0.500 | 0.500 | 0.500 |
| aligned_state_bridge | order_reltype | state_query | 0.500 | 0.500 | 0.500 | 0.500 |
| aligned_state_bridge | order_relid | state_query | 0.750 | 0.750 | 0.500 | 0.750 |
| inverted_state_bridge | order_only | state_query | 0.500 | 0.500 | 0.500 | 0.500 |
| inverted_state_bridge | order_reltype | state_query | 0.500 | 0.500 | 0.500 | 0.500 |
| inverted_state_bridge | order_relid | state_query | 0.750 | 0.250 | 0.500 | 0.250 |
| neutral_decoupled | order_only | state_query | 0.625 | 0.500 | 0.500 | 0.500 |
| neutral_decoupled | order_reltype | state_query | 0.625 | 0.500 | 0.500 | 0.500 |
| neutral_decoupled | order_relid | state_query | 0.750 | 0.500 | 0.500 | 0.500 |
| mixed_event_bridge | order_only | state_query | nan | 0.500 | 0.500 | 0.500 |
| mixed_event_bridge | order_reltype | state_query | nan | 0.500 | 0.500 | 0.500 |
| mixed_event_bridge | order_relid | state_query | nan | 0.500 | 0.500 | 0.500 |

## Interpretation

A repaired interface is acceptable for low-cost model pilots only if order_only is at chance on mixed held-seen comparison and changed-state readouts. The order_relid group is intentionally stronger; if it exceeds chance without bridge evidence, the lexical surface itself still leaks orientation.

- results_json: `experiments/archive/representation_and_objectives/data/surface_baselines/surface_baselines_no_common.json`

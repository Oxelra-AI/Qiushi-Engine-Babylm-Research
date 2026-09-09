# equivariant symmetry repair and macro context slot-orbit solver (with_common)

Transparent positive control over the actual text: parse active/passive/give/receive events into reusable argument slots, then enumerate held-relation orientation assignments from train labels.

| arm | sat. assignments | true? | inverted? | hh true | mixed true | mixed inverted | state true changed | state inverted changed | state true both | state inverted both |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| exposure_only | 16 | True | True | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 |
| heldheld_only | 2 | True | True | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 |
| aligned_state_bridge | 1 | True | False | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 |
| inverted_state_bridge | 1 | False | True | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 |
| neutral_decoupled | 2 | True | True | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 |
| mixed_event_bridge | 1 | True | False | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 |

Reading: heldheld/neutral leave two train-consistent orientations; aligned state keeps only true; inverted keeps only inverted; mixed event keeps only true. On the repaired surface, success therefore requires a role-slot interface plus sparse orientation, not name order or lexical BoW statistics.

- results_json: `experiments/archive/representation_and_objectives/data/slot_orbit_solver/slot_orbit_solver_with_common.json`

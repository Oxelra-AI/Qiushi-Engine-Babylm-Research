# equivariant symmetry repair and macro context slot-orbit assignment-set analysis (no_common)

This is the corrected reading of the slot-orbit positive control. Singleton train-consistent assignment means the bridge identifies an orientation; two or more train-consistent assignments means orientation is not identified by that arm.

| arm | sat. | selected | mixed selected | mixed range | state changed selected | changed range | state unchanged selected | pair both selected | pair both range |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| exposure_only | 16 | ambiguous | nan | 0.000-1.000 | nan | 0.000-1.000 | nan | nan | 0.000-1.000 |
| heldheld_only | 2 | ambiguous | nan | 0.000-1.000 | nan | 0.000-1.000 | nan | nan | 0.000-1.000 |
| aligned_state_bridge | 1 | singleton | 1.000 | 1.000-1.000 | 1.000 | 1.000-1.000 | 1.000 | 1.000 | 1.000-1.000 |
| inverted_state_bridge | 1 | singleton | 0.000 | 0.000-0.000 | 0.000 | 0.000-0.000 | 1.000 | 0.000 | 0.000-0.000 |
| neutral_decoupled | 2 | ambiguous | nan | 0.000-1.000 | nan | 0.000-1.000 | nan | nan | 0.000-1.000 |
| mixed_event_bridge | 1 | singleton | 1.000 | 1.000-1.000 | 1.000 | 1.000-1.000 | 1.000 | 1.000 | 1.000-1.000 |

Reading: heldheld_only and neutral_decoupled preserve the true/inverted ambiguity, so their mixed/changed ranges span 0 to 1 but no orientation is selected. Aligned selects the true orientation; inverted selects the opposite orientation and therefore scores 0 on true-labeled mixed/changed rows while preserving unchanged rows. This is the intended positive control for the repaired text surface.

- results_json: `experiments/archive/representation_and_objectives/data/slot_orbit_solver/slot_orbit_assignment_analysis_no_common.json`

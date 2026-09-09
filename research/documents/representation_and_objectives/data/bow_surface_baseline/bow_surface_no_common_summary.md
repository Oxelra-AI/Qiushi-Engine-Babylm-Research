# equivariant symmetry repair and macro context BoW surface baseline (no_common)

## Critical relation-comparison transfer

| arm | normalized | train acc | hh eval | mixed eval |
|---|---:|---:|---:|---:|
| exposure_only | False | nan | 0.500 | 0.500 |
| exposure_only | True | nan | 0.500 | 0.500 |
| heldheld_only | False | 1.000 | 0.500 | 0.500 |
| heldheld_only | True | 0.500 | 0.500 | 0.500 |
| aligned_state_bridge | False | 1.000 | 0.500 | 0.500 |
| aligned_state_bridge | True | 0.500 | 0.500 | 0.500 |
| inverted_state_bridge | False | 1.000 | 0.500 | 0.500 |
| inverted_state_bridge | True | 0.500 | 0.500 | 0.500 |
| neutral_decoupled | False | 1.000 | 0.500 | 0.500 |
| neutral_decoupled | True | 0.500 | 0.500 | 0.500 |
| mixed_event_bridge | False | 1.000 | 0.500 | 0.500 |
| mixed_event_bridge | True | 0.500 | 0.500 | 0.500 |

## Critical state-query transfer

| arm | normalized | train acc | paired changed | paired unchanged | paired both | cross changed |
|---|---:|---:|---:|---:|---:|---:|
| exposure_only | False | nan | 0.500 | 0.500 | 0.000 | 0.500 |
| exposure_only | True | nan | 0.500 | 0.500 | 0.000 | 0.500 |
| heldheld_only | False | nan | 0.500 | 0.500 | 0.000 | 0.500 |
| heldheld_only | True | nan | 0.500 | 0.500 | 0.000 | 0.500 |
| aligned_state_bridge | False | 0.750 | 0.500 | 0.500 | 0.000 | 0.500 |
| aligned_state_bridge | True | 0.500 | 0.500 | 0.500 | 0.000 | 0.500 |
| inverted_state_bridge | False | 0.750 | 0.500 | 0.500 | 0.000 | 0.500 |
| inverted_state_bridge | True | 0.500 | 0.500 | 0.500 | 0.000 | 0.500 |
| neutral_decoupled | False | 0.750 | 0.500 | 0.500 | 0.000 | 0.500 |
| neutral_decoupled | True | 0.500 | 0.500 | 0.500 | 0.000 | 0.500 |
| mixed_event_bridge | False | nan | 0.500 | 0.500 | 0.000 | 0.500 |
| mixed_event_bridge | True | nan | 0.500 | 0.500 | 0.000 | 0.500 |

A transparent BoW model can legitimately solve unchanged rows by textual matching of the static fact. The mechanism-relevant check is whether it can orient held changed-state or mixed held-seen rows without aligned/inverted evidence.

- results_json: `experiments/archive/representation_and_objectives/data/bow_surface_baseline/bow_surface_no_common.json`

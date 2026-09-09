# Step277b: no-common-seen multi-seed pilot

| arm | seed | train_acc | hh_closure | mixed_orient | state_chg | state_unchg |
|---|---:|---:|---:|---:|---:|---:|
| exposure_only | 27700 | nan | 0.583 | 0.609 | 0.484 | 0.516 |
| heldheld_only | 27700 | 1.000 | 1.000 | 0.828 | 0.484 | 0.531 |
| aligned_state_bridge | 27700 | 0.900 | 1.000 | 0.781 | 0.688 | 0.609 |
| inverted_state_bridge | 27700 | 0.938 | 1.000 | 0.797 | 0.625 | 0.609 |
| neutral_decoupled | 27700 | 1.000 | 1.000 | 0.812 | 0.531 | 0.516 |
| mixed_event_bridge | 27700 | 1.000 | 1.000 | 0.859 | 0.531 | 0.516 |
| exposure_only | 27701 | nan | 0.500 | 0.500 | 0.500 | 0.484 |
| heldheld_only | 27701 | 1.000 | 1.000 | 0.797 | 0.500 | 0.531 |
| aligned_state_bridge | 27701 | 0.912 | 1.000 | 0.750 | 0.703 | 0.562 |
| inverted_state_bridge | 27701 | 0.925 | 1.000 | 0.750 | 0.641 | 0.625 |
| neutral_decoupled | 27701 | 1.000 | 1.000 | 0.812 | 0.500 | 0.516 |
| mixed_event_bridge | 27701 | 1.000 | 1.000 | 0.750 | 0.516 | 0.484 |
| exposure_only | 27702 | nan | 0.500 | 0.500 | 0.594 | 0.484 |
| heldheld_only | 27702 | 1.000 | 1.000 | 0.844 | 0.594 | 0.531 |
| aligned_state_bridge | 27702 | 0.938 | 1.000 | 0.812 | 0.688 | 0.656 |
| inverted_state_bridge | 27702 | 0.887 | 1.000 | 0.781 | 0.688 | 0.594 |
| neutral_decoupled | 27702 | 0.975 | 1.000 | 0.750 | 0.469 | 0.500 |
| mixed_event_bridge | 27702 | 1.000 | 1.000 | 0.797 | 0.516 | 0.500 |

## Cross-seed means

| arm | mean_hh | mean_mixed | std_mixed | mean_state_chg |
|---|---:|---:|---:|---:|
| exposure_only | 0.528 | 0.536 | 0.052 | 0.526 |
| heldheld_only | 1.000 | 0.823 | 0.019 | 0.526 |
| aligned_state_bridge | 1.000 | 0.781 | 0.026 | 0.693 |
| inverted_state_bridge | 1.000 | 0.776 | 0.019 | 0.651 |
| neutral_decoupled | 1.000 | 0.792 | 0.029 | 0.500 |
| mixed_event_bridge | 1.000 | 0.802 | 0.045 | 0.521 |

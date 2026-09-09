# Step277b: no-common-seen multi-seed pilot

| arm | seed | train_acc | hh_closure | mixed_orient | state_chg | state_unchg |
|---|---:|---:|---:|---:|---:|---:|
| exposure_only | 27700 | nan | 0.875 | 0.625 | 0.500 | 0.547 |
| heldheld_only | 27700 | 1.000 | 1.000 | 0.750 | 0.500 | 0.500 |
| aligned_state_bridge | 27700 | 0.925 | 1.000 | 0.734 | 0.578 | 0.719 |
| inverted_state_bridge | 27700 | 0.925 | 1.000 | 0.750 | 0.609 | 0.516 |
| neutral_decoupled | 27700 | 1.000 | 1.000 | 0.750 | 0.484 | 0.500 |
| mixed_event_bridge | 27700 | 1.000 | 1.000 | 0.625 | 0.500 | 0.500 |
| exposure_only | 27701 | nan | 0.500 | 0.500 | 0.516 | 0.500 |
| heldheld_only | 27701 | 1.000 | 1.000 | 0.750 | 0.516 | 0.500 |
| aligned_state_bridge | 27701 | 0.912 | 1.000 | 0.750 | 0.734 | 0.719 |
| inverted_state_bridge | 27701 | 0.912 | 1.000 | 0.703 | 0.688 | 0.641 |
| neutral_decoupled | 27701 | 1.000 | 1.000 | 0.750 | 0.484 | 0.469 |
| mixed_event_bridge | 27701 | 0.964 | 1.000 | 0.594 | 0.484 | 0.500 |
| exposure_only | 27702 | nan | 0.500 | 0.469 | 0.578 | 0.422 |
| heldheld_only | 27702 | 1.000 | 1.000 | 0.750 | 0.516 | 0.484 |
| aligned_state_bridge | 27702 | 0.900 | 1.000 | 0.750 | 0.922 | 0.484 |
| inverted_state_bridge | 27702 | 0.912 | 1.000 | 0.750 | 0.828 | 0.625 |
| neutral_decoupled | 27702 | 1.000 | 1.000 | 0.750 | 0.641 | 0.484 |
| mixed_event_bridge | 27702 | 1.000 | 1.000 | 0.672 | 0.531 | 0.516 |
| exposure_only | 27703 | nan | 0.500 | 0.500 | 0.484 | 0.516 |
| heldheld_only | 27703 | 1.000 | 1.000 | 0.750 | 0.562 | 0.500 |
| aligned_state_bridge | 27703 | 0.900 | 1.000 | 0.750 | 0.812 | 0.562 |
| inverted_state_bridge | 27703 | 0.900 | 1.000 | 0.750 | 0.609 | 0.578 |
| neutral_decoupled | 27703 | 0.988 | 1.000 | 0.719 | 0.500 | 0.531 |
| mixed_event_bridge | 27703 | 1.000 | 1.000 | 0.688 | 0.500 | 0.500 |
| exposure_only | 27704 | nan | 0.500 | 0.500 | 0.500 | 0.500 |
| heldheld_only | 27704 | 1.000 | 1.000 | 0.750 | 0.500 | 0.500 |
| aligned_state_bridge | 27704 | 0.887 | 1.000 | 0.750 | 0.828 | 0.672 |
| inverted_state_bridge | 27704 | 0.863 | 1.000 | 0.750 | 0.641 | 0.578 |
| neutral_decoupled | 27704 | 1.000 | 1.000 | 0.750 | 0.453 | 0.531 |
| mixed_event_bridge | 27704 | 1.000 | 1.000 | 0.547 | 0.484 | 0.516 |

## Cross-seed means

| arm | mean_hh | mean_mixed | std_mixed | mean_state_chg |
|---|---:|---:|---:|---:|
| exposure_only | 0.575 | 0.519 | 0.054 | 0.516 |
| heldheld_only | 1.000 | 0.750 | 0.000 | 0.519 |
| aligned_state_bridge | 1.000 | 0.747 | 0.006 | 0.775 |
| inverted_state_bridge | 1.000 | 0.741 | 0.019 | 0.675 |
| neutral_decoupled | 1.000 | 0.744 | 0.012 | 0.512 |
| mixed_event_bridge | 1.000 | 0.625 | 0.051 | 0.500 |

# identity orbit randomization and role coordinate identity-orbit alias x sparse anchor test
Device: `cuda`; elapsed 114.56 s. Minimal predicate texts remove score/date/tournament event-ID channels.

## Mean accuracies by alias mode and sparse-anchor arm

| mode | arm | k/template | fit | train | anchor-held | probe-held | probe-trainfam | held-template |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_names | zero | 0 | 2/2 | 0.999 | 0.639 | 0.578 | 0.583 | 0.589 |
| fixed_names | true | 2 | 2/2 | 1.000 | 0.587 | 0.544 | 0.540 | 0.546 |
| fixed_names | shuffled | 2 | 2/2 | 1.000 | 0.604 | 0.534 | 0.488 | 0.575 |
| fixed_names | exposure | 2 | 2/2 | 1.000 | 0.652 | 0.523 | 0.535 | 0.577 |
| fixed_names | true | 8 | 2/2 | 1.000 | 0.727 | 0.646 | 0.694 | 0.623 |
| fixed_names | shuffled | 8 | 2/2 | 0.999 | 0.669 | 0.492 | 0.499 | 0.591 |
| fixed_names | exposure | 8 | 2/2 | 0.999 | 0.648 | 0.553 | 0.584 | 0.583 |
| fixed_names | true | 32 | 2/2 | 1.000 | 0.616 | 0.581 | 0.574 | 0.557 |
| fixed_names | shuffled | 32 | 2/2 | 1.000 | 0.612 | 0.405 | 0.378 | 0.543 |
| fixed_names | exposure | 32 | 2/2 | 0.999 | 0.659 | 0.512 | 0.528 | 0.588 |
| family_alias | zero | 0 | 2/2 | 1.000 | 0.886 | 0.737 | 0.771 | 0.720 |
| family_alias | true | 2 | 2/2 | 0.999 | 0.908 | 0.743 | 0.758 | 0.736 |
| family_alias | shuffled | 2 | 2/2 | 1.000 | 0.906 | 0.520 | 0.522 | 0.604 |
| family_alias | exposure | 2 | 2/2 | 1.000 | 0.900 | 0.670 | 0.674 | 0.779 |
| family_alias | true | 8 | 2/2 | 1.000 | 0.895 | 0.846 | 0.864 | 0.773 |
| family_alias | shuffled | 8 | 2/2 | 1.000 | 0.903 | 0.374 | 0.363 | 0.706 |
| family_alias | exposure | 8 | 2/2 | 0.999 | 0.904 | 0.552 | 0.556 | 0.746 |
| family_alias | true | 32 | 2/2 | 1.000 | 0.893 | 0.861 | 0.876 | 0.785 |
| family_alias | shuffled | 32 | 2/2 | 0.999 | 0.883 | 0.206 | 0.135 | 0.667 |
| family_alias | exposure | 32 | 2/2 | 1.000 | 0.877 | 0.547 | 0.540 | 0.743 |
| per_item_alias | zero | 0 | 2/2 | 0.999 | 0.968 | 0.818 | 0.819 | 0.856 |
| per_item_alias | true | 2 | 2/2 | 0.998 | 0.966 | 0.885 | 0.878 | 0.853 |
| per_item_alias | shuffled | 2 | 2/2 | 0.994 | 0.957 | 0.574 | 0.544 | 0.702 |
| per_item_alias | exposure | 2 | 2/2 | 0.999 | 0.963 | 0.639 | 0.642 | 0.754 |
| per_item_alias | true | 8 | 2/2 | 0.999 | 0.973 | 0.925 | 0.928 | 0.801 |
| per_item_alias | shuffled | 8 | 2/2 | 0.995 | 0.963 | 0.292 | 0.289 | 0.782 |
| per_item_alias | exposure | 8 | 2/2 | 0.999 | 0.966 | 0.557 | 0.527 | 0.735 |
| per_item_alias | true | 32 | 2/2 | 0.996 | 0.960 | 0.913 | 0.920 | 0.804 |
| per_item_alias | shuffled | 32 | 2/2 | 0.999 | 0.965 | 0.107 | 0.098 | 0.787 |
| per_item_alias | exposure | 32 | 2/2 | 0.999 | 0.978 | 0.550 | 0.540 | 0.796 |

Interpretation should use only train-fit runs. `probe_heldfam` is the mixed seen-hypothesis/held-context surface with new source families; `probe_trainfam` shows same-family shortcut sensitivity.

Summary JSON: `experiments/archive/representation_and_objectives/data/identity_orbit_anchor_test/identity_orbit_anchor_summary.json`

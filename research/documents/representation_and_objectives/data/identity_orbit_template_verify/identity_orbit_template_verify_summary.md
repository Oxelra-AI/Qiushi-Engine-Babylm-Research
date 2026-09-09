# identity orbit randomization and role coordinate focused template verification

Device `cuda`, elapsed 154.78 s. Same train/eval event selection as the main identity orbit randomization and role coordinate identity-orbit run; 5 seeds for fixed-names and per-item aliases.

## Aggregate held-family transfer

| mode | arm | k/template | fit | train | probe-held | held-template |
|---|---:|---:|---:|---:|---:|---:|
| fixed_names | zero | 0 | 5/5 | 1.000 | 0.550 | 0.568 |
| fixed_names | true | 2 | 5/5 | 0.998 | 0.546 | 0.550 |
| fixed_names | shuffled | 2 | 5/5 | 1.000 | 0.497 | 0.573 |
| fixed_names | exposure | 2 | 5/5 | 0.999 | 0.534 | 0.571 |
| fixed_names | true | 8 | 5/5 | 0.999 | 0.570 | 0.548 |
| fixed_names | shuffled | 8 | 5/5 | 0.999 | 0.476 | 0.546 |
| fixed_names | exposure | 8 | 5/5 | 0.999 | 0.500 | 0.580 |
| per_item_alias | zero | 0 | 5/5 | 1.000 | 0.788 | 0.793 |
| per_item_alias | true | 2 | 5/5 | 0.997 | 0.835 | 0.764 |
| per_item_alias | shuffled | 2 | 5/5 | 0.998 | 0.571 | 0.779 |
| per_item_alias | exposure | 2 | 5/5 | 0.998 | 0.687 | 0.774 |
| per_item_alias | true | 8 | 5/5 | 0.998 | 0.905 | 0.777 |
| per_item_alias | shuffled | 8 | 5/5 | 0.997 | 0.302 | 0.753 |
| per_item_alias | exposure | 8 | 5/5 | 0.996 | 0.647 | 0.710 |

## Per-item alias per-template held-family accuracy

| arm | k/template | T04 | T05 | T09 | T10 | T15 | T16 | T17 | T18 | T19 | T20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| zero | 0 | 0.851 | 0.666 | 0.914 | 0.829 | 0.668 | 0.841 | 0.856 | 0.748 | 0.849 | 0.702 |
| true | 2 | 0.896 | 0.786 | 0.924 | 0.818 | 0.776 | 0.785 | 0.820 | 0.709 | 0.833 | 0.691 |
| shuffled | 2 | 0.562 | 0.492 | 0.672 | 0.616 | 0.511 | 0.921 | 0.779 | 0.715 | 0.749 | 0.756 |
| exposure | 2 | 0.669 | 0.582 | 0.791 | 0.661 | 0.760 | 0.846 | 0.804 | 0.676 | 0.799 | 0.776 |
| true | 8 | 0.906 | 0.901 | 0.933 | 0.871 | 0.933 | 0.836 | 0.812 | 0.664 | 0.824 | 0.747 |
| shuffled | 8 | 0.416 | 0.180 | 0.334 | 0.331 | 0.193 | 0.721 | 0.783 | 0.681 | 0.856 | 0.694 |
| exposure | 8 | 0.777 | 0.524 | 0.726 | 0.610 | 0.576 | 0.661 | 0.693 | 0.743 | 0.768 | 0.709 |

Summary JSON: `experiments/archive/representation_and_objectives/data/identity_orbit_template_verify/identity_orbit_template_verify_summary.json`

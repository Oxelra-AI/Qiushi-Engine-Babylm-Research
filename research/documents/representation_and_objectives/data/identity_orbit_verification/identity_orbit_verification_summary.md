# earlier analysis independent verification of the identity orbit randomization and role coordinate identity-orbit mechanism

Device `cuda`, elapsed 102.48 s.

## Part A construction audits

- family id overlap: 0; participant-name overlap between splits: 41 of 192 held names.
- anchor/probe/held template overlaps: [] / [] / [].

| mode | alias-order predicts label | best first-alias majority | alias collisions | ctx/hyp alias shared |
|---|---:|---:|---:|---:|
| fixed_names | 0.500 | 0.500 | 0 | 1.000 |
| family_alias | 0.500 | 0.500 | 0 | 1.000 |
| per_item_alias | 0.489 | 0.567 | 0 | 1.000 |

## Part B matched-row-count realization diversity

Total anchor rows fixed at 1600; D distinct anchor predicates per event trades against event breadth.

| mode | D | events | rows | arm | fit | train | probe-held | anchor-held | held-template |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| fixed_names | 1 | 800 | 1600 | zero | 0/3 | 0.505 | 0.502 | 0.501 | 0.500 |
| fixed_names | 1 | 800 | 1680 | true | 0/3 | 0.505 | 0.501 | 0.499 | 0.501 |
| fixed_names | 1 | 800 | 1680 | shuffled | 0/3 | 0.507 | 0.500 | 0.501 | 0.500 |
| fixed_names | 2 | 400 | 1600 | zero | 0/3 | 0.767 | 0.533 | 0.606 | 0.534 |
| fixed_names | 2 | 400 | 1680 | true | 0/3 | 0.844 | 0.544 | 0.621 | 0.547 |
| fixed_names | 2 | 400 | 1680 | shuffled | 0/3 | 0.781 | 0.484 | 0.580 | 0.518 |
| fixed_names | 5 | 160 | 1600 | zero | 0/3 | 0.922 | 0.512 | 0.532 | 0.511 |
| fixed_names | 5 | 160 | 1680 | true | 0/3 | 0.889 | 0.497 | 0.502 | 0.497 |
| fixed_names | 5 | 160 | 1680 | shuffled | 0/3 | 0.935 | 0.500 | 0.520 | 0.507 |
| fixed_names | 10 | 80 | 1600 | zero | 2/3 | 0.976 | 0.503 | 0.505 | 0.504 |
| fixed_names | 10 | 80 | 1680 | true | 2/3 | 0.974 | 0.504 | 0.496 | 0.493 |
| fixed_names | 10 | 80 | 1680 | shuffled | 2/3 | 0.980 | 0.493 | 0.493 | 0.494 |
| family_alias | 1 | 800 | 1600 | zero | 0/3 | 0.626 | 0.552 | 0.599 | 0.575 |
| family_alias | 1 | 800 | 1680 | true | 0/3 | 0.523 | 0.501 | 0.512 | 0.502 |
| family_alias | 1 | 800 | 1680 | shuffled | 0/3 | 0.509 | 0.501 | 0.502 | 0.503 |
| family_alias | 2 | 400 | 1600 | zero | 3/3 | 0.989 | 0.654 | 0.924 | 0.753 |
| family_alias | 2 | 400 | 1680 | true | 2/3 | 0.977 | 0.833 | 0.893 | 0.698 |
| family_alias | 2 | 400 | 1680 | shuffled | 1/3 | 0.854 | 0.467 | 0.762 | 0.615 |
| family_alias | 5 | 160 | 1600 | zero | 3/3 | 0.997 | 0.592 | 0.758 | 0.673 |
| family_alias | 5 | 160 | 1680 | true | 3/3 | 0.999 | 0.761 | 0.853 | 0.720 |
| family_alias | 5 | 160 | 1680 | shuffled | 3/3 | 0.996 | 0.459 | 0.789 | 0.698 |
| family_alias | 10 | 80 | 1600 | zero | 3/3 | 1.000 | 0.507 | 0.538 | 0.522 |
| family_alias | 10 | 80 | 1680 | true | 3/3 | 0.992 | 0.586 | 0.546 | 0.512 |
| family_alias | 10 | 80 | 1680 | shuffled | 3/3 | 0.991 | 0.468 | 0.536 | 0.517 |
| per_item_alias | 1 | 800 | 1600 | zero | 2/3 | 0.978 | 0.633 | 0.740 | 0.609 |
| per_item_alias | 1 | 800 | 1680 | true | 2/3 | 0.985 | 0.749 | 0.846 | 0.696 |
| per_item_alias | 1 | 800 | 1680 | shuffled | 1/3 | 0.951 | 0.536 | 0.737 | 0.583 |
| per_item_alias | 2 | 400 | 1600 | zero | 3/3 | 0.996 | 0.687 | 0.886 | 0.702 |
| per_item_alias | 2 | 400 | 1680 | true | 3/3 | 0.995 | 0.753 | 0.868 | 0.683 |
| per_item_alias | 2 | 400 | 1680 | shuffled | 2/3 | 0.969 | 0.479 | 0.807 | 0.596 |
| per_item_alias | 5 | 160 | 1600 | zero | 3/3 | 0.990 | 0.675 | 0.840 | 0.636 |
| per_item_alias | 5 | 160 | 1680 | true | 2/3 | 0.969 | 0.713 | 0.793 | 0.676 |
| per_item_alias | 5 | 160 | 1680 | shuffled | 1/3 | 0.955 | 0.531 | 0.762 | 0.599 |
| per_item_alias | 10 | 80 | 1600 | zero | 3/3 | 0.989 | 0.647 | 0.820 | 0.631 |
| per_item_alias | 10 | 80 | 1680 | true | 2/3 | 0.976 | 0.719 | 0.820 | 0.671 |
| per_item_alias | 10 | 80 | 1680 | shuffled | 0/3 | 0.921 | 0.507 | 0.651 | 0.577 |

## Aligned minus anti-aligned separation at fixed row count

| mode | D | zero | true | shuffled | true-shuffled |
|---|---:|---:|---:|---:|---:|
| fixed_names | 1 | 0.502 | 0.501 | 0.500 | +0.002 |
| fixed_names | 2 | 0.533 | 0.544 | 0.484 | +0.060 |
| fixed_names | 5 | 0.512 | 0.497 | 0.500 | -0.003 |
| fixed_names | 10 | 0.503 | 0.504 | 0.493 | +0.011 |
| family_alias | 1 | 0.552 | 0.501 | 0.501 | +0.000 |
| family_alias | 2 | 0.654 | 0.833 | 0.467 | +0.366 |
| family_alias | 5 | 0.592 | 0.761 | 0.459 | +0.303 |
| family_alias | 10 | 0.507 | 0.586 | 0.468 | +0.118 |
| per_item_alias | 1 | 0.633 | 0.749 | 0.536 | +0.213 |
| per_item_alias | 2 | 0.687 | 0.753 | 0.479 | +0.274 |
| per_item_alias | 5 | 0.675 | 0.713 | 0.531 | +0.182 |
| per_item_alias | 10 | 0.647 | 0.719 | 0.507 | +0.212 |

Summary JSON: `experiments/archive/representation_and_objectives/data/identity_orbit_verification/identity_orbit_verification_summary.json`

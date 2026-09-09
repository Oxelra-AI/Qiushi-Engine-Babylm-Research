# orbit pair audit and slot reuse update slot reuse control refinement

Device `cuda`, elapsed 88.18 s. All modes have matched extra-row count.

## Held-pool exposure

| mode | held types | total held occ | mean | min | max |
|---|---:|---:|---:|---:|---:|
| absent_control | 0/32 | 0 | 0.0 | 0 | 0 |
| pure_match | 32/32 | 4800 | 150.0 | 60 | 240 |
| ordered_neutral | 32/32 | 9600 | 300.0 | 144 | 456 |
| anchor_random | 32/32 | 9600 | 300.0 | 144 | 456 |
| anchor_true | 32/32 | 9600 | 300.0 | 144 | 456 |
| anchor_flip | 32/32 | 9600 | 300.0 | 144 | 456 |

## Balanced held-pool probe accuracy

| mode | arm | fit | train | probe | W-first | L-first | anchor | held-template |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| absent_control | zero | 3/3 | 1.000 | 0.583 | 0.524 | 0.641 | 0.759 | 0.569 |
| absent_control | true | 3/3 | 1.000 | 0.747 | 0.750 | 0.744 | 0.817 | 0.651 |
| absent_control | anti | 3/3 | 0.999 | 0.416 | 0.370 | 0.462 | 0.735 | 0.605 |
| pure_match | zero | 3/3 | 1.000 | 0.671 | 0.598 | 0.744 | 0.839 | 0.607 |
| pure_match | true | 3/3 | 0.999 | 0.777 | 0.768 | 0.786 | 0.844 | 0.691 |
| pure_match | anti | 3/3 | 1.000 | 0.419 | 0.333 | 0.506 | 0.830 | 0.673 |
| ordered_neutral | zero | 3/3 | 1.000 | 0.685 | 0.748 | 0.623 | 0.797 | 0.669 |
| ordered_neutral | true | 3/3 | 0.999 | 0.763 | 0.783 | 0.742 | 0.852 | 0.622 |
| ordered_neutral | anti | 3/3 | 0.999 | 0.359 | 0.387 | 0.331 | 0.881 | 0.676 |
| anchor_random | zero | 0/3 | 0.840 | 0.512 | 0.496 | 0.527 | 0.553 | 0.534 |
| anchor_random | true | 0/3 | 0.819 | 0.557 | 0.548 | 0.567 | 0.569 | 0.526 |
| anchor_random | anti | 0/3 | 0.838 | 0.472 | 0.478 | 0.467 | 0.527 | 0.520 |
| anchor_true | zero | 3/3 | 1.000 | 0.689 | 0.501 | 0.877 | 0.933 | 0.769 |
| anchor_true | true | 3/3 | 1.000 | 0.815 | 0.749 | 0.881 | 0.923 | 0.754 |
| anchor_true | anti | 3/3 | 0.999 | 0.310 | 0.212 | 0.408 | 0.936 | 0.652 |
| anchor_flip | zero | 3/3 | 1.000 | 0.401 | 0.530 | 0.272 | 0.168 | 0.356 |
| anchor_flip | true | 3/3 | 1.000 | 0.270 | 0.335 | 0.205 | 0.176 | 0.388 |
| anchor_flip | anti | 3/3 | 1.000 | 0.560 | 0.654 | 0.466 | 0.134 | 0.319 |

## Coordinate separation

| mode | zero | true | anti | true-anti | true-zero |
|---|---:|---:|---:|---:|---:|
| absent_control | 0.583 | 0.747 | 0.416 | +0.331 | +0.164 |
| pure_match | 0.671 | 0.777 | 0.419 | +0.358 | +0.107 |
| ordered_neutral | 0.685 | 0.763 | 0.359 | +0.404 | +0.078 |
| anchor_random | 0.512 | 0.557 | 0.472 | +0.085 | +0.045 |
| anchor_true | 0.689 | 0.815 | 0.310 | +0.505 | +0.127 |
| anchor_flip | 0.401 | 0.270 | 0.560 | -0.290 | -0.131 |

Summary JSON: `experiments/archive/representation_and_objectives/data/slot_reuse_control_refinement/slot_reuse_control_refinement_summary.json`

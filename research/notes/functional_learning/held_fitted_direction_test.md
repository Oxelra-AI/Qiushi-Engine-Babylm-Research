# held fitted direction test held-fitted direction test

## Purpose

This tests whether direct-full held failure is only a rotation away from the preparation/train direction. Directions are fitted from train, one-held-query, or two-held-query L1 states in the current endpoint model, then evaluated by classification and d-only donor-query redirection. The two-held condition separates query-conditioned selection among held tokens from simply finding the unique held slot.

## Mean results across seeds

### prep

| fitted direction | cls train | cls one-held | cls two-held | redir train | redir one-held | redir two-held | orth two-held |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 0.980 | 0.852 | 0.675 | 1.000 | 0.875 | 0.686 | 0.252 |
| held_one | 0.980 | 0.873 | 0.675 | 1.000 | 0.869 | 0.664 | 0.275 |
| two_held | 0.982 | 0.873 | 0.690 | 1.000 | 0.873 | 0.666 | 0.262 |

### direct_full

| fitted direction | cls train | cls one-held | cls two-held | redir train | redir one-held | redir two-held | orth two-held |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 0.500 | 0.372 | 0.328 | 0.879 | 0.408 | 0.451 | 0.146 |
| held_one | 0.485 | 0.368 | 0.312 | 0.395 | 0.297 | 0.338 | 0.254 |
| two_held | 0.475 | 0.343 | 0.312 | 0.623 | 0.377 | 0.338 | 0.256 |

### static_1over17

| fitted direction | cls train | cls one-held | cls two-held | redir train | redir one-held | redir two-held | orth two-held |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 0.902 | 0.598 | 0.550 | 1.000 | 0.719 | 0.691 | 0.258 |
| held_one | 0.952 | 0.742 | 0.627 | 0.979 | 0.639 | 0.598 | 0.338 |
| two_held | 0.897 | 0.663 | 0.575 | 0.988 | 0.656 | 0.621 | 0.322 |

### interleaved_ans_full

| fitted direction | cls train | cls one-held | cls two-held | redir train | redir one-held | redir two-held | orth two-held |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 0.998 | 0.745 | 0.643 | 1.000 | 0.834 | 0.723 | 0.232 |
| held_one | 1.000 | 0.788 | 0.672 | 1.000 | 0.826 | 0.705 | 0.275 |
| two_held | 1.000 | 0.788 | 0.663 | 1.000 | 0.824 | 0.707 | 0.268 |

## Interpretation

If direct-full held failure were only a hidden rotation, a direction fitted from held or two-held states should recover high held/two-held redirection even when the train/preparation direction fails. If held-fitted directions do not rescue two-held redirection while train redirection is high, the endpoint has narrowed the functional interface to trained query symbols rather than preserving a rotated held selector.

## Files

- Data: `experiments/archive/functional_learning/data/held_fitted_direction_test/results.json`
- Script: `experiments/archive/functional_learning/scripts/held_fitted_direction_test.py`

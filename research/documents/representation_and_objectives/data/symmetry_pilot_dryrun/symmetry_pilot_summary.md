# symmetry pilot boundary findings symmetry-identification learned pilot

## Predicted pattern

| arm | heldheld_closure | mixed_orientation | state_changed | state_unchanged |
|---|---|---|---|---|
| exposure_only | chance | chance | chance | chance |
| heldheld_only | high | ~0.500 | varies | varies |
| aligned_state | high | high | high | high |
| inverted_state | high | LOW (~0.0) | inverted | varies |
| neutral | high | ~0.500 | varies | varies |
| mixed_event | high | high | varies | varies |

## Observed results

| arm | train_acc | hh_closure | mixed_orient | state_conserv | state_conserv_chg | state_conserv_unchg | cross_tmpl | name_perm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| heldheld_only | 0.588 | 1.000 | 0.656 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 |

## Interpretation

If aligned >> inverted on mixed_orientation while both high on hh_closure, sparse bridges identify absolute role orientation from nonce lexical components. If heldheld_only and neutral are near 0.500 on mixed, the Z2 ambiguity is real and cannot be resolved without bridge evidence.

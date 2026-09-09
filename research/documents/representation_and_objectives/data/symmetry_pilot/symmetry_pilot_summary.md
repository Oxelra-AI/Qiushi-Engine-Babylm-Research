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
| exposure_only | 0.984 | 0.500 | 0.500 | 0.742 | 0.719 | 0.766 | 0.734 | 0.688 |
| heldheld_only | 0.988 | 1.000 | 0.766 | 0.836 | 0.859 | 0.812 | 0.875 | 0.766 |
| aligned_state_bridge | 0.805 | 1.000 | 0.734 | 0.773 | 0.750 | 0.797 | 0.781 | 0.547 |
| inverted_state_bridge | 0.849 | 1.000 | 0.766 | 0.828 | 0.828 | 0.828 | 0.844 | 0.531 |
| neutral_decoupled | 0.971 | 1.000 | 0.750 | 0.781 | 0.844 | 0.719 | 0.812 | 0.609 |
| mixed_event_bridge | 0.996 | 1.000 | 0.719 | 0.688 | 0.656 | 0.719 | 0.719 | 0.609 |

## Interpretation

If aligned >> inverted on mixed_orientation while both high on hh_closure, sparse bridges identify absolute role orientation from nonce lexical components. If heldheld_only and neutral are near 0.500 on mixed, the Z2 ambiguity is real and cannot be resolved without bridge evidence.

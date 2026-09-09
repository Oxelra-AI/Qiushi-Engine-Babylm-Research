# earlier analysis balanced heuristic baselines

CPU-only transparent rules on the exact `replace_k00_spread` and `replace_k16_spread` files used by the earlier analysis learned probe.  These are not learned results; they define shortcut ceilings/floors for interpreting the learned run.

## Training state rows: changed-true accuracy

| condition | arm | rule | changed true acc | pair-both acc | changed same acc | changed opposite acc | offdiag same/st0 | offdiag opposite/st1 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| replace_k00_spread | aligned_state_bridge | anti_copy_changed__static_copy_unchanged | 1.000 | 1.000 | n/a | 1.000 | n/a | 1.000 |
| replace_k00_spread | aligned_state_bridge | copy_initial_changed__static_copy_unchanged | 0.000 | 0.000 | n/a | 0.000 | n/a | 0.000 |
| replace_k00_spread | aligned_state_bridge | static_slot_step282_switch | 0.500 | 0.500 | n/a | 0.500 | n/a | 0.000 |
| replace_k00_spread | aligned_state_bridge | oracle_event_role_state | 1.000 | 1.000 | n/a | 1.000 | n/a | 1.000 |
| replace_k00_spread | inverted_state_bridge | anti_copy_changed__static_copy_unchanged | 1.000 | 1.000 | n/a | 1.000 | n/a | 1.000 |
| replace_k00_spread | inverted_state_bridge | copy_initial_changed__static_copy_unchanged | 0.000 | 0.000 | n/a | 0.000 | n/a | 0.000 |
| replace_k00_spread | inverted_state_bridge | static_slot_step282_switch | 0.500 | 0.500 | n/a | 0.500 | n/a | 0.000 |
| replace_k00_spread | inverted_state_bridge | oracle_event_role_state | 1.000 | 1.000 | n/a | 1.000 | n/a | 1.000 |
| replace_k00_spread | heldheld_only | anti_copy_changed__static_copy_unchanged | n/a | n/a | n/a | n/a | n/a | n/a |
| replace_k00_spread | heldheld_only | copy_initial_changed__static_copy_unchanged | 0.000 | 0.000 | n/a | n/a | n/a | n/a |
| replace_k00_spread | heldheld_only | static_slot_step282_switch | n/a | n/a | n/a | n/a | n/a | n/a |
| replace_k00_spread | heldheld_only | oracle_event_role_state | 1.000 | 1.000 | n/a | n/a | n/a | n/a |
| replace_k16_spread | aligned_state_bridge | anti_copy_changed__static_copy_unchanged | 0.500 | 0.500 | 0.000 | 1.000 | 0.000 | 1.000 |
| replace_k16_spread | aligned_state_bridge | copy_initial_changed__static_copy_unchanged | 0.056 | 0.056 | 1.000 | 0.000 | 1.000 | 0.000 |
| replace_k16_spread | aligned_state_bridge | static_slot_step282_switch | 0.500 | 0.500 | 0.500 | 0.500 | 0.000 | 0.000 |
| replace_k16_spread | aligned_state_bridge | oracle_event_role_state | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| replace_k16_spread | inverted_state_bridge | anti_copy_changed__static_copy_unchanged | 0.500 | 0.500 | 0.000 | 1.000 | 0.000 | 1.000 |
| replace_k16_spread | inverted_state_bridge | copy_initial_changed__static_copy_unchanged | 0.056 | 0.056 | 1.000 | 0.000 | 1.000 | 0.000 |
| replace_k16_spread | inverted_state_bridge | static_slot_step282_switch | 0.500 | 0.500 | 0.500 | 0.500 | 0.000 | 0.000 |
| replace_k16_spread | inverted_state_bridge | oracle_event_role_state | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| replace_k16_spread | heldheld_only | anti_copy_changed__static_copy_unchanged | n/a | n/a | n/a | n/a | n/a | n/a |
| replace_k16_spread | heldheld_only | copy_initial_changed__static_copy_unchanged | 0.000 | 0.000 | n/a | n/a | n/a | n/a |
| replace_k16_spread | heldheld_only | static_slot_step282_switch | n/a | n/a | n/a | n/a | n/a | n/a |
| replace_k16_spread | heldheld_only | oracle_event_role_state | 1.000 | 1.000 | n/a | n/a | n/a | n/a |

## Evaluation paired-state-conservation: changed-true accuracy

| condition | rule | changed true acc | pair-both acc | same | opposite | same/st0 | same/st1 | opposite/st0 | opposite/st1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| replace_k00_spread | anti_copy_changed__static_copy_unchanged | 0.500 | 0.500 | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 |
| replace_k00_spread | copy_initial_changed__static_copy_unchanged | 0.500 | 0.500 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| replace_k00_spread | static_slot_step282_switch | 0.500 | 0.500 | 0.500 | 0.500 | 0.000 | 1.000 | 1.000 | 0.000 |
| replace_k00_spread | static_slot_opposite_switch | 0.500 | 0.500 | 0.500 | 0.500 | 1.000 | 0.000 | 0.000 | 1.000 |
| replace_k00_spread | oracle_event_role_state | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| replace_k16_spread | anti_copy_changed__static_copy_unchanged | 0.500 | 0.500 | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 |
| replace_k16_spread | copy_initial_changed__static_copy_unchanged | 0.500 | 0.500 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| replace_k16_spread | static_slot_step282_switch | 0.500 | 0.500 | 0.500 | 0.500 | 0.000 | 1.000 | 1.000 | 0.000 |
| replace_k16_spread | static_slot_opposite_switch | 0.500 | 0.500 | 0.500 | 0.500 | 1.000 | 0.000 | 0.000 | 1.000 |
| replace_k16_spread | oracle_event_role_state | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## Scientific reading

- `replace_k00_spread` is a redundant observational-equivalence condition: anti-copy and oracle event-role agree on opposite-initial changed training rows.
- `replace_k16_spread` breaks the analysis framework for factorial probe static-slot diagonal: in train and eval, both `same|0` and `opposite|1` off-diagonal cells exist.  The analysis framework for factorial probe static-slot switch scores 0.5 on changed rows rather than 1.0.
- The oracle event-role state rule is 1.0 on all state cells; anti-copy and copy-initial are 0.5 on balanced k16 changed rows.  Learned k16 success must therefore be compared to all four pattern×static cells and pair-both conservation, not aggregate state accuracy.

## Files

- full JSON: `experiments/archive/representation_and_objectives/data/balanced_heuristic_baselines/heuristic_baselines_report.json`

# earlier analysis pair-binding heuristic baselines

CPU-only transparent rules on the corrected neutral-block pair-binding substrate.

## Train state shortcuts: true-choice and pair-both

| condition | arm | rule | true state acc | same | opposite | pair-both |
|---|---|---|---:|---:|---:|---:|
| pair_connected | aligned_state_bridge | anti_copy | 0.950 | 0.500 | 1.000 | 0.900 |
| pair_connected | aligned_state_bridge | copy_initial | 0.550 | 1.000 | 0.500 | 0.100 |
| pair_connected | aligned_state_bridge | static_slot_step282_switch | 0.750 | 0.750 | 0.750 | 0.500 |
| pair_connected | aligned_state_bridge | event_role_oracle | 1.000 | 1.000 | 1.000 | 1.000 |
| pair_connected | inverted_state_bridge | anti_copy | 0.950 | 0.500 | 1.000 | 0.900 |
| pair_connected | inverted_state_bridge | copy_initial | 0.550 | 1.000 | 0.500 | 0.100 |
| pair_connected | inverted_state_bridge | static_slot_step282_switch | 0.750 | 0.750 | 0.750 | 0.500 |
| pair_connected | inverted_state_bridge | event_role_oracle | 1.000 | 1.000 | 1.000 | 1.000 |
| pair_connected | heldheld_only | anti_copy | 1.000 | n/a | 1.000 | 1.000 |
| pair_connected | heldheld_only | copy_initial | 0.500 | n/a | 0.500 | 0.000 |
| pair_connected | heldheld_only | static_slot_step282_switch | 0.750 | n/a | 0.750 | 0.500 |
| pair_connected | heldheld_only | event_role_oracle | 1.000 | n/a | 1.000 | 1.000 |
| pair_rewired | aligned_state_bridge | anti_copy | 0.950 | 0.500 | 1.000 | 0.900 |
| pair_rewired | aligned_state_bridge | copy_initial | 0.550 | 1.000 | 0.500 | 0.100 |
| pair_rewired | aligned_state_bridge | static_slot_step282_switch | 0.750 | 0.750 | 0.750 | 0.500 |
| pair_rewired | aligned_state_bridge | event_role_oracle | 1.000 | 1.000 | 1.000 | 1.000 |
| pair_rewired | inverted_state_bridge | anti_copy | 0.950 | 0.500 | 1.000 | 0.900 |
| pair_rewired | inverted_state_bridge | copy_initial | 0.550 | 1.000 | 0.500 | 0.100 |
| pair_rewired | inverted_state_bridge | static_slot_step282_switch | 0.750 | 0.750 | 0.750 | 0.500 |
| pair_rewired | inverted_state_bridge | event_role_oracle | 1.000 | 1.000 | 1.000 | 1.000 |
| pair_rewired | heldheld_only | anti_copy | 1.000 | n/a | 1.000 | 1.000 |
| pair_rewired | heldheld_only | copy_initial | 0.500 | n/a | 0.500 | 0.000 |
| pair_rewired | heldheld_only | static_slot_step282_switch | 0.750 | n/a | 0.750 | 0.500 |
| pair_rewired | heldheld_only | event_role_oracle | 1.000 | n/a | 1.000 | 1.000 |

## Train held-held comparison rules

| condition | arm | rule | all acc | neutral=False | neutral=True |
|---|---|---|---:|---:|---:|
| pair_connected | aligned_state_bridge | geom_same_true_parity | 0.750 | 1.000 | 0.500 |
| pair_connected | aligned_state_bridge | neutral_voice_xor | 0.750 | 0.500 | 1.000 |
| pair_connected | aligned_state_bridge | pair_set_training_oracle | 1.000 | n/a | n/a |
| pair_connected | inverted_state_bridge | geom_same_true_parity | 0.750 | 1.000 | 0.500 |
| pair_connected | inverted_state_bridge | neutral_voice_xor | 0.750 | 0.500 | 1.000 |
| pair_connected | inverted_state_bridge | pair_set_training_oracle | 1.000 | n/a | n/a |
| pair_connected | heldheld_only | geom_same_true_parity | 0.750 | 1.000 | 0.500 |
| pair_connected | heldheld_only | neutral_voice_xor | 0.750 | 0.500 | 1.000 |
| pair_connected | heldheld_only | pair_set_training_oracle | 1.000 | n/a | n/a |
| pair_rewired | aligned_state_bridge | geom_same_true_parity | 0.750 | 1.000 | 0.500 |
| pair_rewired | aligned_state_bridge | neutral_voice_xor | 0.750 | 0.500 | 1.000 |
| pair_rewired | aligned_state_bridge | pair_set_training_oracle | 1.000 | n/a | n/a |
| pair_rewired | inverted_state_bridge | geom_same_true_parity | 0.750 | 1.000 | 0.500 |
| pair_rewired | inverted_state_bridge | neutral_voice_xor | 0.750 | 0.500 | 1.000 |
| pair_rewired | inverted_state_bridge | pair_set_training_oracle | 1.000 | n/a | n/a |
| pair_rewired | heldheld_only | geom_same_true_parity | 0.750 | 1.000 | 0.500 |
| pair_rewired | heldheld_only | neutral_voice_xor | 0.750 | 0.500 | 1.000 |
| pair_rewired | heldheld_only | pair_set_training_oracle | 1.000 | n/a | n/a |

## Eval state shortcuts

| suite | rule | true state acc | same | opposite | pair-both |
|---|---|---:|---:|---:|---:|
| paired_state_conservation | anti_copy | 0.750 | 0.500 | 1.000 | 0.500 |
| paired_state_conservation | copy_initial | 0.750 | 1.000 | 0.500 | 0.500 |
| paired_state_conservation | static_slot_step282_switch | 0.750 | 0.750 | 0.750 | 0.500 |
| paired_state_conservation | event_role_oracle | 1.000 | 1.000 | 1.000 | 1.000 |
| cross_template_state_readout | anti_copy | 0.750 | 0.500 | 1.000 | 0.500 |
| cross_template_state_readout | copy_initial | 0.750 | 1.000 | 0.500 | 0.500 |
| cross_template_state_readout | static_slot_step282_switch | 0.750 | 0.750 | 0.750 | 0.500 |
| cross_template_state_readout | event_role_oracle | 1.000 | 1.000 | 1.000 | 1.000 |

## Scientific reading

Anti-copy remains the expected shortcut: perfect on opposite-initial changed rows and wrong on same-initial changed rows.  The event-role oracle is the only state rule that solves both.  The comparison rows contain both informative parity labels and neutral rank-zero labels; a pair-set-specific training oracle can fit them by using whether a row belongs to the B or R dyad set.  Because the B/R exposure, text, label marginals, and exact pair degrees are matched across conditions, this oracle reflects the intended intervention rather than a simple exposure imbalance.

## Files
- full JSON: `experiments/archive/representation_and_objectives/data/pair_binding_heuristic_baselines/heuristic_baselines.json`

# complete orientation probe results fine-tuning orientation probe

Seeds: [27900, 27901, 27902, 27903, 27904], epochs: 50

| arm | train | hh | mixed | state_chg | state_unchg | pair_both |
|---|---:|---:|---:|---:|---:|---:|
| exposure_only | nan | 0.514±0.031 | 0.503±0.018 | 0.502±0.005 | 0.500±0.005 | 0.255±0.004 |
| heldheld_only | 1.000±0.000 | 0.372±0.019 | 0.495±0.019 | 0.498±0.011 | 0.510±0.010 | 0.255±0.009 |
| aligned_state_bridge | 0.909±0.008 | 0.423±0.071 | 0.492±0.010 | 0.794±0.052 | 0.758±0.063 | 0.559±0.054 |
| inverted_state_bridge | 0.914±0.010 | 0.414±0.051 | 0.499±0.013 | 0.738±0.056 | 0.842±0.064 | 0.592±0.073 |
| neutral_decoupled | 0.995±0.006 | 0.436±0.043 | 0.503±0.019 | 0.566±0.090 | 0.654±0.092 | 0.366±0.063 |
| mixed_event_bridge | 1.000±0.000 | 0.405±0.043 | 0.493±0.015 | 0.517±0.029 | 0.473±0.022 | 0.236±0.017 |

## Critical comparison
- heldheld_only mixed: 0.4949
- aligned mixed: 0.4922
- inverted mixed: 0.4992
- aligned - inverted: -0.0070

- results: `experiments/archive/representation_and_objectives/data/finetune_orientation_probe/finetune_orientation_results.json`

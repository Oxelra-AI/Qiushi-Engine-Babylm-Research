# complete orientation probe results frozen-encoder probe (random)

Init: random, seeds: [27900, 27901, 27902, 27903, 27904], epochs: 80

| arm | train | hh | mixed | state_chg | state_unchg | pair_both |
|---|---:|---:|---:|---:|---:|---:|
| exposure_only | nan | 0.500±0.000 | 0.500±0.000 | 0.500±0.000 | 0.500±0.000 | 0.250±0.000 |
| heldheld_only | 0.864±0.035 | 0.547±0.029 | 0.502±0.012 | 0.506±0.011 | 0.507±0.016 | 0.242±0.016 |
| aligned_state_bridge | 0.661±0.067 | 0.492±0.038 | 0.507±0.017 | 0.491±0.013 | 0.496±0.008 | 0.225±0.018 |
| inverted_state_bridge | 0.701±0.088 | 0.539±0.049 | 0.512±0.012 | 0.503±0.004 | 0.499±0.010 | 0.231±0.013 |
| neutral_decoupled | 0.582±0.040 | 0.505±0.045 | 0.513±0.032 | 0.501±0.017 | 0.494±0.009 | 0.227±0.018 |
| mixed_event_bridge | 0.781±0.052 | 0.517±0.036 | 0.488±0.026 | 0.516±0.008 | 0.484±0.019 | 0.233±0.018 |

## Interpretation guide
- mixed: aligned > 0.5 AND inverted < 0.5 → frozen encoder carries orientable role interface
- mixed: aligned ≈ inverted ≈ heldheld → head calibration absorbs bridge polarity
- mixed: all ≈ 0.5 → frozen representations don't carry role info
- pair_both: conservation of changed+unchanged states

- results: `experiments/archive/representation_and_objectives/data/frozen_random_probe/frozen_random_results.json`

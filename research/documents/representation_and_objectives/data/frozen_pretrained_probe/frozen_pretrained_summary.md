# complete orientation probe results frozen-encoder probe (pretrained)

Init: pretrained, seeds: [27900, 27901, 27902, 27903, 27904], epochs: 80

| arm | train | hh | mixed | state_chg | state_unchg | pair_both |
|---|---:|---:|---:|---:|---:|---:|
| exposure_only | nan | 0.514±0.031 | 0.503±0.018 | 0.502±0.005 | 0.500±0.005 | 0.255±0.004 |
| heldheld_only | 0.960±0.005 | 0.338±0.019 | 0.496±0.002 | 0.500±0.000 | 0.500±0.000 | 0.250±0.000 |
| aligned_state_bridge | 0.778±0.018 | 0.334±0.036 | 0.507±0.011 | 0.517±0.015 | 0.548±0.015 | 0.252±0.011 |
| inverted_state_bridge | 0.769±0.016 | 0.372±0.008 | 0.497±0.008 | 0.519±0.012 | 0.526±0.011 | 0.239±0.022 |
| neutral_decoupled | 0.646±0.005 | 0.397±0.026 | 0.498±0.006 | 0.508±0.015 | 0.494±0.010 | 0.258±0.011 |
| mixed_event_bridge | 0.915±0.009 | 0.391±0.014 | 0.514±0.005 | 0.527±0.020 | 0.486±0.014 | 0.250±0.010 |

## Interpretation guide
- mixed: aligned > 0.5 AND inverted < 0.5 → frozen encoder carries orientable role interface
- mixed: aligned ≈ inverted ≈ heldheld → head calibration absorbs bridge polarity
- mixed: all ≈ 0.5 → frozen representations don't carry role info
- pair_both: conservation of changed+unchanged states

- results: `experiments/archive/representation_and_objectives/data/frozen_pretrained_probe/frozen_pretrained_results.json`

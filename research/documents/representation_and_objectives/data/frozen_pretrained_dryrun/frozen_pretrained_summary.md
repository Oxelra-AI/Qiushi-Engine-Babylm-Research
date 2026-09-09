# complete orientation probe results frozen-encoder probe (pretrained)

Init: pretrained, seeds: [27900], epochs: 5

| arm | train | hh | mixed | state_chg | state_unchg | pair_both |
|---|---:|---:|---:|---:|---:|---:|
| exposure_only | nan | 0.484±0.000 | 0.500±0.000 | 0.512±0.000 | 0.504±0.000 | 0.258±0.000 |
| heldheld_only | 0.625±0.000 | 0.352±0.000 | 0.482±0.000 | 0.500±0.000 | 0.500±0.000 | 0.250±0.000 |
| aligned_state_bridge | 0.575±0.000 | 0.438±0.000 | 0.482±0.000 | 0.500±0.000 | 0.512±0.000 | 0.266±0.000 |
| inverted_state_bridge | 0.537±0.000 | 0.484±0.000 | 0.512±0.000 | 0.500±0.000 | 0.508±0.000 | 0.258±0.000 |
| neutral_decoupled | 0.521±0.000 | 0.344±0.000 | 0.478±0.000 | 0.500±0.000 | 0.500±0.000 | 0.250±0.000 |
| mixed_event_bridge | 0.637±0.000 | 0.414±0.000 | 0.459±0.000 | 0.500±0.000 | 0.500±0.000 | 0.250±0.000 |

## Interpretation guide
- mixed: aligned > 0.5 AND inverted < 0.5 → frozen encoder carries orientable role interface
- mixed: aligned ≈ inverted ≈ heldheld → head calibration absorbs bridge polarity
- mixed: all ≈ 0.5 → frozen representations don't carry role info
- pair_both: conservation of changed+unchanged states

- results: `experiments/archive/representation_and_objectives/data/frozen_pretrained_dryrun/frozen_pretrained_results.json`

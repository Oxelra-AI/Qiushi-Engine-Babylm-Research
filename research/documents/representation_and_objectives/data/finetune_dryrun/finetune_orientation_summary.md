# complete orientation probe results fine-tuning orientation probe

Seeds: [27900], epochs: 3

| arm | train | hh | mixed | state_chg | state_unchg | pair_both |
|---|---:|---:|---:|---:|---:|---:|
| heldheld_only | 0.620±0.000 | 0.422±0.000 | 0.484±0.000 | 0.516±0.000 | 0.488±0.000 | 0.273±0.000 |
| aligned_state_bridge | 0.516±0.000 | 0.406±0.000 | 0.500±0.000 | 0.519±0.000 | 0.492±0.000 | 0.234±0.000 |

## Critical comparison
- heldheld_only mixed: 0.4844
- aligned mixed: 0.5
- inverted mixed: ?

- results: `experiments/archive/representation_and_objectives/data/finetune_dryrun/finetune_orientation_results.json`

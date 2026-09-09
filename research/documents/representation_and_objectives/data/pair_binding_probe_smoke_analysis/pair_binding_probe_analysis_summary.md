# earlier analysis pair-binding learned pilot analysis

This analysis reads split learned outputs from the neutral-block pair-binding substrate.  The decisive interpretation requires local train fit, same-initial changed exact choice, exact pair-both conservation, and signed aligned-vs-inverted mixed margins.

- completed results: 1; errors: 0
- train rows: 3072; eval rows: 2176

## Train fit

| condition | arm | seed | all train acc | informative comparison acc | neutral comparison acc |
|---|---|---:|---:|---:|---:|
| pair_connected | aligned_state_bridge | 28700 | 0.607 | 0.691 | 0.707 |

## Paired-state exact choice

| condition | arm | suite | pattern | changed exact | pair-both | changed margin |
|---|---|---|---|---:|---:|---:|
| pair_connected | aligned_state_bridge | cross_template_state_readout | opposite | 0.672 | 0.516 | 0.032 |
| pair_connected | aligned_state_bridge | cross_template_state_readout | same | 0.312 | 0.312 | -0.032 |
| pair_connected | aligned_state_bridge | paired_state_conservation | opposite | 0.672 | 0.523 | 0.030 |
| pair_connected | aligned_state_bridge | paired_state_conservation | same | 0.305 | 0.289 | -0.032 |

## Direct-anchor versus graph-transfer state readout

| condition | arm | suite | relation family | pattern | changed exact | pair-both |
|---|---|---|---|---|---:|---:|
| pair_connected | aligned_state_bridge | paired_state_conservation | direct_anchor | opposite | 0.734 | 0.531 |
| pair_connected | aligned_state_bridge | paired_state_conservation | direct_anchor | same | 0.234 | 0.234 |
| pair_connected | aligned_state_bridge | paired_state_conservation | graph_transfer | opposite | 0.609 | 0.516 |
| pair_connected | aligned_state_bridge | paired_state_conservation | graph_transfer | same | 0.375 | 0.344 |

## Mixed held-seen signed margins

| condition | arm | acc vs true labels | pred true frac | signed margin mean | true accept | false reject |
|---|---|---:|---:|---:|---:|---:|
| pair_connected | aligned_state_bridge | 0.486 | 0.443 | -0.002 | 0.430 | 0.543 |

| condition | aligned-inverted signed margin | aligned-inverted accuracy |
|---|---:|---:|
| pair_connected | n/a±n/a | n/a±n/a |

## Scientific reading

A pair-binding alignment signal requires pair_connected to outperform pair_rewired specifically on same-initial changed choice and pair-both conservation while also showing an aligned-vs-inverted reversal of mixed held-seen signed margins.  High train fit without those eval readouts reproduces the earlier analysis pattern: local fitting without reusable coordinate transport.  If train fit is poor, optimize the supervised head/epochs before interpreting the substrate.

## Files
- full JSON: `experiments/archive/representation_and_objectives/data/pair_binding_probe_smoke_analysis/pair_binding_probe_analysis.json`

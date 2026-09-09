# earlier analysis pair-binding learned pilot analysis

This analysis reads split learned outputs from the neutral-block pair-binding substrate.  The decisive interpretation requires local train fit, same-initial changed exact choice, exact pair-both conservation, and signed aligned-vs-inverted mixed margins.

- completed results: 4; errors: 0
- train rows: 12288; eval rows: 8704

## Train fit

| condition | arm | seed | all train acc | informative comparison acc | neutral comparison acc |
|---|---|---:|---:|---:|---:|
| pair_connected | aligned_state_bridge | 28700 | 1.000 | 1.000 | 1.000 |
| pair_connected | inverted_state_bridge | 28700 | 1.000 | 1.000 | 1.000 |
| pair_rewired | aligned_state_bridge | 28700 | 1.000 | 1.000 | 1.000 |
| pair_rewired | inverted_state_bridge | 28700 | 1.000 | 1.000 | 1.000 |

## Paired-state exact choice

| condition | arm | suite | pattern | changed exact | pair-both | changed margin |
|---|---|---|---|---:|---:|---:|
| pair_connected | aligned_state_bridge | cross_template_state_readout | opposite | 0.812 | 0.812 | 18.955 |
| pair_connected | aligned_state_bridge | cross_template_state_readout | same | 0.453 | 0.453 | -2.156 |
| pair_connected | aligned_state_bridge | paired_state_conservation | opposite | 0.930 | 0.930 | 21.362 |
| pair_connected | aligned_state_bridge | paired_state_conservation | same | 0.508 | 0.508 | -0.893 |
| pair_connected | inverted_state_bridge | cross_template_state_readout | opposite | 0.766 | 0.750 | 16.177 |
| pair_connected | inverted_state_bridge | cross_template_state_readout | same | 0.234 | 0.234 | -17.073 |
| pair_connected | inverted_state_bridge | paired_state_conservation | opposite | 0.711 | 0.703 | 13.549 |
| pair_connected | inverted_state_bridge | paired_state_conservation | same | 0.281 | 0.281 | -15.429 |
| pair_rewired | aligned_state_bridge | cross_template_state_readout | opposite | 0.781 | 0.781 | 16.801 |
| pair_rewired | aligned_state_bridge | cross_template_state_readout | same | 0.609 | 0.609 | 11.100 |
| pair_rewired | aligned_state_bridge | paired_state_conservation | opposite | 0.766 | 0.758 | 17.088 |
| pair_rewired | aligned_state_bridge | paired_state_conservation | same | 0.594 | 0.594 | 10.723 |
| pair_rewired | inverted_state_bridge | cross_template_state_readout | opposite | 0.797 | 0.766 | 16.745 |
| pair_rewired | inverted_state_bridge | cross_template_state_readout | same | 0.188 | 0.188 | -17.286 |
| pair_rewired | inverted_state_bridge | paired_state_conservation | opposite | 0.742 | 0.719 | 13.632 |
| pair_rewired | inverted_state_bridge | paired_state_conservation | same | 0.273 | 0.273 | -13.124 |

## Direct-anchor versus graph-transfer state readout

| condition | arm | suite | relation family | pattern | changed exact | pair-both |
|---|---|---|---|---|---:|---:|
| pair_connected | aligned_state_bridge | paired_state_conservation | direct_anchor | opposite | 1.000 | 1.000 |
| pair_connected | aligned_state_bridge | paired_state_conservation | direct_anchor | same | 0.969 | 0.969 |
| pair_connected | aligned_state_bridge | paired_state_conservation | graph_transfer | opposite | 0.859 | 0.859 |
| pair_connected | aligned_state_bridge | paired_state_conservation | graph_transfer | same | 0.047 | 0.047 |
| pair_connected | inverted_state_bridge | paired_state_conservation | direct_anchor | opposite | 0.594 | 0.578 |
| pair_connected | inverted_state_bridge | paired_state_conservation | direct_anchor | same | 0.438 | 0.438 |
| pair_connected | inverted_state_bridge | paired_state_conservation | graph_transfer | opposite | 0.828 | 0.828 |
| pair_connected | inverted_state_bridge | paired_state_conservation | graph_transfer | same | 0.125 | 0.125 |
| pair_rewired | aligned_state_bridge | paired_state_conservation | direct_anchor | opposite | 1.000 | 0.984 |
| pair_rewired | aligned_state_bridge | paired_state_conservation | direct_anchor | same | 1.000 | 1.000 |
| pair_rewired | aligned_state_bridge | paired_state_conservation | graph_transfer | opposite | 0.531 | 0.531 |
| pair_rewired | aligned_state_bridge | paired_state_conservation | graph_transfer | same | 0.188 | 0.188 |
| pair_rewired | inverted_state_bridge | paired_state_conservation | direct_anchor | opposite | 0.609 | 0.562 |
| pair_rewired | inverted_state_bridge | paired_state_conservation | direct_anchor | same | 0.422 | 0.422 |
| pair_rewired | inverted_state_bridge | paired_state_conservation | graph_transfer | opposite | 0.875 | 0.875 |
| pair_rewired | inverted_state_bridge | paired_state_conservation | graph_transfer | same | 0.125 | 0.125 |

## Mixed held-seen signed margins

| condition | arm | acc vs true labels | pred true frac | signed margin mean | true accept | false reject |
|---|---|---:|---:|---:|---:|---:|
| pair_connected | aligned_state_bridge | 0.436 | 0.865 | -0.939 | 0.801 | 0.070 |
| pair_connected | inverted_state_bridge | 0.428 | 0.826 | -0.572 | 0.754 | 0.102 |
| pair_rewired | aligned_state_bridge | 0.531 | 0.430 | 0.571 | 0.461 | 0.602 |
| pair_rewired | inverted_state_bridge | 0.518 | 0.783 | 0.068 | 0.801 | 0.234 |

| condition | aligned-inverted signed margin | aligned-inverted accuracy |
|---|---:|---:|
| pair_connected | -0.368±0.000 | 0.008±0.000 |
| pair_rewired | 0.504±0.000 | 0.014±0.000 |

## Scientific reading

A pair-binding alignment signal requires pair_connected to outperform pair_rewired specifically on same-initial changed choice and pair-both conservation while also showing an aligned-vs-inverted reversal of mixed held-seen signed margins.  High train fit without those eval readouts reproduces the earlier analysis pattern: local fitting without reusable coordinate transport.  If train fit is poor, optimize the supervised head/epochs before interpreting the substrate.

## Files
- full JSON: `experiments/archive/representation_and_objectives/data/pair_binding_probe_analysis/pair_binding_probe_analysis.json`

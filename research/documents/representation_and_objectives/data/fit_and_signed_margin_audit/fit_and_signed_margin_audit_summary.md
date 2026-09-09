# coordinate connected experience route audit of earlier analysis local fit and signed mixed margins

This CPU-only audit reads the merged earlier analysis fast-probe logits. It verifies whether the balanced k16 informative bridge rows were fit locally and replaces mixed true-row acceptance with all-row label-signed margins against the true coordinate.

- train rows analyzed: 23424
- eval rows analyzed: 48384

## Training fit

All condition/arm/seed groups reached 1.0 train accuracy over the model-consumed rows. The table below focuses on the held bridge state rows where `initial_pattern` is explicit; common seen-coordinate rows have `initial_pattern=None`.

| condition | arm | pattern | exact changed choice | exact pair-both | n changed choices | n pair records |
|---|---|---|---:|---:|---:|---:|
| replace_k00_spread | aligned_state_bridge | None | 1.000 | 1.000 | 768 | 768 |
| replace_k00_spread | aligned_state_bridge | opposite | 1.000 | 1.000 | 96 | 96 |
| replace_k00_spread | inverted_state_bridge | None | 1.000 | 1.000 | 768 | 768 |
| replace_k00_spread | inverted_state_bridge | opposite | 1.000 | 1.000 | 96 | 96 |
| replace_k00_spread | heldheld_only | None | 1.000 | 1.000 | 768 | 768 |
| replace_k16_spread | aligned_state_bridge | None | 1.000 | 1.000 | 768 | 768 |
| replace_k16_spread | aligned_state_bridge | opposite | 1.000 | 1.000 | 48 | 48 |
| replace_k16_spread | aligned_state_bridge | same | 1.000 | 1.000 | 48 | 48 |
| replace_k16_spread | inverted_state_bridge | None | 1.000 | 1.000 | 768 | 768 |
| replace_k16_spread | inverted_state_bridge | opposite | 1.000 | 1.000 | 48 | 48 |
| replace_k16_spread | inverted_state_bridge | same | 1.000 | 1.000 | 48 | 48 |
| replace_k16_spread | heldheld_only | None | 1.000 | 1.000 | 768 | 768 |

This supports the restricted wording: the same-initial informative training rows were fit locally in k16, but their event-role solution did not extend to disjoint held evaluation fillers.

## Evaluation state exact choice

| condition | arm | pattern | exact changed choice | exact pair-both | changed margin |
|---|---|---|---:|---:|---:|
| replace_k00_spread | aligned_state_bridge | opposite | 0.990 | 0.966 | 25.063 |
| replace_k00_spread | aligned_state_bridge | same | 0.020 | 0.020 | -25.047 |
| replace_k00_spread | inverted_state_bridge | opposite | 0.944 | 0.885 | 23.641 |
| replace_k00_spread | inverted_state_bridge | same | 0.049 | 0.036 | -23.801 |
| replace_k00_spread | heldheld_only | opposite | 0.992 | 0.988 | 23.329 |
| replace_k00_spread | heldheld_only | same | 0.021 | 0.021 | -23.256 |
| replace_k16_spread | aligned_state_bridge | opposite | 0.995 | 0.993 | 20.412 |
| replace_k16_spread | aligned_state_bridge | same | 0.009 | 0.009 | -20.147 |
| replace_k16_spread | inverted_state_bridge | opposite | 0.982 | 0.979 | 20.370 |
| replace_k16_spread | inverted_state_bridge | same | 0.008 | 0.008 | -21.153 |
| replace_k16_spread | heldheld_only | opposite | 0.992 | 0.988 | 23.329 |
| replace_k16_spread | heldheld_only | same | 0.021 | 0.021 | -23.256 |

## Mixed held-seen orientation using signed margins

For each relation-comparison row, signed margin = `margin(true label)`; a real true-coordinate model should have positive signed margin, while a clean inverted-coordinate model would have negative signed margin against these true labels. The decisive quantity is aligned minus inverted, paired by seed.

| condition | arm | acc vs true coordinate | pred true frac | signed margin mean | true-row accept | false-row reject |
|---|---|---:|---:|---:|---:|---:|
| replace_k00_spread | aligned_state_bridge | 0.516 | 0.626 | 0.130 | 0.642 | 0.389 |
| replace_k00_spread | inverted_state_bridge | 0.525 | 0.581 | 0.125 | 0.605 | 0.444 |
| replace_k00_spread | heldheld_only | 0.523 | 0.478 | 0.063 | 0.501 | 0.546 |
| replace_k16_spread | aligned_state_bridge | 0.487 | 0.676 | 0.089 | 0.663 | 0.311 |
| replace_k16_spread | inverted_state_bridge | 0.513 | 0.665 | 0.031 | 0.678 | 0.348 |
| replace_k16_spread | heldheld_only | 0.523 | 0.478 | 0.063 | 0.501 | 0.546 |

| condition | aligned-inverted signed-margin gap | aligned-inverted accuracy gap | aligned-inverted pred-true gap |
|---|---:|---:|---:|
| replace_k00_spread | 0.005±0.082 | -0.009±0.023 | 0.046±0.065 |
| replace_k16_spread | 0.058±0.051 | -0.026±0.022 | 0.010±0.071 |

## Scientific reading

- The k16 bridge did not fail because same-initial informative training rows were ignored; they were fit in the model-consumed training set.
- On disjoint held evaluation fillers, exact changed choice and pair-both remained anti-copy-like: high for opposite-initial rows and near zero for same-initial rows.
- Mixed held-seen transfer shows common acceptance bias but not a polarity-controlled coordinate: the k16 aligned-minus-inverted signed-margin gap is small relative to margins and has unstable sign across seeds.
- The next experiment should therefore manipulate learner-usable interface connectivity or architecture, not add more isolated counterexample rows on the same surface.

## Files
- full JSON: `experiments/archive/representation_and_objectives/data/fit_and_signed_margin_audit/fit_and_signed_margin_audit.json`

# earlier analysis learned balanced k0/k16 analysis

This analysis is generated from per-row learned predictions.  It separates redundant versus independently informative state evidence, direct state-anchor relations versus graph-transfer relations, off-diagonal pattern×static cells, joint changed/unchanged conservation, and signed mixed-relation transfer.

## Run completeness

- all_results entries: 18
- error entries: 0
- per-seed groups analyzed: 18

## Central per-condition metrics

| condition | arm | changed same | changed opposite | pair-both same | pair-both opposite | same/st0 | opposite/st1 | direct-anchor changed | graph-transfer changed | mixed true |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| replace_k00_spread | aligned_state_bridge | 0.031±0.026 | 0.953±0.011 | 0.031±0.026 | 0.932±0.026 | 0.031±0.026 | 0.953±0.013 | 0.497±0.051 | 0.487±0.018 | 0.642±0.032 |
| replace_k00_spread | inverted_state_bridge | 0.000 | 0.875±0.088 | 0.000 | 0.823±0.151 | 0.000 | 0.875±0.088 | 0.427±0.015 | 0.448±0.074 | 0.605±0.045 |
| replace_k00_spread | heldheld_only | 0.016±0.013 | 0.953±0.013 | 0.016±0.013 | 0.953±0.013 | 0.016±0.013 | 0.953±0.013 | n/a | 0.484±0.013 | 0.501±0.049 |
| replace_k16_spread | aligned_state_bridge | 0.049±0.010 | 0.945±0.019 | 0.049±0.010 | 0.940±0.013 | 0.042±0.015 | 0.953±0.013 | 0.492±0.011 | 0.503±0.026 | 0.663±0.055 |
| replace_k16_spread | inverted_state_bridge | 0.016±0.022 | 0.878±0.050 | 0.016±0.022 | 0.875±0.046 | 0.005±0.007 | 0.896±0.039 | 0.406±0.050 | 0.487±0.013 | 0.678±0.007 |
| replace_k16_spread | heldheld_only | 0.016±0.013 | 0.953±0.013 | 0.016±0.013 | 0.953±0.013 | 0.016±0.013 | 0.953±0.013 | n/a | 0.484±0.013 | 0.501±0.049 |

## k16 minus k0 within each arm

| arm | Δ changed same | Δ changed opposite | Δ pair-both same | Δ graph-transfer changed | Δ mixed true |
|---|---:|---:|---:|---:|---:|
| aligned_state_bridge | 0.018±0.016 | -0.008±0.029 | 0.018±0.016 | 0.016±0.042 | 0.021±0.040 |
| inverted_state_bridge | 0.016±0.022 | 0.003±0.137 | 0.016±0.022 | 0.039±0.083 | 0.073±0.045 |
| heldheld_only | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## Signed mixed-relation orientation

Positive means aligned bridge assigns higher probability/accuracy to true mixed statements than inverted bridge.  The slot-orbit symbolic control predicts a large positive gap; chance or tiny gaps mean no reusable signed coordinate reached the comparison surface.

| condition | aligned mixed true − inverted mixed true |
|---|---:|
| replace_k00_spread | 0.036±0.048 |
| replace_k16_spread | -0.016±0.050 |

## Reading the result

- If `replace_k16_spread` raises same-initial changed and pair-both cells relative to k0, including `same|0` and `opposite|1`, while unchanged rows remain strong, then independently varied state evidence induced an event-state rule on this surface.
- If that state improvement is concentrated only in direct-anchor relations (`h0_dax`, `h2_norp`) and graph-transfer relations stay low, the result is relation-local rather than component-wide.
- If aligned−inverted mixed true-statement separation remains near zero, state learning did not produce a reusable signed relation coordinate, even if state rows improve.
- Only the combination of joint state success and signed mixed transfer supports moving from the binary experiment to a counterallocated information-efficiency study over k and relation coverage.

## Files

- input run outputs: `experiments/archive/representation_and_objectives/data/fast_balanced_probe_v2_merged`
- full analysis JSON: `experiments/archive/representation_and_objectives/data/fast_balanced_probe_v2_analysis/balanced_probe_analysis.json`

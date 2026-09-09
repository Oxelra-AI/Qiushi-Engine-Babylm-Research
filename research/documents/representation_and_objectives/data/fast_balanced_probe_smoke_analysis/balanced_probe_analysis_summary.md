# earlier analysis learned balanced k0/k16 analysis

This analysis is generated from per-row learned predictions.  It separates redundant versus independently informative state evidence, direct state-anchor relations versus graph-transfer relations, off-diagonal pattern×static cells, joint changed/unchanged conservation, and signed mixed-relation transfer.

## Run completeness

- all_results entries: 1
- error entries: 0
- per-seed groups analyzed: 1

## Central per-condition metrics

| condition | arm | changed same | changed opposite | pair-both same | pair-both opposite | same/st0 | opposite/st1 | direct-anchor changed | graph-transfer changed | mixed true |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| replace_k16_spread | aligned_state_bridge | 0.531 | 0.617 | 0.508 | 0.578 | 0.547 | 0.578 | 0.672 | 0.477 | 0.941 |

## k16 minus k0 within each arm

| arm | Δ changed same | Δ changed opposite | Δ pair-both same | Δ graph-transfer changed | Δ mixed true |
|---|---:|---:|---:|---:|---:|
| aligned_state_bridge | n/a | n/a | n/a | n/a | n/a |

## Signed mixed-relation orientation

Positive means aligned bridge assigns higher probability/accuracy to true mixed statements than inverted bridge.  The slot-orbit symbolic control predicts a large positive gap; chance or tiny gaps mean no reusable signed coordinate reached the comparison surface.

| condition | aligned mixed true − inverted mixed true |
|---|---:|
| replace_k16_spread | n/a |

## Reading the result

- If `replace_k16_spread` raises same-initial changed and pair-both cells relative to k0, including `same|0` and `opposite|1`, while unchanged rows remain strong, then independently varied state evidence induced an event-state rule on this surface.
- If that state improvement is concentrated only in direct-anchor relations (`h0_dax`, `h2_norp`) and graph-transfer relations stay low, the result is relation-local rather than component-wide.
- If aligned−inverted mixed true-statement separation remains near zero, state learning did not produce a reusable signed relation coordinate, even if state rows improve.
- Only the combination of joint state success and signed mixed transfer supports moving from the binary experiment to a counterallocated information-efficiency study over k and relation coverage.

## Files

- input run outputs: `experiments/archive/representation_and_objectives/data/fast_balanced_probe_smoke`
- full analysis JSON: `experiments/archive/representation_and_objectives/data/fast_balanced_probe_smoke_analysis/balanced_probe_analysis.json`

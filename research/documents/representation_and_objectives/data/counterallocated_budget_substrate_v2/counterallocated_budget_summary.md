# earlier analysis counterallocated information-budget substrate

This CPU-only construction prepares a future information-efficiency experiment only if the balanced k16 learned probe supports the mechanism.  Each condition keeps total state-pair count fixed and changes only which state worlds are independently informative (`initial=same`).

## Panels

| panel | allocations | selected cell counts across allocations | selected pair min..max | anti-copy changed acc values | event-role changed acc values | analysis framework for factorial probe-switch changed acc values |
|---|---:|---|---|---|---|---|
| `single_h0_dax|k01` | 4 | `{'h0_dax|active|st0': 1, 'h0_dax|active|st1': 1, 'h0_dax|passive|st0': 1, 'h0_dax|passive|st1': 1}` | `[1, 1]` | `[0.969, 0.969, 0.969, 0.969]` | `[1.0, 1.0, 1.0, 1.0]` | `[0.469, 0.531, 0.469, 0.531]` |
| `single_h0_dax|k02` | 4 | `{'h0_dax|active|st0': 2, 'h0_dax|active|st1': 2, 'h0_dax|passive|st0': 2, 'h0_dax|passive|st1': 2}` | `[2, 2]` | `[0.938, 0.938, 0.938, 0.938]` | `[1.0, 1.0, 1.0, 1.0]` | `[0.5, 0.5, 0.5, 0.5]` |
| `single_h0_dax|k04` | 4 | `{'h0_dax|active|st0': 4, 'h0_dax|active|st1': 4, 'h0_dax|passive|st0': 4, 'h0_dax|passive|st1': 4}` | `[1, 1]` | `[0.875, 0.875, 0.875, 0.875]` | `[1.0, 1.0, 1.0, 1.0]` | `[0.5, 0.5, 0.5, 0.5]` |
| `single_h0_dax|k08` | 4 | `{'h0_dax|active|st0': 8, 'h0_dax|active|st1': 8, 'h0_dax|passive|st0': 8, 'h0_dax|passive|st1': 8}` | `[2, 2]` | `[0.75, 0.75, 0.75, 0.75]` | `[1.0, 1.0, 1.0, 1.0]` | `[0.5, 0.5, 0.5, 0.5]` |
| `single_h0_dax|k16` | 1 | `{'h0_dax|active|st0': 4, 'h0_dax|active|st1': 4, 'h0_dax|passive|st0': 4, 'h0_dax|passive|st1': 4}` | `[1, 1]` | `[0.5]` | `[1.0]` | `[0.5]` |
| `single_h2_norp|k01` | 4 | `{'h2_norp|active|st0': 1, 'h2_norp|active|st1': 1, 'h2_norp|passive|st0': 1, 'h2_norp|passive|st1': 1}` | `[1, 1]` | `[0.969, 0.969, 0.969, 0.969]` | `[1.0, 1.0, 1.0, 1.0]` | `[0.469, 0.531, 0.469, 0.531]` |
| `single_h2_norp|k02` | 4 | `{'h2_norp|active|st0': 2, 'h2_norp|active|st1': 2, 'h2_norp|passive|st0': 2, 'h2_norp|passive|st1': 2}` | `[2, 2]` | `[0.938, 0.938, 0.938, 0.938]` | `[1.0, 1.0, 1.0, 1.0]` | `[0.5, 0.5, 0.5, 0.5]` |
| `single_h2_norp|k04` | 4 | `{'h2_norp|active|st0': 4, 'h2_norp|active|st1': 4, 'h2_norp|passive|st0': 4, 'h2_norp|passive|st1': 4}` | `[1, 1]` | `[0.875, 0.875, 0.875, 0.875]` | `[1.0, 1.0, 1.0, 1.0]` | `[0.5, 0.5, 0.5, 0.5]` |
| `single_h2_norp|k08` | 4 | `{'h2_norp|active|st0': 8, 'h2_norp|active|st1': 8, 'h2_norp|passive|st0': 8, 'h2_norp|passive|st1': 8}` | `[2, 2]` | `[0.75, 0.75, 0.75, 0.75]` | `[1.0, 1.0, 1.0, 1.0]` | `[0.5, 0.5, 0.5, 0.5]` |
| `single_h2_norp|k16` | 1 | `{'h2_norp|active|st0': 4, 'h2_norp|active|st1': 4, 'h2_norp|passive|st0': 4, 'h2_norp|passive|st1': 4}` | `[1, 1]` | `[0.5]` | `[1.0]` | `[0.5]` |
| `spread|k00` | 1 | `{}` | `[0, 0]` | `[1.0]` | `[1.0]` | `[0.5]` |
| `spread|k01` | 16 | `{'h0_dax|active|st0': 2, 'h0_dax|active|st1': 2, 'h0_dax|passive|st0': 2, 'h0_dax|passive|st1': 2, 'h2_norp|active|st0': 2, 'h2_norp|active|st1': 2, 'h2_norp|passive|st0': 2, 'h2_norp|passive|st1': 2}` | `[2, 2]` | `[0.969, 0.969, 0.969, 0.969, 0.969, 0.969, 0.969, 0.969, 0.969, 0.969, 0.969, 0.969, 0.969, 0.969, 0.969, 0.969]` | `[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]` | `[0.469, 0.531, 0.469, 0.531, 0.469, 0.531, 0.469, 0.531, 0.469, 0.531, 0.469, 0.531, 0.469, 0.531, 0.469, 0.531]` |
| `spread|k02` | 8 | `{'h0_dax|active|st0': 2, 'h0_dax|active|st1': 2, 'h0_dax|passive|st0': 2, 'h0_dax|passive|st1': 2, 'h2_norp|active|st0': 2, 'h2_norp|active|st1': 2, 'h2_norp|passive|st0': 2, 'h2_norp|passive|st1': 2}` | `[2, 2]` | `[0.938, 0.938, 0.938, 0.938, 0.938, 0.938, 0.938, 0.938]` | `[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]` | `[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]` |
| `spread|k04` | 8 | `{'h0_dax|active|st0': 4, 'h0_dax|active|st1': 4, 'h0_dax|passive|st0': 4, 'h0_dax|passive|st1': 4, 'h2_norp|active|st0': 4, 'h2_norp|active|st1': 4, 'h2_norp|passive|st0': 4, 'h2_norp|passive|st1': 4}` | `[4, 4]` | `[0.875, 0.875, 0.875, 0.875, 0.875, 0.875, 0.875, 0.875]` | `[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]` | `[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]` |
| `spread|k08` | 4 | `{'h0_dax|active|st0': 4, 'h0_dax|active|st1': 4, 'h0_dax|passive|st0': 4, 'h0_dax|passive|st1': 4, 'h2_norp|active|st0': 4, 'h2_norp|active|st1': 4, 'h2_norp|passive|st0': 4, 'h2_norp|passive|st1': 4}` | `[1, 1]` | `[0.75, 0.75, 0.75, 0.75]` | `[1.0, 1.0, 1.0, 1.0]` | `[0.5, 0.5, 0.5, 0.5]` |
| `spread|k16` | 4 | `{'h0_dax|active|st0': 8, 'h0_dax|active|st1': 8, 'h0_dax|passive|st0': 8, 'h0_dax|passive|st1': 8, 'h2_norp|active|st0': 8, 'h2_norp|active|st1': 8, 'h2_norp|passive|st0': 8, 'h2_norp|passive|st1': 8}` | `[2, 2]` | `[0.5, 0.5, 0.5, 0.5]` | `[1.0, 1.0, 1.0, 1.0]` | `[0.5, 0.5, 0.5, 0.5]` |

## How to use scientifically

- Do not launch these panels unless the earlier analysis balanced k0/k16 learned run shows that k16 has genuine joint state success and signed mixed-relation transfer.
- For spread panels, interpret k through allocation-averaged behavior, because individual low-k files cannot cover every relation × voice × static-slot cell.
- The single-relation panels test whether an informative state bridge in one relation can propagate through the held-held comparison graph to another relation; failure there with spread success means relation coverage matters.
- Compare every learned result against the transparent shortcut values in this summary and the earlier analysis heuristic-baseline file.  Aggregate accuracy alone is not meaningful.

## Files

- full JSON report: `experiments/archive/representation_and_objectives/data/counterallocated_budget_substrate_v2/counterallocated_budget_report.json`
- condition directories: `experiments/archive/representation_and_objectives/data/counterallocated_budget_substrate_v2/ca_kXX_*`
- source redundant substrate: `data/factorial_initial_ownership/underdetermined/`

# activation marker probe activation query-match marker probe

Models are continuations from query-first answer-only preparation. A shared scalar linear probe is trained on train-entity context-attribute hidden states to select the queried entity's attribute slot, then evaluated on held-entity rows. The probe is representational evidence only, not a causal intervention.

## Means

| arm | epochs | held h4 | held B | train h4 | ctx CE | probe train acc | probe val acc | probe held acc | probe held margin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| context_only_lr_half | 250 | 0.244 | -0.062 | 0.251 | 1.289 | 0.547 | 0.567 | 0.482 | -0.068 |
| direct_full | 250 | 0.389 | +1.787 | 0.794 | 1.247 | 0.916 | 0.899 | 0.484 | +0.059 |
| prep | 0 | 0.887 | +6.639 | 1.000 | 0.000 | 1.000 | 1.000 | 0.854 | +5.832 |
| slot0_static_1over17 | 250 | 0.238 | +0.536 | 0.221 | 1.251 | 0.528 | 0.537 | 0.479 | -0.330 |
| static_1over17 | 250 | 0.863 | +9.360 | 1.000 | 1.256 | 1.000 | 1.000 | 0.833 | +3.817 |

## Per-seed

| seed | arm | held h4 | held B | probe held acc | probe held margin |
|---:|---|---:|---:|---:|---:|
| 43 | context_only_lr_half | 0.246 | -0.060 | 0.674 | +0.609 |
| 43 | direct_full | 0.352 | +1.063 | 0.527 | +2.235 |
| 43 | prep | 0.820 | +5.504 | 0.746 | +4.643 |
| 43 | slot0_static_1over17 | 0.238 | +0.651 | 0.629 | +0.248 |
| 43 | static_1over17 | 0.879 | +8.880 | 0.844 | +4.144 |
| 100 | context_only_lr_half | 0.242 | -0.065 | 0.291 | -0.746 |
| 100 | direct_full | 0.426 | +2.510 | 0.441 | -2.118 |
| 100 | prep | 0.953 | +7.774 | 0.963 | +7.021 |
| 100 | slot0_static_1over17 | 0.238 | +0.420 | 0.330 | -0.907 |
| 100 | static_1over17 | 0.848 | +9.841 | 0.822 | +3.490 |

## Interpretation

If the probe accuracy is high in static_1over17 and low in direct/context_only/slot0, behavioral preservation is associated with retention of a linearly accessible query-match signal at context attribute positions. If the probe remains high in collapsed arms, the marker may be present but decoupled from the answer readout; that would require causal readout or activation interventions. This note records the observed association but does not by itself prove causality.

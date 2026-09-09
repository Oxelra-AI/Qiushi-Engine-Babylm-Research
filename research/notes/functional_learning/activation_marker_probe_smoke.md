# activation marker probe activation query-match marker probe

Models are continuations from query-first answer-only preparation. A shared scalar linear probe is trained on train-entity context-attribute hidden states to select the queried entity's attribute slot, then evaluated on held-entity rows. The probe is representational evidence only, not a causal intervention.

## Means

| arm | epochs | held h4 | held B | train h4 | ctx CE | probe train acc | probe val acc | probe held acc | probe held margin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| direct_full | 5 | 0.646 | +5.304 | 1.000 | 24.614 | 1.000 | 1.000 | 0.844 | +5.416 |
| prep | 0 | 0.958 | +7.646 | 1.000 | 0.000 | 1.000 | 1.000 | 0.977 | +6.875 |
| static_1over17 | 5 | 0.667 | +5.336 | 1.000 | 24.614 | 1.000 | 1.000 | 0.852 | +3.878 |

## Per-seed

| seed | arm | held h4 | held B | probe held acc | probe held margin |
|---:|---|---:|---:|---:|---:|
| 100 | direct_full | 0.646 | +5.304 | 0.844 | +5.416 |
| 100 | prep | 0.958 | +7.646 | 0.977 | +6.875 |
| 100 | static_1over17 | 0.667 | +5.336 | 0.852 | +3.878 |

## Interpretation

If the probe accuracy is high in static_1over17 and low in direct/context_only/slot0, behavioral preservation is associated with retention of a linearly accessible query-match signal at context attribute positions. If the probe remains high in collapsed arms, the marker may be present but decoupled from the answer readout; that would require causal readout or activation interventions. This note records the observed association but does not by itself prove causality.

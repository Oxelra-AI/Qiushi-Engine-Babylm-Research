# Bounded review of graph packet v0
This CPU-only review reads the actual endpoint sweep and dualview route graph packet and asks whether it supplies the missing route2 factorial causal review computation: an entity-keyed, event-conditioned state write and query-conditioned read.
## Main counts
- Rows: `50`; families: `{'entity_state_update': 11, 'polarity_contrast_event': 9, 'quantity_change_compare': 11, 'social_belief_report': 8, 'event_temporal_causal': 11}`.
- Relation types: `{'social': 117, 'state': 15, 'temporal': 24, 'causal': 44, 'spatial': 33, 'possessive': 10, 'conditional': 1, 'hypothetical': 1}`.
- Rows with nonempty `states`: `0`; rows with `state_change` relations: `0`; rows with `action` relations: `0`.
- Entity-state-update family rows: `11`, with state/action/state_change structure: `0`.
- Edge-changed real proxy: `35/50`; similarity summary `{'n': 50, 'mean': 0.9267762243826896, 'median': 0.9881690140845071, 'p10': 0.8550811785929044, 'p90': 1.0}`.

## Surface correspondence and compaction
- Source entity hits, graph vs ordinary compact: `{'n': 50, 'mean': 4.52, 'median': 5.0, 'p10': 3.0, 'p90': 6.0}` vs `{'n': 50, 'mean': 3.36, 'median': 3.0, 'p10': 2.0, 'p90': 5.0}`.
- Relation-keyword coverage, graph vs ordinary compact: `{'n': 50, 'mean': 0.6989255731622311, 'median': 0.7037037037037037, 'p10': 0.5806451612903226, 'p90': 0.8}` vs `{'n': 50, 'mean': 0.3482089365846475, 'median': 0.3333333333333333, 'p10': 0.2222222222222222, 'p90': 0.5}`; graph-minus-ordinary `{'n': 50, 'mean': 0.35071663657758356, 'median': 0.3571428571428572, 'p10': 0.1578947368421053, 'p90': 0.5}`.
- True graph compact has lexical overlap with same-row renamed source above same-family shuffled graph compact: `{'n': 50, 'mean': 0.14380262492303775, 'median': 0.14537815126050418, 'p10': 0.07042253521126761, 'p90': 0.2113385315139701}`. This reflects same-source surface correspondence, not necessarily an event-state write/read mechanism.
- Graph-minus-ordinary lexical overlap to renamed source: `{'n': 50, 'mean': 0.04857326002299577, 'median': 0.047791893526920745, 'p10': -0.05362035225048924, 'p90': 0.13982213438735178}`.

## Scientific reading
The packet does contain real same-source correspondence and graph-constrained compaction preserves more relation/entity vocabulary than ordinary compaction. But it does **not** instantiate the route2 factorial causal review missing computation. The extracted graphs almost never contain explicit before/after state slots, no rows use `state_change` or `action` relation types, and the nominal entity-state-update family is mostly social/causal/spatial factual compaction. Edge-changed controls are weak. Therefore this packet is useful as evidence that a teacher can produce relation-preserving compact views, but it should not receive H100 training as the next mechanism route unless rebuilt into explicit entity-keyed event-state transitions with matched correspondence-destroying controls.

## Consequence
Proceed to specify a minimal main-path memory operation that writes event-result state to an entity key and reads by the queried entity, then test it in a small controlled training screen with held-out entities, verbs, and state families and unchanged natural interaction readouts. Practical endpoint AoA confirmation can proceed separately, but endpoint arithmetic does not solve the missing computation.

JSON: `experiments/archive/representation_and_objectives/data/graph_packet_review/graph_packet_review.json`
Per-row review: `experiments/archive/representation_and_objectives/data/graph_packet_review/per_row_review.jsonl`

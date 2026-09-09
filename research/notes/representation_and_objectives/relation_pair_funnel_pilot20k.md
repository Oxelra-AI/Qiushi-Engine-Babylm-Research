# muon50 gp and relation pair synthesis — legal relation pair funnel

This no-model structural audit tests whether the pvdm compliance and control design legal event labels can support a two-context/two-target interaction objective after fullctx aux budget audit showed local relation compatibility is saturated.

Events kept after legal/control/token/leak filters: **16100** from 65188 raw events in 20000 rows.
Pairs constructed: **6991** using 13982 events; event-use fraction 0.8684.

## Pairs by relation family

| family | pairs |
|---|---:|
| causal_connector | 1791 |
| spatial | 1704 |
| temporal | 1574 |
| negation | 1431 |
| physical_change | 469 |
| comparative | 22 |

## Matching

Match levels: `{'1': 640, '0': 6351}`
Pair cost: `{'n': 6991, 'mean': 0.1999367758546703, 'median': 0.0, 'std': 0.6130582638319603, 'p05': 0.0, 'p95': 2.1, 'min': 0.0, 'max': 2.5}`

Top target multiplicities:

- give: 32
- could: 32
- now: 32
- i'm: 32
- find: 32
- there's: 32
- that's: 32
- it's: 32
- can: 32
- time: 32

## Natural-mask exposure estimate

With ordinary WWM p≈0.15, expected pairs with both target groups naturally masked per epoch ≈ 157.3, over 10 epochs ≈ 1573.0. This only estimates a zero-extra-forward gather path; exact replay is still needed.

Files:
- summary: `experiments/archive/representation_and_objectives/data/relation_pair_funnel_pilot20k/relation_pair_funnel_summary.json`
- pairs: `experiments/archive/representation_and_objectives/data/relation_pair_funnel_pilot20k/relation_pair_pool.jsonl`
- csv: `experiments/archive/representation_and_objectives/data/relation_pair_funnel_pilot20k/relation_pair_pool_summary.csv`

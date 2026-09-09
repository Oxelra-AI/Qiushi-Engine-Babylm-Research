# muon50 gp and relation pair synthesis — active-token legal relation pair funnel

This repairs the earlier raw structural pilot by requiring each event to survive the actual 256-token training view and by saving target token positions/ids from that view.

Segment rows: 20000 rows / 3115669 words starting tail row 255.
Events kept after active-token/control/token/leak filters: **15834**.
Pairs constructed: **6875** using 13750 events; event-use fraction 0.8684.

## Pairs by relation family

| family | pairs |
|---|---:|
| causal_connector | 1767 |
| spatial | 1670 |
| temporal | 1546 |
| negation | 1408 |
| physical_change | 462 |
| comparative | 22 |

## Matching

Match levels: `{'1': 623, '0': 6252}`
Pair cost: `{'n': 6875, 'mean': 0.19786996363636364, 'median': 0.0, 'std': 0.610215994396916, 'p05': 0.0, 'p95': 2.1, 'min': 0.0, 'max': 2.4}`

Top target multiplicities:
- now: 32
- care: 32
- could: 32
- that's: 32
- i'm: 32
- it's: 32
- there's: 32
- find: 32
- knew: 32
- would: 32

## Natural-mask exposure estimate

Ordinary WWM p≈0.15: expected both-target-masked pairs per epoch ≈ 154.7, over 10 epochs ≈ 1546.9. Exact replay remains required.

Files:
- summary: `experiments/archive/representation_and_objectives/data/relation_active_pair_funnel_pilot20k/active_relation_pair_funnel_summary.json`
- pairs: `experiments/archive/representation_and_objectives/data/relation_active_pair_funnel_pilot20k/active_relation_pair_pool.jsonl`
- csv: `experiments/archive/representation_and_objectives/data/relation_active_pair_funnel_pilot20k/active_relation_pair_pool_summary.csv`

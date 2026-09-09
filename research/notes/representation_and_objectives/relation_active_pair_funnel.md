# muon50 gp and relation pair synthesis — active-token legal relation pair funnel

This repairs the earlier raw structural pilot by requiring each event to survive the actual 256-token training view and by saving target token positions/ids from that view.

Segment rows: 64000 rows / 9971289 words starting tail row 255.
Events kept after active-token/control/token/leak filters: **50865**.
Pairs constructed: **18651** using 37302 events; event-use fraction 0.7334.

## Pairs by relation family

| family | pairs |
|---|---:|
| causal_connector | 5340 |
| spatial | 4585 |
| temporal | 3826 |
| negation | 3538 |
| physical_change | 1285 |
| comparative | 77 |

## Matching

Match levels: `{'0': 17465, '1': 1186}`
Pair cost: `{'n': 18651, 'mean': 0.13989732454023912, 'median': 0.0, 'std': 0.5190151773273075, 'p05': 0.0, 'p95': 2.1, 'min': 0.0, 'max': 2.517}`

Top target multiplicities:
- give: 32
- table: 32
- real: 32
- five: 32
- live: 32
- actually: 32
- really: 32
- well: 32
- show: 32
- city: 32

## Natural-mask exposure estimate

Ordinary WWM p≈0.15: expected both-target-masked pairs per epoch ≈ 419.6, over 10 epochs ≈ 4196.5. Exact replay remains required.

Files:
- summary: `experiments/archive/representation_and_objectives/data/relation_active_pair_funnel/active_relation_pair_funnel_summary.json`
- pairs: `experiments/archive/representation_and_objectives/data/relation_active_pair_funnel/active_relation_pair_pool.jsonl`
- csv: `experiments/archive/representation_and_objectives/data/relation_active_pair_funnel/active_relation_pair_pool_summary.csv`

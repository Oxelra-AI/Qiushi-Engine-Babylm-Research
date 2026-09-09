# matched graph transfer probe result matched probe: chck82 vs chck100

Common items: 10

- lift_delta_structured_minus_neutral: chck82 mean 0.8111348986625672, chck100 mean 0.8634558916091919, 82-100 mean -0.052320992946624754, range [-0.6024150848388672, 0.756342887878418]
- sensitivity_delta_structured_minus_reversed: chck82 mean 1.4901619791984557, chck100 mean 1.511034369468689, 82-100 mean -0.020872390270233153, range [-0.2548403739929199, 0.5716552734375]
- target_nll_improvement_neutral_minus_structured: chck82 mean 1.580626255273819, chck100 mean 1.6995874524116517, 82-100 mean -0.11896119713783264, range [-0.7067720890045166, 0.43731689453125]
- target_nll_improvement_reversed_minus_structured: chck82 mean -0.16103140711784364, chck100 mean 0.011340188980102538, 82-100 mean -0.17237159609794617, range [-0.5924677848815918, 0.31918907165527344]

Interpretation: if this diagnostic captured the protected 82M relation/state competence, 82M should dominate 100M on structured lift/sensitivity. Similar or higher 100M values indicate the probe is mostly synthetic/query-prior/local relation-word behavior, not the official late-loss phenomenon.

By-item CSV: `experiments/archive/frontier_consolidation/data/matched_graph_transfer_probe/matched_probe_82_100_comparison/matched_probe_82_100_by_item.csv`
JSON: `experiments/archive/frontier_consolidation/data/matched_graph_transfer_probe/matched_probe_82_100_comparison/matched_probe_82_100_comparison.json`

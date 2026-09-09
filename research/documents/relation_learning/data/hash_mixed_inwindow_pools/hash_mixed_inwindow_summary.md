# price tradeoff hash-mixed in-window arm

This arm keeps every selected source in the original MAX changed block and assigns the same-window companion per pair by deterministic hash: exact rotated source fragment for one half, compact rewrite for the other half. The suffix/filler rows and 10M/100M word budgets are inherited from the original MAX family.

- pairs: 33,291
- repeat-assigned pairs: 16,560
- view-assigned pairs: 16,731
- repeat fraction: 0.4974
- paired relation rows: 7,923; suffix/top-up rows: 57,390
- exact 10M pool: True
- row length sequence matches original changed block: True
- training stream written: True

Pre-state joint reading for the held-out compact-rewrite and natural-copy probes. On token-nonoverlap compact-rewrite gain versus CLEAN, a proportional mixture is near −0.034 nats; movement toward original REPEAT means local exact copies capture the readout disproportionately; movement toward original VIEW means content-conditioned source use tolerates substantial exact-copy admixture. A fourth outcome needs both probes: VIEW-like rewrite T together with REPEAT-like natural-copy gain would indicate context-selected dual competence, where the learner uses the companion onset/relation to choose between content and identity readouts rather than averaging them. Intermediate behavior on both probes indicates blended readout.

Metadata: `experiments/archive/relation_learning/data/hash_mixed_inwindow_pools/hash_mixed_inwindow_metadata.json`

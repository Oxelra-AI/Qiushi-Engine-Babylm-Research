# scale1p75 bottleneck and next route scale1.75 score establishment
This note is a research score ledger, not a final deliverable. SuperGLUE uses accuracy except MRPC/QQP use F1.
Summary JSON: `experiments/archive/representation_and_objectives/data/scale1p75_score_collation/scale1p75_score_summary.json`

## Current endpoint arithmetic
- 80M: missing SuperGLUE tasks = []; Overall = 41.77163494053074; scores = {'BLiMP': 68.11, 'Supplement': 62.62, 'EWoK': 49.24, 'Entity': 28.2, 'COMPS': 52.11, 'SuperGLUE': 69.25971446477669, 'GlobalPIQA': 38.105000000000004, 'Reading': 8.3, 'AoA': 0.0}; required SuperGLUE for 41.80 = 69.51499999999999
- 100M: missing SuperGLUE tasks = ['multirc', 'rte', 'wsc', 'mrpc', 'qqp', 'mnli']; Overall = None; scores = {'BLiMP': 68.63, 'Supplement': 62.9, 'EWoK': 49.08, 'Entity': 27.46, 'COMPS': 52.3, 'SuperGLUE': None, 'GlobalPIQA': 36.105000000000004, 'Reading': 8.32, 'AoA': 0.0}; required SuperGLUE for 41.80 = 71.40499999999997

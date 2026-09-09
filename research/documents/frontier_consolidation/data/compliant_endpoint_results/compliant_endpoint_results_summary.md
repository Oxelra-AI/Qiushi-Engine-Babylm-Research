# earlier analysis compliant endpoint result summary

Live Strict-Small target Overall: **41.800000**.
Old non-submittable reference Overall: **42.033135**.

| endpoint | Overall | margin vs 41.8 | delta vs old 42.033 | BLiMP | Supplement | EWoK | Entity | COMPS | SuperGLUE | GlobalPIQA | Reading | AoA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| same_pool_tokenizer | 41.257771 | -0.542229 | -0.775364 | 65.870718 | 61.165661 | 50.393237 | 27.400834 | 52.008345 | 70.279868 | 36.063107 | 8.138168 | 0.000000 |
| bytealphabet_tokenizer | 40.703956 | -1.096044 | -1.329178 | 66.335010 | 59.280593 | 49.837621 | 26.195600 | 51.810982 | 69.174306 | 36.106796 | 7.594698 | 0.000000 |

## Deltas vs old non-submittable 42.033 endpoint

| endpoint | BLiMP | Supplement | EWoK | Entity | COMPS | SuperGLUE | GlobalPIQA | Reading | AoA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| same_pool_tokenizer | -1.001605 | -2.110103 | -3.143338 | -0.344907 | 0.039517 | -0.756182 | 0.441748 | -0.103405 | 0.000000 |
| bytealphabet_tokenizer | -0.537313 | -3.995171 | -3.698955 | -1.550141 | -0.157846 | -1.861744 | 0.485437 | -0.646874 | 0.000000 |

## Byte-alphabet minus spatial repair route status same-pool legal endpoint

Overall delta: **-0.553814**.

| column | bytealphabet - spatial repair route status |
|---|---:|
| BLiMP | 0.464292 |
| Supplement | -1.885068 |
| EWoK | -0.555617 |
| Entity | -1.205233 |
| COMPS | -0.197363 |
| SuperGLUE | -1.105562 |
| GlobalPIQA | 0.043689 |
| Reading | -0.543470 |
| AoA | 0.000000 |

## Evidence files

- `same_pool_tokenizer` summary: `experiments/archive/frontier_consolidation/data/compliant_pristine_collate/complianttok_reinvest_seed43022/pristine_collate_complianttok_reinvest_seed43022_summary.json`
  - inspection: `experiments/archive/frontier_consolidation/data/compliant_retrain_inspection/compliant_retrain_inspection.json` complete=True failed_required=[]
  - training metrics: `experiments/archive/frontier_consolidation/training/runs/complianttok_reinvest_seed43022_r2/scientific_metrics.json` loss_last=2.5525617599487305 checkpoints=100
- `bytealphabet_tokenizer` summary: `experiments/archive/frontier_consolidation/data/bytealphatok_pristine_collate/bytealphatok_reinvest_seed43022/pristine_collate_bytealphatok_reinvest_seed43022_summary.json`
  - inspection: `experiments/archive/frontier_consolidation/data/bytealphatok_retrain_inspection/compliant_retrain_inspection.json` complete=True failed_required=[]
  - training metrics: `experiments/archive/frontier_consolidation/training/runs/bytealphatok_reinvest_seed43022/scientific_metrics.json` loss_last=2.571291208267212 checkpoints=100

Full JSON: `experiments/archive/frontier_consolidation/data/compliant_endpoint_results/compliant_endpoint_results_summary.json`

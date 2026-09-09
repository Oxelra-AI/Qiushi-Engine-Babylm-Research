# reopen structured context margin route coherence-margin pilot readiness

Status: **COMPLETE**

| endpoint | model exists | payload exists | cheap7 |
|---|---:|---:|---:|
| cohmargin4M | False | False | NA |
| scale1p75_chck4M_ref | True | False | NA |

## Evaluation commands (not run)

### cohmargin4M

```bash
PYTHONDONTWRITEBYTECODE=1 python -B experiments/archive/frontier_consolidation/scripts/evaluate_compliant_endpoint.py --arm reinvest --run-dir experiments/archive/frontier_consolidation/training/runs/cohmargin4M_scale1p75_seed43022 --target cohmargin4M_scale1p75_seed43022 --endpoint final --out-root experiments/archive/frontier_consolidation/data/cohmargin4M_eval --collate-root experiments/archive/frontier_consolidation/data/cohmargin4M_collate --gpu <free_gpu> --columns BLiMP Supplement EWoK Entity COMPS GlobalPIQA_parallel GlobalPIQA_nonparallel Reading
```

### scale1p75_chck4M_ref

```bash
PYTHONDONTWRITEBYTECODE=1 python -B experiments/archive/frontier_consolidation/scripts/evaluate_compliant_endpoint.py --arm reinvest --run-dir experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder --target scale1p75_standard_chck4M_ref --endpoint chck_4M --out-root experiments/archive/frontier_consolidation/data/scale1p75_chck4M_ref_eval --collate-root experiments/archive/frontier_consolidation/data/scale1p75_chck4M_ref_collate --gpu <free_gpu> --columns BLiMP Supplement EWoK Entity COMPS GlobalPIQA_parallel GlobalPIQA_nonparallel Reading
```


CPU-only helper; no training/evaluation/upload/submission. Use eval commands after managed pilot finishes and a GPU is free.

JSON: `experiments/archive/frontier_consolidation/data/cohmargin_pilot_readiness/cohmargin_pilot_readiness.json`

# private scale endpoint vs mechanism synthesis alpha endpoint readiness

Status: **COMPLETE**
Protected chck82 Overall `41.942481167385985`; cheap7 `43.95944987645173`; SuperGLUE `69.7661813713118`.

| endpoint | status | cheap7 | SG | Overall(AoA0) | delta vs chck82 |
|---|---|---:|---:|---:|---:|
| alpha0p5 | SUPERGLUE_AVAILABLE | 44.1778571429 | 69.78975515726182 | 42.11497279525131 | 0.1724916278653268 |
| alpha0p75 | SUPERGLUE_AVAILABLE | 44.1814285714 | 69.81922238969935 | 42.1210247099666 | 0.17854354258061278 |

## Materialization commands (not run)

### alpha0p5

```bash
PYTHONDONTWRITEBYTECODE=1 python -B experiments/archive/frontier_consolidation/scripts/materialize_truthful_private_scale_carrier.py --label coherent86_alpha0p5 --panel-arm coherent86_private_alpha0p5 --model-dir experiments/archive/frontier_consolidation/training/runs/coherent86_private_scale_0p5/hf_model/final --cheap-payload experiments/archive/frontier_consolidation/data/private_scale_sentinel_eval/coherent86_private_scale_0p5/per_target/coherent86_private_scale_0p5.json --cheap-summary experiments/archive/frontier_consolidation/data/private_scale_sentinel_eval/coherent86_private_scale_0p5/coherent86_private_scale_0p5_summary.json --superglue-payload experiments/archive/frontier_consolidation/data/private_scale_superglue_eval/coherent86_private_scale_0p5/per_target/coherent86_private_scale_0p5_sg.json --superglue-summary experiments/archive/frontier_consolidation/data/private_scale_superglue_summary/coherent86_private_scale_0p5_sg_superglue_summary.json
```

### alpha0p75

```bash
PYTHONDONTWRITEBYTECODE=1 python -B experiments/archive/frontier_consolidation/scripts/materialize_truthful_private_scale_carrier.py --label coherent86_alpha0p75 --panel-arm coherent86_private_alpha0p75 --model-dir models/frontier --cheap-payload experiments/archive/frontier_consolidation/data/private_scale_sentinel_eval/coherent86_private_scale_0p75/per_target/coherent86_private_scale_0p75.json --cheap-summary experiments/archive/frontier_consolidation/data/private_scale_sentinel_eval/coherent86_private_scale_0p75/coherent86_private_scale_0p75_summary.json --superglue-payload experiments/archive/frontier_consolidation/data/private_scale_superglue_eval/coherent86_private_scale_0p75/per_target/coherent86_private_scale_0p75_sg_retry.json --superglue-summary experiments/archive/frontier_consolidation/data/private_scale_superglue_summary_alpha0p75/coherent86_private_scale_0p75_sg_retry_superglue_summary.json
```


This helper never uploads or submits. It only computes endpoint arithmetic and prints local carrier materialization commands when SuperGLUE summaries exist.

JSON: `experiments/archive/frontier_consolidation/data/alpha_endpoint_readiness/alpha_endpoint_readiness.json`

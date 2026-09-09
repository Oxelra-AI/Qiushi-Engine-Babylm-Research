# fixedinit and phase2 dynamics — Fixed-initialization mix25 replication evaluation

Summary JSON: `experiments/archive/compact_experience/data/fixedinit_eval/fixedinit_eval_summary.json`

This evaluates whether the 25% aligned-data gain from mixture eval repaired survives a shared model initialization and shared training RNG on the inherited 8×480/baseline16k WWM recipe.

| target | BLiMP | Supp | EWoK | Entity_fast | Entity_full | COMPS | GPIQA | Reading | equal7_fast | equal7_fullEnt | wproxy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| official_fixedinit | 67.940 | 60.000 | 50.640 | 21.160 | 22.320 | 51.960 | 36.620 | 7.310 | 42.233 | 42.399 | 48.053 |
| mix25_fixedinit | 67.560 | 63.200 | 50.910 | 20.810 | 22.160 | 51.470 | 38.590 | 8.010 | 42.936 | 43.129 | 48.757 |

## Contrasts

- **mix25_fixedinit_minus_official_fixedinit**: BLiMP -0.380, Supplement +3.200, EWoK +0.270, Entity -0.350, Entity_full -0.160, COMPS -0.490, GlobalPIQA_mean +1.970, Reading +0.700, equal7_mean +0.703, equal7_full_entity +0.730, weighted_fast_proxy +0.703

# fastpath entity focus ordinary scale1.75 chck_86M cheap7

Endpoint: `experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_86M`
Checkpoint record: `{'name': 'chck_86M', 'target_word_exposure': 86000000, 'actual_cumulative_word_exposure': 86006729, 'path': 'experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_86M'}`

Cheap7: `43.770714285714284` (delta vs protected chck82 `-0.188736`).

| column | ordinary86 | chck82 | Δ vs chck82 | coherent4M | Δ vs coherent |
|---|---:|---:|---:|---:|---:|
| BLiMP | 68.47 | 68.49128403651986 | -0.021 | 68.52 | -0.050 |
| Supplement | 62.68 | 62.9378112562002 | -0.258 | 63.65 | -0.970 |
| EWoK | 50.14 | 50.05545332553276 | +0.085 | 49.91 | +0.230 |
| Entity | 28.58 | 28.314041930298774 | +0.266 | 28.44 | +0.140 |
| COMPS | 52.29 | 52.19117509443596 | +0.099 | 51.99 | +0.300 |
| GlobalPIQA | 36.135 | 37.57766990291262 | -1.443 | 38.065 | -1.930 |
| Reading | 8.1 | 8.148713589261902 | -0.049 | 8.17 | -0.070 |

Payload: `experiments/archive/representation_and_objectives/data/scale1p75_chck86_cheap7_eval/per_target/scale1p75_chck86_cheap7.json`
JSON: `experiments/archive/representation_and_objectives/data/scale1p75_chck86_cheap7_summary/scale1p75_chck86_cheap7_summary.json`

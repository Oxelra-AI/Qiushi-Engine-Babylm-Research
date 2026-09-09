# shuffle control audit conclusion dual-view cheap evaluation — dualview_mlm_only_20M

Run: `experiments/archive/frontier_consolidation/training/runs/dualview_mlm_only_20M_seed43022`; endpoint: `chck_20M`; charged words: `20000000`.

| Column | score | spatial repair route status 20M | delta |
|---|---:|---:|---:|
| BLiMP | 59.5400 | 59.6900 | -0.1500 |
| Supplement | 58.4300 | 55.4500 | +2.9800 |
| EWoK | 50.1000 | 50.7300 | -0.6300 |
| Entity | 18.3900 | 18.6500 | -0.2600 |
| COMPS | 50.7000 | 50.2600 | +0.4400 |
| GlobalPIQA | 32.6650 | 34.1950 | -1.5300 |
| Reading | 8.6800 | 8.6700 | +0.0100 |
| **cheap7** | **39.786429** | **39.663571** | **+0.122857** |

Payload: `experiments/archive/frontier_consolidation/data/dualview_mlm_only_20m_eval/per_target/dualview_mlm_only_20M.json`
Summary JSON: `experiments/archive/frontier_consolidation/data/dualview_mlm_only_20m_summary/dualview_mlm_only_20M_summary.json`

# earlier analysis fixed retention-vector cross-trajectory test

The earlier analysis probe/readout was applied unchanged to the verified scale1.75 trajectory, the U256 trajectory, and the spatial repair route status legal baseline trajectory. The U256 and spatial repair route status JSON files inherit a stale embedded scale1.75 cheap7 dictionary from the evaluator; this summary ignores those labels except for explicit known 80M/100M annotations.

## Selector outcomes
| selector | scale1.75 | U256 | spatial repair route status |
|---|---|---|---|
| `overall_piece_nll` | chck_100M | chck_100M | chck_100M |
| `macro_source_structure_nll` | chck_100M | chck_100M | chck_100M |
| `source_structure_q90_nll` | chck_100M | chck_100M | chck_100M |
| `forgetting_mean_vs_past_best` | chck_77M | chck_77M | chck_77M |
| `retention_score_mean_plus_forget_plus_0p25std` | chck_100M | chck_100M | chck_100M |

Known scale1.75 official cheap7 peak: **chck_82M = 43.95944987645173**. None of the frozen lower-is-better selectors selects it.

## Scale1.75 correlation sign
- `cheap7_vs_negative_overall_piece_nll`: 0.2304445636180343
- `cheap7_vs_negative_macro_source_structure_nll`: 0.2682065489360738
- `cheap7_vs_negative_source_structure_q90_nll`: 0.15926279110245847
- `cheap7_vs_negative_forgetting_mean_vs_past_best`: -0.8424491296119421
- `cheap7_vs_negative_retention_score_mean_plus_forget_plus_0p25std`: -0.07489466294756265

For `forgetting_mean_vs_past_best`, the evaluator correlates cheap7 with the negative metric because the selector was defined as lower-is-better. The correlation is negative, so higher official cheap7 coincides with **larger**, not smaller, measured corpus-stratum lag from past best. This invalidates the simple preservation interpretation.

## 80M→100M endpoint annotations
| trajectory | cheap7 80M | cheap7 100M | cheap7 delta | overall piece NLL delta | macro source×structure NLL delta | forgetting mean delta |
|---|---:|---:|---:|---:|---:|---:|
| scale1p75 | 43.81214285714286 | 43.543159919261925 | -0.26898293788093497 | -0.032631 | -0.036447 | -0.009894 |
| u256 | 42.98714285714286 | 43.084070958610745 | 0.09692810146788844 | -0.029598 | -0.030117 | -0.002385 |
| spatial repair route status | 42.9486 | 43.0057243457474 | 0.05712434574740399 | -0.029397 | -0.031526 | -0.000890 |

## Scientific consequence
The fixed targeted corpus-retention vector is a useful negative result: it shows that corpus MLM improvement and even balanced source×structure MLM improvement continue toward 100M on all tested trajectories, while the winning scale1.75 official surface peaks at 82M. The frozen selectors should be closed as a stopping rule; the next scientific work should build a different legal, label-free signal for relation/state capability retention or measure official late-window surfaces on an independent trajectory before trusting any proposed selector.

JSON: `experiments/archive/frontier_consolidation/data/retention_cross_trajectory/retention_cross_trajectory_summary.json`

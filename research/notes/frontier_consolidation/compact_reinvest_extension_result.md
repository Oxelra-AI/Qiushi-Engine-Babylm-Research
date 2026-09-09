# compact core full eval projection thresholds compact reinvest extension — positive, not a verdict

Fast no-AoA official-compatible screen on the frozen medium risk-hard clean-Qwen overlay:

| column | compact_view_core | compact_view_reinvest | reinvest - core |
|---|---:|---:|---:|
| BLiMP | 66.93 | 66.63 | -0.30 |
| Supplement | 65.60 | 66.40 | +0.80 |
| EWoK | 51.55 | 53.09 | +1.54 |
| Entity | 27.30 | 28.07 | +0.77 |
| Entity_full | 27.85 | 27.75 | -0.10 |
| COMPS | 52.18 | 51.97 | -0.21 |
| GlobalPIQA_mean | 35.135 | 35.620 | +0.485 |
| Reading | 8.25 | 8.24 | -0.01 |
| equal7_mean | 43.8493 | 44.2886 | +0.4393 |
| equal7_full_entity | 43.9279 | 44.2429 | +0.3150 |

Evidence: `experiments/archive/frontier_consolidation/data/density_noaoa_eval_reinvest/density_noaoa_eval_summary.json`.

Reinvest training completed cleanly (100M, 100 checkpoints, loss_last 2.565617), run dir `experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022`.

## Interpretation

- Reinvesting the compact-view saved word budget into additional distinct compact source-view packets **improves** the fast task-family surface over compact_view_core, mainly through EWoK and Entity and GlobalPIQA, with small BLiMP/COMPS/Reading costs.
- This means the density mechanism composes with modest source-breadth expansion at fixed 10M-word budget; added compact sources did not damage the compact-core signal.
- This is still a no-AoA fast screen. The protected full official-compatible evaluation is on compact_view_core; if it is positive, reinvest becomes a strong second full-eval candidate.
- Reinvest full-eval checkpoints exist (chck_1M..chck_100M), so it can enter the full evaluator without new training.

## Next-work rule (unchanged)

Judge the mechanism first by the protected compact_view_core full evaluation and seed replication. Reinvest is a positive extension to carry to full evaluation after the core candidate, not a substitute for it.

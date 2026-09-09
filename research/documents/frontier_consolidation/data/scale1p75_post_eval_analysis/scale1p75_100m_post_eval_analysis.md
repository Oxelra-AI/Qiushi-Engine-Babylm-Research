# active endpoint consumption and u256 interpretation — scale1.75 100M post-evaluation analysis

Overall: **41.570747**; margin vs 41.8: **-0.229253**; score signal: `below_41p8`.
cheap7: **43.543160**; cheap7 delta vs spatial repair route status: **+0.537435**; Overall delta vs spatial repair route status: **+0.312976**.

| Column | scale1.75 100M | spatial repair route status 100M | Delta |
|---|---:|---:|---:|
| BLiMP | 68.631753 | 65.870718 | +2.761035 |
| Supplement | 62.895171 | 61.165661 | +1.729510 |
| EWoK | 49.080093 | 50.393237 | -1.313144 |
| Entity | 27.464549 | 27.400834 | +0.063715 |
| COMPS | 52.303917 | 52.008345 | +0.295571 |
| SuperGLUE | 69.334599 | 70.279868 | -0.945268 |
| GlobalPIQA | 36.106796 | 36.063107 | +0.043689 |
| Reading | 8.319840 | 8.138168 | +0.181673 |
| AoA | 0.000000 | 0.000000 | +0.000000 |

Validation arithmetic and endpoint/collation checks reported no errors.

Candidate payload backfill: patched=False, path=`experiments/archive/frontier_consolidation/data/scale1p75_100M_repaired_merge/staged_full_eval/per_target/scale1p75_100M_seed43022.json`.
Item-flip analysis JSON: `experiments/archive/frontier_consolidation/data/scale1p75_post_eval_analysis/item_flips/scale1p75_100M_repaired_merge_vs_100M.json`
Item-flip aggregate: {'discrete_reconstructed_mean_delta': 0.5967294182678575, 'discrete_payload_mean_delta': 0.596213207388189, 'total_gain_items': 24195, 'total_loss_items': 22488, 'total_common_items': 170722, 'total_gain_minus_loss': 1707, 'total_gain_minus_loss_pct': 0.9998711355302773}

Consequence: if Overall is above 41.8, this artifact supports the next verification and reproducibility work; if below, use the column and item-family deltas to locate the remaining gap without rerunning inference.

# active endpoint consumption and u256 interpretation — U256 100M post-evaluation analysis

Overall: **41.329242**; margin vs 41.8: **-0.470758**; score signal: `below_41p8`.
cheap7: **43.084071**; cheap7 delta vs spatial repair route status: **+0.078346**; Overall delta vs spatial repair route status: **+0.071471**.

| Column | U256 100M | spatial repair route status 100M | Delta |
|---|---:|---:|---:|
| BLiMP | 66.817235 | 65.870718 | +0.946516 |
| Supplement | 60.847700 | 61.165661 | -0.317961 |
| EWoK | 50.386096 | 50.393237 | -0.007141 |
| Entity | 28.433145 | 27.400834 | +1.032311 |
| COMPS | 51.645492 | 52.008345 | -0.362854 |
| SuperGLUE | 70.374683 | 70.279868 | +0.094815 |
| GlobalPIQA | 36.135922 | 36.063107 | +0.072816 |
| Reading | 7.322907 | 8.138168 | -0.815260 |
| AoA | 0.000000 | 0.000000 | +0.000000 |

Validation arithmetic and endpoint/collation checks reported no errors.

Candidate payload backfill: patched=False, path=`experiments/archive/frontier_consolidation/data/u256_100M_full_eval_hardened/staged_full_eval/per_target/U256_100M_seed43022.json`.
Item-flip analysis JSON: `experiments/archive/frontier_consolidation/data/u256_100m_post_eval_analysis/item_flips/u256_100M_vs_100M.json`
Item-flip aggregate: {'discrete_reconstructed_mean_delta': 0.22728110546359126, 'discrete_payload_mean_delta': 0.2267648945839197, 'total_gain_items': 25790, 'total_loss_items': 25478, 'total_common_items': 170722, 'total_gain_minus_loss': 312, 'total_gain_minus_loss_pct': 0.1827532479703846}

Interpret U256 together with `notes/u256_visibility_profile.md`: the mechanism is small, source-tail-concentrated visibility recovery, not new text or a new objective.

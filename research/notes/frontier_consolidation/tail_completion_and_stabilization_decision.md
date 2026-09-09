# seed43122 tail completion rationale: seed43122 tail completion and first stabilization-average result

## Seed43122 tail completion

Full selected-grid integrity: **True**; valid endpoints: **16/16**.

The completed seed43122 best endpoint is **chck_88M** with cheap7 **43.909286**. The best newly scored tail endpoint is **chck_96M** with cheap7 **43.902143**, which is -0.007143 relative to the observed 88M peak. Thus the missing tail does not contain a higher selected cheap-task peak.

The completed cross-seed structural readout now gives seed43122 peak-vector Pearson 0.709859/Spearman 0.718182 vs reference, aggregate peak shift +4M, mean Δcheap7 +0.161652 across complete common endpoints, but mean Δcheap6(no GlobalPIQA) -0.358542 and mean Δcheap5(no GlobalPIQA/Reading) -0.430375. This preserves the seed43122 route decision reading: coarse family timing is partly related, but the stable broad aggregate remains weaker and the reference signed-transition structure did not recur.

## Same-trajectory 80/82/84 uniform average

Average candidate SHA: `d47c15f96e3424fc0946cfa4e747f9a6374006d9a3cf68d58d5031509585ac50`. The pseudo-run metadata records zero added exposure and a non-chronological scoring wrapper. Selected cheap-task integrity: **True**.

| model/function | cheap7 | cheap6 no GP | cheap5 no GP/Reading | EWoK+Entity | BLiMP | Supplement | EWoK | Entity | COMPS | GlobalPIQA | Reading |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| reference chck82 | 43.958571 | 45.021667 | 52.396000 | 39.185000 | 68.480000 | 62.940000 | 50.060000 | 28.310000 | 52.190000 | 37.580000 | 8.150000 |
| reference chck84 | 44.123571 | 45.124167 | 52.518000 | 39.325000 | 68.250000 | 63.480000 | 50.070000 | 28.580000 | 52.210000 | 38.120000 | 8.155000 |
| avg80/82/84 | 43.964286 | 45.024167 | 52.384000 | 39.090000 | 68.340000 | 63.220000 | 49.730000 | 28.450000 | 52.180000 | 37.605000 | 8.225000 |

| comparison | Δcheap7 | Δcheap6 no GP | Δcheap5 no GP/Reading | ΔEWoK+Entity | ΔGlobalPIQA | ΔReading |
|---|---:|---:|---:|---:|---:|---:|
| avg - reference chck84 | -0.159286 | -0.100000 | -0.134000 | -0.235000 | -0.515000 | 0.070000 |
| avg - reference chck82 | 0.005714 | 0.002500 | -0.012000 | -0.095000 | 0.025000 | 0.075000 |

The average is below the ordinary 84M endpoint by -0.159286 cheap7, -0.100000 cheap6 without GlobalPIQA, -0.134000 cheap5 without GlobalPIQA/Reading, and -0.235000 on EWoK+Entity. Relative to chck82 it is essentially flat on cheap7 (+0.005714) and cheap6 (+0.002500) but still negative on cheap5 (-0.012000) and EWoK+Entity (-0.095000). This fails the pre-stated stabilization target.

## Decision

1. The seed43122 own-peak uncertainty is closed: 88M remains the best selected cheap checkpoint after 92--100M completion. No additional signed transition readout ready/198 run is needed because the true own-peak window is still 86M->88M->90M, already analyzed in seed43122 route decision.
2. The first same-trajectory low-pass average does not stabilize broad competence beyond the ordinary 84M endpoint and should not be extended into an averaging sweep by inertia.
3. The next useful move is critical route review and/or construction of a better stabilization mechanism from the actual failures: broad competence is seed/mask-sensitive, naive late weight averaging smooths away relation/state and GlobalPIQA gains, alpha/private scaling is redistributive, and compact-order training remains paused until directional-fork evidence is available.

No leaderboard submission was performed.

JSON: `experiments/archive/frontier_consolidation/data/tail_and_average_decision/tail_and_average_decision.json`

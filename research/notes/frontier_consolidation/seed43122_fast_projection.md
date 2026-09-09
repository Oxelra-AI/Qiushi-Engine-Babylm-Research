# seed43122 fast localization — seed43122 fast-screen projection before full-surface spending
This is CPU-only arithmetic from the newly readable seed43122 fast/no-AoA file. It does **not** use or infer the running official-rowcount AoA task.
## Seed43122 official-relevant fast columns (SuperGLUE and AoA missing)
- BLiMP: 65.75
- Supplement: 63.2
- EWoK: 49.36
- Entity: 26.29
- COMPS: 51.54
- GlobalPIQA: 35.135
- Reading: 8.865

## Change vs seed43022 compact_view_reinvest
Against seed43022 fast columns (using full Entity where available):
- BLiMP: -0.88
- Supplement: -3.2
- EWoK: -3.73
- Entity: -1.46
- COMPS: -0.43
- GlobalPIQA: -0.485
- Reading: +0.625

Against seed43022 full columns (where direct full values exist):
- BLiMP: -1.12
- Supplement: -0.08
- EWoK: -4.31
- Entity: -1.46
- COMPS: -0.43
- GlobalPIQA: -0.485
- Reading: +0.625

## What would seed43122 need to beat 41.8?

### optimistic_no_fast_to_full_penalty
7-column sum excluding SuperGLUE/AoA: 300.140000
Delta of those 7 columns vs seed43022 full: -7.260000
Required SuperGLUE if AoA=0: 76.060000
Required AoA leaderboard units if SuperGLUE equals seed43022 full (71.381071): 4.678929
Required SuperGLUE to reach clean-Qwen if AoA=0: 71.958616
Required AoA to reach clean-Qwen if SuperGLUE equals seed43022: 0.577545

### seed43022_fast_to_full_calibrated
7-column sum excluding SuperGLUE/AoA: 297.840000
Delta of those 7 columns vs seed43022 full: -9.560000
Required SuperGLUE if AoA=0: 78.360000
Required AoA leaderboard units if SuperGLUE equals seed43022 full (71.381071): 6.978929
Required SuperGLUE to reach clean-Qwen if AoA=0: 74.258616
Required AoA to reach clean-Qwen if SuperGLUE equals seed43022: 2.877545

## Interpretation
Seed43122 is much weaker than seed43022 on the cheap no-AoA surface: BLiMP −0.88, Supplement −3.2, EWoK −3.73, full Entity −1.46, COMPS −0.43, GlobalPIQA −0.485, Reading +0.625 relative to seed43022's fast screen. Applying seed43022's fast→full deltas makes the SOTA threshold stricter, not easier, especially because Supplement full was lower than fast by 3.12.
If 8005-row AoA maps to 0 and SuperGLUE lands anywhere near the observed neighboring values (compact_view_core 68.90, visible leader 69.79, clean-Qwen 70.31, compact_repeat_core 70.62, seed43022 reinvest 71.38), seed43122 is not likely to clear 41.8. It would need SuperGLUE about 76.06 even under the no-penalty optimistic projection, or about 78.36 under the seed43022-calibrated projection. Equivalently, if SuperGLUE equaled the seed43022 reinvest value, it would need positive AoA of about +4.68 to +6.98 leaderboard units.
Therefore the official aggregation and aoa convention resolved interpretation should be refined: after the two-seed AoA recomputation completes, AoA merely staying non-significant for seed43122 is not by itself enough to justify full second-seed official evaluation as a likely SOTA-confirming action. Full seed43122 evaluation becomes a minimum reliable expensive action only if the 8005-row AoA is positive enough to keep an above-41.8 path plausible, or if the research chooses to spend the evaluation cost for scientific replication despite the cheap evidence that the second seed's non-AoA surface is much weaker.

Machine-readable arithmetic: `experiments/archive/frontier_consolidation/data/seed43122_fast_projection/seed43122_fast_projection.json`.

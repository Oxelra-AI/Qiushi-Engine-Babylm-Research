# depth decision calibration — depth decision calibration on the legal40k coordinate

CPU-only. No training or evaluation; no inferred scores. This quantifies the concrete threshold the depth vector must meet.

## Matched baseline: legal40k 8x480 seed43022 Overall 41.140578 (gap to 41.80 = +0.659422)

### Per-column gap to the visible leader
- BLiMP: base 67.9494, leader 67.2000, gap -0.7494
- Supplement: base 60.9656, leader 56.0100, gap -4.9556
- EWoK: base 51.4740, leader 56.0700, gap +4.5960
- Entity: base 27.2037, leader 28.4500, gap +1.2463
- COMPS: base 51.6747, leader 53.5700, gap +1.8953
- SuperGLUE: base 68.5102, leader 69.7900, gap +1.2798
- GlobalPIQA: base 34.6650, leader 39.6700, gap +5.0050
- Reading: base 7.8227, leader 5.4200, gap -2.4027
- AoA: base 0.0000, leader 0.0000, gap +0.0000

### Two-preserve-column ceiling (GlobalPIQA + Entity restored to leader)
- GlobalPIQA base 34.6650 (parallel 22.33, nonparallel 47.00), max gain +5.0050
- Entity base 27.2037, max gain +1.2463
- Overall gain if BOTH restored to leader: +0.6946 -> Overall 41.8352
- Reaches 41.8 from these two columns alone: **True**

### Cheap7 calibration (legal40k coordinate)
- cheap7 mean base: 43.1079
- SuperGLUE base: 68.5102, AoA base: 0.0000
- needed cheap7 mean if SuperGLUE/AoA flat: 43.9557 (gain +0.8478)

### Depth decision thresholds
- crosses_frontier: depth_overall >= 41.80: protect endpoint, reproduce seed43122
- strong_progress: depth_overall >= 41.55 OR matched_delta >= +0.35: seed43122 or focused combination
- flat_or_worse: matched_delta <= -0.10: do not spend second seed on depth alone
- ambiguous_middle: otherwise: use column pattern + evidence, consider experience-utilization U256 next
- matched_baseline_overall: 41.140577774478444

JSON: `experiments/archive/representation_and_objectives/data/depth_decision_calibration/depth_decision_calibration.json`

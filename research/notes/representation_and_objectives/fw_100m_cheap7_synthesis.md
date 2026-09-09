# fw 100m cheap7 synthesis — FW 100M cheap7 synthesis (compact vs row-block vs interleaved)

cheap7 = mean(BLiMP, Supplement, EWoK, Entity, COMPS, GlobalPIQA, Reading). SuperGLUE and AoA are NOT run; this is a cheap decision readout, not a submission Overall.

| arm | BLiMP | Supplement | EWoK | Entity | COMPS | GlobalPIQA | Reading | cheap7 | gap_to_leader_cheap7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| compact (same-proposition recurrence anchor) | 66.87 | 58.86 | 50.25 | 28.36 | 51.84 | 38.635 | 7.455 | 43.1814 | -0.5886 |
| row-block whole-sentence breadth | 67.9 | 59.69 | 50.5 | 23.88 | 51.33 | 37.065 | 8.105 | 42.6386 | -1.1314 |
| interleaved whole-sentence breadth | 66.89 | 56.42 | 51.85 | 25.6 | 51.75 | 41.105000000000004 | 7.9399999999999995 | 43.0793 | -0.6907 |

## Breadth minus compact (column deltas)

- **row-block whole-sentence breadth**: cheap7 -0.5429; BLiMP +1.03, Supplement +0.83, EWoK +0.25, Entity -4.48, COMPS -0.51, GlobalPIQA -1.570 (parallel +4.86, nonparallel -8.00), Reading +0.65
- **interleaved whole-sentence breadth**: cheap7 -0.1021; BLiMP +0.02, Supplement -2.44, EWoK +1.60, Entity -2.76, COMPS -0.09, GlobalPIQA +2.470 (parallel +1.94, nonparallel +3.00), Reading +0.48

## SOTA plausibility

- Leader cheap7 = 43.77; needed cheap7 for Overall 41.80 at leader SuperGLUE 69.79 = 43.7729.
- compact (same-proposition recurrence anchor): cheap7 43.1814; to reach Overall 41.80 would need SuperGLUE+AoA = 73.93 (leader SuperGLUE+AoA ≈ 69.79).
- row-block whole-sentence breadth: cheap7 42.6386; to reach Overall 41.80 would need SuperGLUE+AoA = 77.73 (leader SuperGLUE+AoA ≈ 69.79).
- interleaved whole-sentence breadth: cheap7 43.0793; to reach Overall 41.80 would need SuperGLUE+AoA = 74.64 (leader SuperGLUE+AoA ≈ 69.79).

Files: `experiments/archive/representation_and_objectives/data/fw_100m_cheap7_synthesis/fw_100m_cheap7_synthesis.json`

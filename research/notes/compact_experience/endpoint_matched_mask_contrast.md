# Endpoint-matched masking contrast

Source trajectory files are under `data/external/mask_noaoa_eval`.

All values below are no-AoA screening metrics, not official SuperGLUE+AoA Overall.

## Exposure-matching correction

The initial interpretation omitted exposure-matched `chck_100M` contrasts, especially uniform `chck_100M`. The required measurement is present in the trajectory file:

- uniform_control `chck_100M` equal7 = **43.2114**, equal6 without GlobalPIQA = **44.5625**.
- inverse_priority `chck_100M` equal7 = **43.7050**, equal6 without GlobalPIQA = **45.0500**.
- inverse−uniform at true `chck_100M`: equal7 **+0.4936**, equal6 without GlobalPIQA **+0.4875**.

Thus the true-100M inverse signal is not mostly a GlobalPIQA artifact. Its per-column chck_100M deltas vs uniform are: BLiMP +0.37, Supplement +2.28, EWoK -0.17, Entity +0.17, COMPS +0.25, GlobalPIQA +0.53, Reading +0.025.

The exposure-matched contrast survives when GlobalPIQA is removed. The separate evidence-visible arm does look GlobalPIQA-driven at chck_100M: evidence_visible−uniform equal7 +0.3393 but equal6 -0.1042, with GlobalPIQA +3.0.

## Per-checkpoint equal7

| arm | chck_85M | chck_90M | chck_95M | chck_100M | best | best-selection inflation |
|---|---:|---:|---:|---:|---|---:|
| uniform_control | 42.7021 | 43.4343 | 43.1157 | 43.2114 | chck_90M | +0.2229 |
| inverse_priority | 43.1543 | 43.3286 | 43.8657 | 43.7050 | chck_95M | +0.1607 |
| evidence_visible | 43.6164 | 43.6536 | 43.4343 | 43.5507 | chck_90M | +0.1029 |
| random_priority | 42.8257 | 43.0843 | 43.1164 | 43.0714 | chck_95M | +0.0450 |

## Exposure-matched chck_100M contrasts vs uniform

| arm | equal7 | Δequal7 | equal6 no GPIQA | Δequal6 | ΔGPIQA |
|---|---:|---:|---:|---:|---:|
| uniform_control | 43.2114 | +0.0000 | 44.5625 | +0.0000 | +0.000 |
| inverse_priority | 43.7050 | +0.4936 | 45.0500 | +0.4875 | +0.530 |
| evidence_visible | 43.5507 | +0.3393 | 44.4583 | -0.1042 | +3.000 |
| random_priority | 43.0714 | -0.1400 | 44.4775 | -0.0850 | -0.470 |

## Leader-crossing arithmetic from no-AoA

Needed combined SuperGLUE + AoA leaderboard units = `9*41.8 - 7*equal7`.

- inverse_priority frozen 95M: 69.140
- inverse_priority true 100M: 70.265
- evidence_visible true 100M: 71.345
- uniform_control true 100M: 73.720

## Remaining scientific caveats

The inverse signal still needs the pending complete SuperGLUE+AoA result, seed43122 replication, and later controls that separate linguistic target identity from the geometric side effect of masking more/shorter words at fixed token budget. Corruption-RNG streams also diverge across non-uniform mask modes; a later RNG-decoupled pair can isolate target identity more cleanly.

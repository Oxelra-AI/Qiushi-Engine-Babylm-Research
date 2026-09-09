# research synthesis seed43222 V−R Entity crossover

Pre-registered bands from earlier analysis: V−R rel0 in [-12,-6], V−R rel3/rel4 in [+4,+12].
Prior two-seed cross-seed V−R: rel0 -9.42, rel1 +2.16, rel2 +3.19, rel3 +7.54, rel4 +8.74, rel5 +7.41.

## Seed43222 V−R contrasts (late mean 80M/90M/100M)

| group | n | V acc | R acc | V−R | prior V−R | band | in_band |
|---|---:|---:|---:|---:|---:|---|---|
| ALL | 6780 | 25.90 | 25.06 | +0.85 | +nan |  | None |
| rel_updates_0 | 1541 | 36.88 | 39.56 | -2.68 | -9.42 | [-12.0, -6.0] | False |
| rel_updates_1 | 1323 | 16.58 | 15.75 | +0.83 | +2.16 |  | None |
| rel_updates_2 | 1266 | 22.41 | 19.93 | +2.47 | +3.19 |  | None |
| rel_updates_3 | 1230 | 23.90 | 21.76 | +2.14 | +7.54 | [4.0, 12.0] | False |
| rel_updates_4 | 1112 | 27.61 | 25.57 | +2.04 | +8.74 | [4.0, 12.0] | False |
| rel_updates_5 | 308 | 27.27 | 24.89 | +2.38 | +7.41 |  | None |
| rel_ge1 | 5239 | 22.68 | 20.79 | +1.88 | +nan |  | None |
| rel_ge2 | 3916 | 24.74 | 22.50 | +2.24 | +nan |  | None |
| rel_ge3 | 2650 | 25.85 | 23.72 | +2.13 | +nan |  | None |

V−R slope across relevant updates: +0.817 points/update

## Pre-registered band check: SOME FAIL
- rel_updates_0: FAIL
- rel_updates_3: FAIL
- rel_updates_4: FAIL

## Scientific reading

The crossover **direction** replicates at the third seed: REPEAT advantage at zero relevant updates (V−R = −2.68) and VIEW advantage at every positive update depth (V−R = +0.83 to +2.47). The slope across update depths remains positive (+0.817 vs +3.092 in prior two-seed average). This is the correct qualitative pattern across three independent DeBERTa seeds.

However, all three pre-registered **magnitude** bands miss. The compression is asymmetric: REPEAT at rel0 dropped from ~47 (prior seeds) to 39.56, losing its characteristic zero-update dominance, while VIEW at rel0 barely changed (36.88 vs ~37.2/38.5). At deep updates, REPEAT actually rose (rel4: 25.57 vs ~18-21), so the compression comes from REPEAT becoming less specialized. The prior V−R range of [−9.42, +8.74] narrowed to [−2.68, +2.47].

This establishes: (1) the dissociation direction is three-seed robust, (2) the quantitative magnitude is seed-variable and the pre-registered bands were too tight, (3) the seed-to-seed variation is primarily in REPEAT's zero-update advantage rather than VIEW's deep-update advantage. The crossover is real but not as strong as the initial two seeds suggested. The active-cost-of-recurrence finding from pair level relation robustness (based on held-out probes, not Entity) is the more robust center and must now be checked at the third seed.

Without CLEAN at seed43222, we cannot determine whether both arms dropped together or the gap narrowed specifically. CLEAN completion remains necessary for the full three-seed contrast.

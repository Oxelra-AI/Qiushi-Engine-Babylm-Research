# invariance probe design: Pre-stated decision rule for V-C seed replication

## Why this test is decisive

corrected seedxdata decomposition showed that row-level heldout MLM-loss improvement does NOT carry a reproducible
VIEW-vs-CLEAN signal across seeds (data effect r ≈ 0.025-0.057 reproducibility).
But VIEW arms are reproducibly worse language models of the held-out distribution
(~0.09 nats higher loss than CLEAN at 100M, in both seeds, seed spread <0.01).

The fork: If V-C benchmark movement replicates across seeds while V is a worse LM,
we have a double dissociation (worse distributional fit + better competence).  That means
conversion lives in representation, not in fitting.  If it does not replicate, VIEW
is just a worse LM plus benchmark noise.

## First-basin (seed43022) V-C pattern at 80M/90M/100M

| Benchmark  | Direction     | Range (delta, 80/90/100M)  |
|------------|---------------|----------------------------|
| BLiMP      | ~0 (neg)      | -0.09 to -0.03             |
| Supplement | **POSITIVE**  | +1.67 to +2.30             |
| EWoK       | Mixed         | -0.21 to +0.49             |
| Entity     | **POSITIVE**  | +3.24 to +3.93             |
| COMPS      | **NEGATIVE**  | -0.70 to -0.45             |
| Reading    | **POSITIVE**  | +0.47 to +0.52             |
| cheap6     | positive      | +0.74 to +1.12             |
| exEntity5  | weak positive | +0.24 to +0.56             |

## Existing seed43122 Entity-only results (D_V_43122)

- Entity at 80M: 27.96
- Entity at 100M: 27.42
- (Entity at 90M: to be read)

## Decision rule (stated before seeing numbers)

**Primary test column: Supplement.**  In seed43022, Supplement V-C is the strongest
consistent non-Entity positive effect (+1.67 to +2.30), stable across all three late
checkpoints.

1. **Route continues** if seed43122 Supplement V-C is positive (>+0.5) at ≥2 of 3
   late checkpoints AND at least 2 of 3 other stable patterns replicate:
   - COMPS V-C negative
   - BLiMP V-C near zero (within ±0.5)
   - Reading V-C positive

2. **Route resets** if Supplement V-C flips sign or is near zero (<+0.3) in seed43122,
   regardless of other benchmarks.  In that case, the principle must be rebuilt around
   seed-stable sub-benchmark facts rather than one-seed aggregate substitution effects.

3. **Intermediate**: Supplement V-C is positive but weaker (+0.3 to +0.5) — continue
   with reduced confidence, need the third seed.

## What this does NOT test

- Entity V-C replication is informative but not decisive (already known volatile from
  register experiments).
- exEntity5 aggregates are too noisy for single-seed V-C comparison.
- This does not test whether the representation mechanism is identity-under-variation;
  that is the next measurement if the positive branch survives.

## Heldout loss V-C gap (from shared private fitting index/corrected seedxdata decomposition data)

| Seed  | 60M V-C loss gap | 100M V-C loss gap |
|-------|------------------|-------------------|
| 43022 | +0.078           | +0.097            |
| 43122 | +0.075           | +0.028            |

Both positive (VIEW worse), confirming the double-dissociation setup.  The gap narrows
for seed43122 at 100M, which is interesting but does not change the benchmark test.

## Files to be produced

- `experiments/archive/relation_learning/data/vc_seed_replication` — per-target evaluation JSONs
- `research/notes/relation_learning/vc_seed_replication_result.md` — comparison table

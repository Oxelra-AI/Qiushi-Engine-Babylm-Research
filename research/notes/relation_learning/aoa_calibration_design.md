# aoa calibration design AoA calibration design and route preregistration

Created: 2026-09-08

## Faithful v4 baseline (updated)

The faithful v4 SuperGLUE now has two downstream seeds:
- Seed 42: SuperGLUE 68.946, Overall 42.024
- Seed 43: SuperGLUE 69.040, Overall 42.034
- Mean: SuperGLUE 68.993, Overall 42.029
- Spread: 0.094 SuperGLUE, 0.010 Overall

The faithful v4 bar is Overall ≈ 42.024–42.034. Any v5 must exceed this with both
private seeds and no important column degradation beyond the coherent two-seed band.

## Scientific principle under test

**Acquisition order follows cumulative credited exposure.**

Under uniform WWM masking, a word's credit is proportional to its frequency.
Model AoA is therefore strongly frequency-driven (r=-0.62 with log whole-stream freq).
Child AoA is weakly frequency-driven (r=-0.07 with the same measure), but moderately
correlated with CHILDES-enrichment (r=-0.32 with CHILDES-minus-whole enrichment).

The key question: can a legal training intervention that decouples credit from frequency
move model AoA toward child AoA ordering, yielding a positive official AoA score?

## Two calibration arms (30M words each)

### Arm 1: Row-schedule
- Best legal schedule: `source_childes_taper_rowsort_ratio_f40_d40`
- Reorders rows across 10 passes to front-load CHILDES-enriched material
- Keeps the 100M row multiset byte-identical (but only trains first 30M)
- Same init, masking, LR schedule as v4

### Arm 2: Enrichment-weighted masking
- V4-order stream (first 30M words, identical to v4 training data)
- Modified masking: per-token probability proportional to CHILDES-minus-whole enrichment
- alpha_start = 1.5 (z-score units), linearly tapers to uniform by 67% of training
- Same init, LR schedule as v4; only masking weights differ

### Why both arms
- Arm 1 (row schedule) moves per-word *exposure* timing
- Arm 2 (masking credit) moves per-word *prediction credit* timing
- The design rationale: rows blur words (row enrichment averages over all words in the row),
  but masking targets individual words directly. The ceiling for arm 2 should be higher.
- Both provide independent β measurements

## Measurement plan

After training (12 matched checkpoints at 1M–10M, 20M, 30M):
1. Extract per-word surprisals at each checkpoint using the official AoA evaluator
2. Compute per-word Δsurprisal = arm_surprisal - v4_surprisal at each checkpoint
3. Compute per-word Δlog_exposure from the schedule/masking construction
4. Regress: Δsurprisal = β · Δlog_exposure + intercept
5. Use measured β to recalibrate the across-pass AoA predictor

## Decision rules

1. If measured β is large enough that the recalibrated predictor gives r > 0.15
   for a legal 100M schedule → preregister and launch 100M×2 trunk
2. If measured β gives 0.112 < r < 0.15 → one exploratory 100M trunk seed
3. If measured β gives r < 0.112 → the AoA route closes
4. In all cases: endpoint columns must stay within the coherent two-seed band

## Endpoint conservation prediction

The CHILDES-light final passes should pressure Supplement (Supplement contains
more formal/written-register content). The floor fraction in the schedule controls
how much child-directed text survives in later passes. This is the design variable
that trades AoA improvement against endpoint conservation.

## Cost

~30 minutes per arm on H100 (30M words ≈ 759 training steps × ~2.4s/step).
Surprisal extraction: ~20 min per arm (30 checkpoints × ~40s/checkpoint).
Total: ~2 hours of H100 for both arms including extraction.

## Files

- Prep outputs: `experiments/archive/relation_learning/data/aoa_calibration_prep`
  - `schedule_reordered_30M.jsonl` (192,487 rows, 30M words)
  - `v4_order_30M.jsonl` (194,220 rows, 30M words)
  - `token_enrichment_weights.json` (11,717 active tokens, mean -2.55, std 2.08)
- Training outputs:
  - `experiments/archive/relation_learning/data/schedule_arm` (arm 1)
  - `experiments/archive/relation_learning/data/enrichment_arm` (arm 2)
- Analysis: `experiments/archive/relation_learning/scripts/beta_measurement.py`

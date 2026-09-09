# entity consistency 20m decision — Entity Mention Consistency 20M screen decision

## Result: first mechanism with a growing multi-column net gain over its structure-destroyed control

Direct-checkpoint available coordinates (higher is better everywhere, including Reading). Deltas are consistency minus shuffled_pair.

### 10M
- BLiMP +0.27, Supplement +0.47, Entity +0.06, COMPS +0.23, GlobalPIQA mean +1.97, Reading mean −0.135
- six-column sum +2.865; target Entity+GlobalPIQA +2.03; guard Supp+BLiMP+Reading +0.605

### 20M
- BLiMP +2.04, Supplement −0.98, Entity +0.39, COMPS −0.03, GlobalPIQA mean +3.92, Reading mean −0.21
- six-column sum +5.135; target Entity+GlobalPIQA +4.315; guard Supp+BLiMP+Reading +0.85

### Signal quality
- Net gain **grows** 10M→20M (+2.87 → +5.13), and target columns grow (+2.03 → +4.31).
- Driven mainly by **GlobalPIQA** (mean +3.92 at 20M: parallel +4.85, nonparallel +3.00) and **BLiMP** (+2.04 at 20M).
- Entity itself moves only +0.39 — the mechanism is **not** merely tightening same-form Entity; the transfer is to world-knowledge/commonsense (GlobalPIQA) and grammar (BLiMP).
- Reading mean slightly negative (−0.21); Supplement negative at 20M (−0.98) but BLiMP more than offsets.

### Training validation (already confirmed entity consistency 20m training validation)
- Matched example order, matched pair/candidate telemetry, distinct 10M/20M checkpoints.
- Consistency auxiliary loss much lower than shuffled (learnable signal); MLM loss essentially unchanged.
- Auxiliary loss fell 5.29→1.22 over training; final MLM loss 3.63.

## Cautions before 100M
1. Absolute levels at 20M are still below the protected 100M baseline (e.g. GlobalPIQA parallel 23.30 vs 100M baseline 24.27; BLiMP 61.04 vs 66.76). The 20M screen tests the *mechanism delta vs control*, not the absolute SOTA — the correct comparison is consistency-vs-shuffled at matched exposure, which is strongly positive.
2. Single seed (42/456/789 triple). GlobalPIQA is a smaller eval and can be noisy. A seed replicate at 20M would strengthen the case, but the effect is large (+3.92 mean) and grows with exposure, and BLiMP (a large, stable eval) also improves +2.04.
3. Reading is a current advantage column; the small −0.21 must be watched at 100M.

## Decision
The mechanism clears the stated bar: multi-column net gain (Entity + GlobalPIQA, plus BLiMP), improvement over the structure-destroyed control, and growth with exposure, without collapsing guard columns. Scale to a full 100M consistency run on the protected geometry, plus a matched 100M shuffled_pair control, then evaluate the complete 9/9 coordinate (including full EWoK word_tokenize and repaired AoA) to see whether the mechanism pushes the protected baseline Overall 40.53 toward/above the Strict-Small top row 41.80.

Reference files:
- `data/entity_consistency_available_coordinate_comparison.json`
- `data/entity_consistency_20m_training_validation.json`
- `data/debertav2_b256_true_9of9_coordinate.json`

# new best verification and assessment — Verification of New Best Internal Coordinate

## The Result

`wwm_seed43 chck_80M` achieves Overall 40.7028, exceeding protected seed42 100M (40.5269) by +0.176.

## Official-Compatibility Verification

| Check | Status | Method |
|---|---|---|
| Architecture | ✓ | DeBERTa-v2 8×480, 34,467,424 params — identical to protected |
| Tokenizer | ✓ | baseline16k — identical to protected |
| Data | ✓ | Official BabyLM 2026 Strict corpus only |
| Exposure | ✓ | 80M words ≤ 100M legal maximum, ≤10 epochs |
| Training | ✓ | Flat WWM, seed 43 (legal different seed) |
| 7 zero-shot/Reading | ✓ | Same evaluator pipeline used for all prior work |
| SuperGLUE | ✓ | Official finetune pipeline, non-collapsed predictions (all tasks have mixed labels) |
| AoA | ✓ | Official AoA evaluator over 19-checkpoint ladder, same method as debertav2 b256 aoa local ckpts result |
| Scoring rule | ✓ | Single checkpoint (80M) for all 8 abilities; AoA from trajectory; 9-column equal average |

## Column-by-Column Comparison

| Column | wwm_seed43_80M | protected_seed42_100M | Delta | Leader | Gap to Leader |
|---|---:|---:|---:|---:|---:|
| BLiMP | 66.54 | 66.76 | **-0.22** | 67.20 | -0.66 |
| Supplement | 61.00 | 59.88 | **+1.12** | 56.04 | +4.96 |
| EWoK | 50.44 | 52.19 | **-1.75** | 56.07 | -5.63 |
| Entity | 22.20 | 22.62 | **-0.42** | 28.45 | -6.25 |
| COMPS | 53.00 | 52.19 | **+0.81** | 53.57 | -0.57 |
| SuperGLUE | 68.26 | 68.02 | **+0.23** | 69.79 | -1.54 |
| GlobalPIQA | 37.59 | 35.64 | **+1.96** | 39.67 | -2.07 |
| Reading | 7.30 | 7.62 | **-0.32** | 5.43 | +1.88 |
| AoA | 0.00 | -0.17 | **+0.17** | 0.00 | 0.00 |
| **Overall** | **40.703** | **40.527** | **+0.176** | **41.80** | **-1.098** |

## Honest Assessment

### Sources of improvement
1. **Seed variance**: seed43 is slightly better on Supplement/COMPS/GlobalPIQA
2. **Endpoint selection**: 80M is better than 100M (learning-dynamics insight)
3. **AoA**: 0.0 vs -0.17 (naturally better trajectory alignment in seed43)

### What this is NOT
- NOT a mechanism innovation
- NOT improvement on the target columns (Entity -0.42, EWoK -1.75 are WORSE!)
- NOT close to SOTA (gap = 1.098 Overall)

### Critical insight
The "improvement" is real under official rules but does NOT advance toward SOTA because the persistent deficits (Entity -6.25, EWoK -5.63) are UNCHANGED or WORSE. The next route must create Entity/EWoK ability, not optimize seed/endpoint selection.

### What the 40k tokenizer experiment (official40k oom and accum repair) adds
The existing official40k oom and accum repair DeBERTa-v2 8×480 + official40k at chck_100M shows:
- Entity 22.16 (-0.46 vs protected) — tokenizer HURTS Entity
- GlobalPIQA 36.61 (+0.97 vs protected) — small gain
- Supplement 59.27 (-0.61), BLiMP 66.65 (-0.11)

**Conclusion: 40k tokenizer is NOT the leader's Entity/EWoK differentiator.** The leader's +6.25 Entity must come from data organization or training dynamics.

## Evidence Files
- Complete 9-column coordinate: `data/current_best_internal_coordinate.json`
- Endpoint comparison: `data/wwm_seed43_endpoint_comparison.json`
- AoA result: `data/wwm_seed43_aoa_local_ckpts_result.json`
- SuperGLUE results: `data/wwm_seed43_superglue_{80M,100M}.json`
- official40k oom and accum repair 40k scores: entity 22.16, GlobalPIQA 36.61, BLiMP 66.65, Supplement 59.27, COMPS 52.32

## Next Route
The RMEC mechanism (Relational Masking + Entity-Dense Curriculum) is designed in `plans/rmec_mechanism_construction.md`. It targets the REAL gap (Entity/EWoK) through changes that make the MLM loss minimum REQUIRE entity tracking at convergence, unlike prior attempts that either selected data (wikiauto pair signal analysis), changed masking difficulty generically (AMLM), or used external annotations (WESS/R1).

# routeB closure and scale1p75 eval — Route B closure and scale-1.75 80M full evaluation

## Route B final closure

The legal from-scratch role-switch screen (earlier analysis) and bridge predictive test (routeB bridge predictive value)
together provide conclusive evidence that Route B's sparse clustered WWM packet insertion
does not produce natural hard-surface transfer:

### earlier analysis legal screen result (shared tokenizer, confound-free)
- Role-switch and role-fixed are **identical** on GlobalPIQA_parallel: 23.30097%
- Hard52: both 1/52, role-switch has **worse** mean top-minus-correct (1.9648 vs 1.8610)
- EWoK: role-switch slightly better (0.5097 vs 0.5009) but still worse than anchor stable failures
- Synthetic packet: role-switch learned the grammar (both_correct 0.4269 vs 0.2436)
- Broad cheap7: anchor80 42.89, role-switch 42.90, role-fixed 42.59

### routeB bridge predictive value bridge predictive test
- 7 small-update checkpoints evaluated on GlobalPIQA hard ranks and EWoK reversals
- Bridge movement correlates with hard surface direction (r=0.83 for ΔM vs hard52 margin)
- But **all objectives** (exchange, correct CE, fixed CE) leave hard52 at 0-1/52
- Correct CE moves bridge more than exchange AND moves hard surfaces more
- Exchange at low dose preserves clean MLM but leaves hard surfaces flat
- The bridge is not a discriminative selector for exchange vs ordinary CE

### Conclusion
Route B is closed. Balanced role-switch experience in sparse clustered WWM packets 
teaches synthetic grammar transfer but produces no exchange-specific natural transfer 
to the official hard binding surfaces. The bridge object is preserved as a stress test, 
not as a training target or selector.

## Scale-1.75 80M evaluation (in progress)

adapter scale-1.75 model at 80M reached cheap7 **43.8121**:
- BLiMP 68.11, Supplement 62.62, EWoK 49.24, Entity 28.20, COMPS 52.11, GlobalPIQA 38.105, Reading 8.30
- Architecture: DeBERTa-v2 8×480 + ZeroOutputBottleneckAdapter(128) × scale 1.75
- Parameters: 35,463,008 (16k tokenizer, compliant on 10M pool)
- EWoK drops -1.77 vs aoa mincontext discrepancy audit base (49.24 vs 51.01)

### Overall projection
For Overall ≥ 41.80 with AoA=0 (missing sub-10M checkpoints):
- Required SuperGLUE ≥ 69.5
- (68.11 + 62.62 + 49.24 + 28.20 + 52.11 + SG + 38.105 + 8.30 + 0) / 9 = 41.80
- If SuperGLUE ≈ 71 (typical): Overall ≈ 41.87 → CROSSES 41.80

### Evaluation status
- SuperGLUE + AoA evaluation running on GPU1
- 100M official ladder: ~49M/100M, ETA ~42 min, running on GPU0

### Next steps if 80M crosses 41.80
1. The 100M model will have all required checkpoints for proper AoA
2. Need hard-surface anatomy on the scale-1.75 model (EWoK reversal pattern, GlobalPIQA hard rows)
3. Coordinate with companion analysis on the 100M evaluation and submission preparation
4. If EWoK damage can be repaired without losing other gains, Overall could go higher

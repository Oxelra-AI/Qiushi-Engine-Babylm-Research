# correctness transition analysis Research State Summary

## Compact-View Mechanism Decomposition

The DeBERTa compact-view triangle has been decomposed into two distinct mechanisms via 
91k+ item-level correctness transitions across 5 reliable tasks (Entity excluded due to 
evaluation alignment; COMPS interpreted by subtask due to unequal weighting).

### Component A: Rewrite Marginals / Information Density  
**Measured by**: adjbreak minus repeat  
**Profile**: BLiMP +209 net items, COMPS base +65, wugs +92, EWoK -27  
**Mechanism**: Compact rewrite vocabulary/syntax diversity improves grammatical competence  
**Attention dependence**: None (should transfer to causal models)  

### Component B: Source-Own Adjacency / Correspondence  
**Measured by**: view minus adjbreak  
**Profile**: Supplement +9 (90.3% retained, STRONG quality), EWoK +51 (domain-specific: social-properties +14, physical-dynamics +12, spatial-relations +9), BLiMP -217  
**Mechanism**: Adjacent source-rewrite pairs enable bidirectional alignment that improves relational reasoning/entailment  
**Attention dependence**: Requires bidirectional attention (explains negative causal result)  

### Why these cancel on aggregate  
adjbreak BLiMP advantage (-1.62 equal7 for view-adjbreak) offsets view's Supplement (+3.6) and EWoK (+4.64) advantages. The aggregate +1.2329 view-adjbreak gap is inside the seed band because of this cancellation, NOT because the arms are similar.

## Specific Prediction for Cross-Architecture Transfer
In causal (unidirectional) models:
- Component A survives: rewrite marginals should improve grammar/distributional competence
- Component B vanishes: own-compact should NOT beat adjbreak-compact
- This explains causal compact-vs-repeat near-zero/negative result

## Alpha0.75 Practical Branch Status  
- Genuine 90M/100M states exist via bit-for-bit verified replay (4 checkpoint SHA matches)
- Path: `training/runs/alpha075_exact_replay_from82M_seed43022/hf_model_alpha0p75/`
- AoA and fast evaluation NOT yet run
- This is a parallel practical branch, does not displace scientific work

## Lowest-Cost Decisive Next Tests
1. **Causal own-vs-adjbreak test**: Uses earlier analysis repaired factor-matched design on causal scaffold. Directly tests Component B bidirectional prediction. ~1 GPU for training. Most informative for transferable principle.
2. **Alpha0.75 AoA+fast**: ~1 GPU for evaluation only. Completes submission-tail. Can run in parallel.
3. **Deeper existing-data analysis**: e.g., COMPS base vs wugs family structure, EWoK domain-level correctness curves. CPU only. Lower value than the causal test.
4. **Second DeBERTa seed**: Less informative than causal test because aggregate view-adjbreak averages cancelling effects.

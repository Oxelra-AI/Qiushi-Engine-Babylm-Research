# decision framework and clean collation plan — Decision framework and clean collation plan

## Three Scientific Questions

### Q1: Are the DiD-negative EWoK items near-zero or confident?
**From**: `s46_t22_tool1` margin extraction
- If near-zero: noise → weight averaging or tail consolidation should stabilize
- If confident: systematic divergence → need recipe change or third seed
- **Key metric**: near_zero_frac for pattern 0110 items (268 worst-DiD items)
  - near_zero_frac > 0.5 → noise-dominated → averaging is the right action
  - near_zero_frac < 0.3 → confident errors → averaging won't fully fix
  - Cross-seed avg accuracy on these items: if >50%, averaging helps

### Q2: Does the cross-seed averaged model improve the expected score?
**From**: `s46_t22_tool1` (EWoK accuracy) + `s46_t22_tool2` (full fast surface)
- seed43022 EWoK: 53.54, seed43122 EWoK: 51.89, mean: 52.72
- If cross-seed avg EWoK > 52.72 → averaging improves expected EWoK
- If cross-seed avg equal7 > individual mean (44.29+42.93)/2=43.61 → good sign
- The target: averaged model's projected Overall > 41.8

### Q3: Does the expected multi-seed result clear the leader?
**From**: Q2 + fresh clean collation
- Current: seed43022 Overall 42.033, seed43122 Overall 41.248, mean 41.641
- Mean 41.641 < 41.8 leader → single-seed SOTA, not robust
- If averaged model Overall > 41.8 → robust result from a single averaged model
- This is more defensible than picking the best of two seeds

## Fresh clean pristine collation plan

### Purpose
The current 2×2 uses clean cells that splice COMPACT_EXPERIENCE columns:
- BLiMP, Supplement, Entity, COMPS, GlobalPIQA, Reading: from COMPACT_EXPERIENCE compact density subtask delta
- EWoK: recomputed on pristine 7618-row official data (globalpiqa practical repair blueprint/030)
- SuperGLUE: recomputed from COMPACT_EXPERIENCE saved per-task results using primary-metric convention
- AoA: 0.0 for both seeds (from their respective training trajectories)

A fresh collation would run clean43022 and clean43122 through the complete pristine
official pipeline end-to-end, removing any splicing uncertainty.

### What needs to happen
1. **Zero-shot evaluation** of clean43022 and clean43122 on the pristine BabyLM eval:
   - BLiMP, Supplement, EWoK, Entity, COMPS, GlobalPIQA, Reading
   - Through the same pristine commit (6f825c29) with isolated writable HF caches
   - ~15 min per seed per GPU

2. **SuperGLUE finetuning** for both seeds:
   - 7 tasks × finetuning → ~45-60 min per seed
   - Must use same finetuning hyperparameters as the reinvest evaluation

3. **AoA evaluation** through official min_context=0:
   - Already done for both clean seeds at COMPACT_EXPERIENCE/REPRESENTATION_FRONTIER_STUDIES level
   - Both have AoA 0.0 — this is already established

4. **Collation** through unmodified official collator:
   - Stage all predictions into the collator tree
   - Run the same `calculate_results_from_pred.py` used for reinvest

### Cost estimate
- Zero-shot: 2 × 15 min = 30 min (parallelizable on 2 GPUs)
- SuperGLUE: 2 × 50 min = 100 min (parallelizable)
- Total wall-clock: ~65 min on 2 GPUs
- No training, just evaluation

### Priority
- This is a VALIDATION action, not a score-improvement action
- Should be done AFTER the margin analysis decision
- If the averaged model looks good, it becomes the primary evaluation target
  and clean collation can be done alongside or after its full evaluation

## Leaderboard refresh plan

The current comparison is against `go76dof/wwm_curriculum_simplification_40k` at 41.8,
parsed at babylm2026 live surface. The leaderboard may have changed:
- New submissions may have appeared
- The 41.8 score may have been updated
- Other models may now be the leader

### Action
- `knowledge_request` or direct web check for current BabyLM Strict-Small leaderboard
- If the leader has changed, update the target margin
- This should be done before any submission decision

## Decision tree after margin results

### Path A: Near-zero margins → averaging works
1. Cross-seed avg EWoK > individual mean ✓
2. Full fast surface confirms improvement
3. Run the averaged model through official pristine collation
4. If Overall > 41.8, submit the averaged model
5. Use seed43022's AoA trajectory (both seeds have AoA 0.0)

### Path B: Confident errors → averaging insufficient
1. Cross-seed avg doesn't fully recover
2. Need either: third seed, tail consolidation, or recipe modification
3. Cheapest next action: tail-averaged seed43022 evaluation
4. If tail avg improves seed43022 → submittable improvement
5. If not → third seed with modified init (~4 hours GPU)

### Path C: Mixed → partial improvement
1. Cross-seed avg helps but not enough
2. Combine: tail-averaged seed43022 as primary, cross-seed as backup
3. Full evaluation determines which is strongest

## Immediate CPU-available evidence
After margins arrive, can compute without GPU:
- Margin histograms for key patterns
- Correlation between reinv430/reinv431 margins (shared structure vs noise)
- Domain-level margin statistics
- Predicted accuracy of averaged model from margin interpolation

# complete synthesis Complete Synthesis

## Interpolation diagnostic (completed)

θ(t) = θ_parent + t·(θ_acquired - θ_parent) over 48 private-adapter tensors.
Total displacement L2 = 6.599. Results on 30 held pairs:

| t   | L2    | neutral_full | retain_full | reassign | retain_over_new |
|-----|-------|-------------|-------------|----------|-----------------|
| 0.0 | 0.000 |  0/30       |  0/30       |  1/30    | -6.938          |
| 0.3 | 1.980 |  2/30       |  0/30       |  3/30    | -5.775          |
| 0.5 | 3.300 | 10/30       |  0/30       |  6/30    | -3.154          |
| 0.7 | 4.619 | 17/30       |  2/30       | 12/30    | -0.668          |
| 1.0 | 6.599 | 30/30       | 28/30       | 27/30    | +7.823          |

### Three structural findings

1. **Neutral source binding is gradual**: basic entity-value retrieval without distractors
   emerges progressively with displacement. At t=0.5 (half the displacement), 10/30 pairs
   show full neutral source retrieval.

2. **Distractor resistance has a sharp threshold**: the retain operation (selecting the correct
   source value when an irrelevant update to the other entity introduces a competing new-value
   preference) requires near-full displacement. The retain_correct_over_new margin crosses
   zero only around t=0.7 and reaches +7.8 at t=1.0.

3. **The distractor-induced shift is uniformly large at intermediate t**: about -9 to -10 nats
   of shift toward new-value preference for all t < 0.5. Even at t=1.0, the shift is -3.2 nats.
   The full operation must overcome this shift, not merely soften it.

### Implication

There is no useful "soft" regime where reducing the adapter displacement recovers broad competence
while maintaining contextual selection. The bridge must achieve sufficient displacement through
training, not through interpolation.

## Bridge training (running)

Two arms launched on the interspersed overlay (13,994,705 words):

### Token accounting per macro-update (avg first 20 updates)

| Arm                | Total supervised | Relation tokens | Ordinary MLM | Rel fraction |
|--------------------|-----------------|-----------------|--------------|-------------|
| ordinary_wwm       | 8,932           | 1,197 (MLM)     | 7,735        | 13.4%       |
| answer_allocation  | 7,906           | 154 (answer)    | 7,752        | 1.9%        |

### Cumulative token budgets (354 updates)

- answer_allocation: ~54,693 relation answer tokens + ~2,744,208 ordinary MLM tokens
- The specialist phase used ~60,160 answer tokens (similar total) but with 0% ordinary MLM
- The bridge tests whether ~2% targeted credit in a 98% ordinary stream can drive L2~6.6

### Experimental status at the time
- ordinary_wwm: training in progress.
- answer_allocation: training in progress.
- Checkpoint every 50 updates (50, 100, 150, 200, 250, 300, 354)

## Post-bridge evaluation plan (checkpoint evaluator ready)

1. Run `revision_046c_bridge_eval.py --eval-relation` on all checkpoints from both arms
2. Run `revision_046c_bridge_eval.py --eval-cheap7` on final + promising checkpoints
3. Compare trajectories: does answer_allocation achieve selection while ordinary_wwm does not?
4. Compare Cheap7: does answer_allocation preserve broad competence better than specialist?

## Files
- Interpolation: `data/param_interpolation/interpolation_summary.json`
- Bridge ordinary_wwm: `data/bridge_ordinary_wwm/` (running)
- Bridge answer_allocation: `data/bridge_answer_allocation/` (running)
- Scripts: `scripts/revision_046a_param_interpolation.py`, `revision_046b_bridge_trainer.py`, `revision_046c_bridge_eval.py`
- Notes: `notes/interpolation_and_bridge.md`

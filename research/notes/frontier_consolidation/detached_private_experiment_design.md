# detached private experiment design — Pathway-Separated Detached-Private Experiment

## Scientific Question

Does factoring auxiliary pathway separation from pair sparsity produce broad 
source-free competence gain rather than displacement or coupling?

## Route from dualview pending panel interim Evidence

The broad all-pair dual-view (earlier analysis) scored cheap7 38.76 aligned vs 39.79 mlm_only 
(−1.02). True correspondence was used (aligned > shuffled by +0.22) but the broad 
auxiliary budget tradeoff destroyed general BabyLM competence. The adapter-on/off 
functional probe confirmed the source-free transfer exists locally but is embedded 
in a destructive coupled training design.

The identified cause is: ordinary MLM traversing the evolving adapter
coupled the nominally private learning to the stock trajectory, overwhelming the small 
but real source-free benefit.

## Detached-Path Design

1. **Main MLM**: adapter OFF → stock-only gradients. Stock evolves independently.
2. **Auxiliary**: adapter ON, stock frozen, detached states → adapter-only gradients.
   Source-conditioned and source-free views on sparse high-utility pairs.
3. **Neutrality**: adapter ON, stock frozen, detached stock logits → adapter-only
   gradients. KL(P_stock || P_adapter) on main-batch tokens keeps adapter transparent
   on non-edit content.

## Experiment Arms (20M charged prefix)

| Arm | Trainer | Pair Data | Mode | Status |
|-----|---------|-----------|------|--------|
| sep-sparse20-aligned | separated | top20 | aligned | launched |
| coupled-sparse20-aligned | coupled | top20 | aligned | launched |
| sep-sparse20-shuffled | separated | top20 | shuffled | pending |

### References (already scored)
- mlm_only (dualview panel readiness and budget): cheap7 39.787
- spatial repair route status baseline: cheap7 39.664
- Broad coupled aligned (earlier analysis): cheap7 38.763
- Broad coupled shuffled (earlier analysis): cheap7 38.541

## Decision Logic

1. **sep-sparse20-aligned > mlm_only (39.787)?** → separated design adds value
2. **sep-sparse20-aligned > coupled-sparse20-aligned?** → separation adds value beyond sparsity
3. **coupled-sparse20-aligned > mlm_only?** → sparsity alone helps
4. **sep-sparse20-aligned > sep-sparse20-shuffled?** → correspondence matters in new design
5. **No arm reproduces EWoK/Reading/SuperGLUE damage pattern?** → separation prevents rotation

### Success gate for 100M continuation
Both conditions required:
- sep-sparse20-aligned cheap7 > mlm_only cheap7 (39.787)  
- No EWoK/Reading/Supplement column worse than mlm_only by ≥1.0

## Context (received detached private experiment design)

scale1.75 chck_82M scored Overall **41.942** — above the 41.8 leader.
The 100M endpoint scored only 41.571, confirming the mature trajectory degradation
that our pathway separation directly addresses. From-corpus reproduction running.

## Implementation

- Trainer: `scripts/detached_private_trainer.py`
- Mechanical check: `scripts/mech_check.py` — ALL PASS
  - Init equivalence: 0.0 logit diff (eval mode)
  - Phase A: adapter grad = 0.0
  - Phase B+C: stock grad = 0.0
- All arms use: seed 43/43022/43023, batch256, seq256, 100M LR horizon (2529 steps),
  legal16k tokenizer (SHA 91b775...), compact-view reinvest stream (SHA 3dd19f...)

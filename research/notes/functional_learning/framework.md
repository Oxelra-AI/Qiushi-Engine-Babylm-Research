# accumulated mechanism synthesis framework: Objective-introduction dynamics vs. inherent representational conflict

## Experimental logic

body objective ablation established that non-answer training pressure on the contextual body is sufficient
for held-symbol binding loss (body-only full: held top4 0.953→0.301; body-only context-only:
0.953→0.250; body-only answer-only: 0.953→0.953+).  A competing explanation was then identified: at the switch from answer-only to full objective, context CE starts
at ~26–31 nats (essentially random), producing massive gradient shock.  The first epoch
of body training under full objective drops seed100 held top4 from 0.953 to 0.648 while
context CE goes from random (~26) to 26 (barely changed).  The question is whether this
catastrophic early collapse reflects:

(A) An inherent representational conflict between context prediction and the binding marker, or
(B) A transitional shock from the abrupt introduction of a poorly calibrated new objective.

## Three testable regimes

### Regime 1: Inherent incompatibility
The body cannot simultaneously support query-conditioned binding features and next-token
prediction features at context positions.  No compatible training trajectory exists.
- Predicted accumulated mechanism synthesis outcome: ALL arms (gradual ramp, slow lr, interleaved, calibration)
  collapse to the same held-symbol level as direct_full.  Context CE converges in all arms
  while held binding converges to near-zero.

### Regime 2: Trajectory-dependent compatibility
Compatible representations exist (the body has enough capacity and the features can be
composed), but the standard training trajectory follows a destructive path.  Gentle
introduction avoids the destructive basin and preserves binding while learning context
prediction.
- Predicted outcome: gradual_ctx100_then_full and/or slow_lr25_then_full preserve held
  binding substantially better than direct_full, while context CE converges to similar
  final levels (~1.5 nats).  The dose-response across the ramp phase shows a threshold:
  binding survives below a critical ctx_weight and collapses above it.

### Regime 3: Active reinforcement required
The binding marker can coexist with context prediction only when answer-relevant
reinforcement continues.  Without ongoing maintenance, even gently introduced context
pressure eventually erases the marker.
- Predicted outcome: interleaved_ans_full preserves held binding substantially better
  than direct_full or gradual ramp, because every other epoch reinforces the marker.
  Gradual ramp eventually fails (delayed but not prevented).

### Mixed outcomes
Partial preservation in some arms, seed-dependent effects, or delayed-but-eventual
collapse across all arms.  These point toward intermediate positions: e.g., the conflict
is real but soft (slow and capacity-dependent rather than sharp and immediate).

## Key comparisons at final epoch

1. held_top4 and held_B across arms at be=500.  If any arm > direct_full by a meaningful
   margin while context CE is comparably low, a compatible trajectory exists.
2. Context CE trajectory: does out-only calibration actually reduce ctx_ce before body
   training?  body objective ablation suggests out-only reduces ctx_ce from ~34 to ~28 in 25 epochs —
   too slow.  This predicts calibrate_out50 will fail.
3. Gradual ramp dose-response: plot held_top4 and ctx_ce vs. branch epoch during the
   ramp phase.  At what effective ctx_weight does binding start to collapse?  If collapse
   starts only at ctx_weight>0.5, the binding marker tolerates significant context
   pressure.  If it starts immediately (even at 0.01), the conflict is very sensitive.
4. Interleaved vs. gradual: if interleaved succeeds and gradual fails, the mechanism
   is reinforcement-dependent.  If both fail, the mechanism is capacity-limited.

## Connection to the general principle

Under finite data, models must allocate limited learning to produce both local predictive
competence and reusable transferable computation.  The orbit-binding task instantiates this
as: the model can learn to predict familiar entity-attribute bindings (local prediction)
AND form a counterfactual binding selector usable for novel symbols (reusable computation).

The accumulated mechanism synthesis result classifies this trade-off:
- If compatible trajectories exist (regime 2), then the principle is: data-efficient
  learning requires managed objective transitions.  Reusable computation is fragile under
  abrupt objective changes but robust under careful introduction.
- If reinforcement is required (regime 3), then the principle is: reusable computation
  is actively maintained, not passively preserved.  Multi-objective learning must include
  explicit maintenance pressure for reusable features.
- If the conflict is inherent (regime 1), then the principle is: standard architectures
  under standard training cannot simultaneously optimize for local prediction and broader
  transfer.  Explicit structural separation (modular networks, subspace protection, etc.)
  may be needed.

Each outcome has different implications for practical data-efficient learning:
- Regime 2 → curriculum and objective scheduling matter
- Regime 3 → replay / rehearsal / multi-objective maintenance is essential
- Regime 1 → architecture changes needed for true transferable learning

## Quantitative calibration from existing data

From body objective ablation (seed100, body-only):
- answer-only: ctx_ce unchanged at ~30, held binding stable/improving
- full: ctx_ce drops from 30 to 1.5 in 25 epochs, held binding destroyed
- context-only: ctx_ce drops from 30 to 1.5, held binding destroyed

The gradual ramp at be=1 with ctx_weight=0.01: effective gradient is ~0.01× full.
For comparison, steady-state full at be=25 has ctx_ce≈1.5, gradient proportional to 1.5.
Initial gradient at ctx_weight=0.01 is proportional to 30×0.01=0.3, which is smaller
than steady-state.  So the ramp starts very gently and increases.

If binding survives through the 100-epoch ramp and the subsequent 400 epochs of full
training, we learn that the marker is compatible with fully calibrated context prediction
and the entire collapse was about introduction dynamics.  This would be a strong result.

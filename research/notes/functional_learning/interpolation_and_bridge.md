# complete synthesis Interpolation Diagnostic and Bridge Construction

## Parameter interpolation diagnostic

θ(t) = θ_parent + t·(θ_acquired - θ_parent) over 48 private-adapter tensors (995,584 parameters). Total displacement L2 = 6.599.

### Held three-way discrimination results at each t

| t | L2 | neutral_entity | neutral_full | retain_entity | retain_full | reassign_both | update_correct |
|---|---|---|---|---|---|---|---|
| 0.0 | 0.000 | 0/30 | 0/30 | 1/30 | 0/30 | 1/30 | 30/30 |
| 0.1 | 0.660 | 3/30 | 2/30 | 2/30 | 0/30 | 2/30 | 30/30 |
| 0.2 | 1.320 | 1/30 | 1/30 | 0/30 | 0/30 | 3/30 | 30/30 |
| 0.3 | 1.980 | 2/30 | 2/30 | 2/30 | 0/30 | 3/30 | 30/30 |
| 0.4 | 2.640 | 3/30 | 3/30 | 2/30 | 0/30 | 3/30 | 30/30 |
| 0.5 | 3.300 | 10/30 | 10/30 | 4/30 | 0/30 | 6/30 | 30/30 |
| 0.7 | 4.619 | 17/30 | 17/30 | 9/30 | 2/30 | 12/30 | 29/30 |
| 1.0 | 6.599 | 30/30 | 30/30 | 29/30 | 28/30 | 27/30 | 30/30 |

### Scientific interpretation

1. **No useful intermediate softening regime.** The acquired contextual state-selection operation does not separate smoothly from the parameter displacement. Reducing t does not recover broad competence while preserving selection — it degrades both.

2. **Neutral source retrieval is gradual.** Basic entity-value binding (without distractors) emerges progressively: 0→10→17→30 as t goes 0→0.5→0.7→1.0. The mean neutral cross-source margin rises from +0.063 at t=0 to +8.409 at t=1.0.

3. **Distractor resistance has a sharp threshold.** Retain full source jumps 0→0→0→2→28 over t=0.3→0.4→0.5→0.7→1.0. The retain_correct_over_new margin changes sign around t=0.7 (from -0.67 to +7.82 at t=1.0). This is the most demanding component.

4. **The distractor-induced shift is uniformly large at intermediate t.** `mean_distractor_delta_correct_over_new` is about -9 to -10 nats for t∈[0.1,0.5], meaning an irrelevant update always causes a large preference shift toward the new value. Only at t=1.0 does this shrink to -3.22.

5. **Source reassignment follows the same late-emergence pattern.** Both-query reassignment following: 1→3→6→12→27 across t=0.0→0.4→0.5→0.7→1.0.

### Implication for bridge design

The bridge trainer cannot achieve the selection operation through a softer displacement. It must drive the private-adapter parameters to near-full specialist-level displacement (L2~6.6) while simultaneously maintaining broad competence through ordinary learning. The interspersed schedule provides continuous answer credit throughout training; the question is whether this credit, competing with ordinary MLM, can reach the threshold displacement before the cosine schedule decays the learning rate to near zero.

### Per-layer displacement structure

Layer 0 has the largest down-weight displacement (L2=2.136) while layers 5-6 have the smallest (L2~1.29). This is consistent with earlier interface-reach findings at early Transformer layers.

## Bridge training

Two arms launched on the same interspersed overlay (13,994,705 words, 28,800 relation + 79,927 ordinary rows, 354 word-paced updates):
- `ordinary_wwm`: all rows get standard 15% MLM masking (control)
- `answer_allocation`: relation rows get answer-span masking, ordinary rows get 15% MLM

Same seed, same schedule (cosine LR from update 101/455, peak 5e-5), same private-adapter-only optimization.

Checkpoints saved every 50 updates. Post-training evaluation will score each checkpoint on:
1. Held three-way discrimination + reassignment (relation operation)
2. Entity / fast Cheap7 (broad competence)

Files:
- Interpolation: `data/param_interpolation/interpolation_summary.json`
- Bridge ordinary_wwm: `data/bridge_ordinary_wwm/` (running)
- Bridge answer_allocation: `data/bridge_answer_allocation/` (running)

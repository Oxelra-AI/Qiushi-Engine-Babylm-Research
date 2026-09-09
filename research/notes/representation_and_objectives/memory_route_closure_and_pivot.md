# memory route closure and pivot — TinyMLM Memory Variant Route: Definitive Closure

## Summary

Four distinct raw-token memory architectures have now been tested against
the same counterbalanced entity-event-state binding task. All four failed
at or below vanilla chance, with zero write-permutation sensitivity. The
hard-coordinate ceiling (0.818–0.849) confirms the mechanism works when
entity/event/query roles are supplied; the consistent raw-token failure
confirms these roles cannot be discovered from 12K template sentences at
TinyMLM scale. The route is closed.

## Complete 4-attempt evidence table

| Attempt | Mechanism | held_recomb | affected | unaffected | paraphrase | multi_event | wp_delta_held |
|---------|-----------|-------------|----------|------------|------------|-------------|---------------|
| rawmem  | Soft attention slots + event routing | 0.555 | 0.714 | 0.396 | 0.482 | 0.516 | -0.003 |
| lexmem  | Lexical identity grouping | 0.560 | 0.641 | 0.479 | 0.396 | 0.484 | 0.000 |
| lexrec  | Left-to-right same-token recurrence | 0.500 | 0.516 | 0.484 | 0.490 | 0.906 | +0.003 |
| **discrete** | **Gumbel-Softmax iterative slot competition** | **0.500** | **0.495** | **0.505** | **0.513** | **0.495** | **0.000** |
| vanilla | No memory (baseline) | 0.526–0.578 | 0.583–0.740 | 0.396–0.526 | 0.344–0.591 | 0.484–0.995 | — |
| **hardcoord** | **Template-supplied positions** | **0.818–0.849** | **1.000** | **0.635–0.698** | **0.911–0.919** | **0.531–1.000** | **-0.635** |

## What each attempt tested

1. **rawmem** (rawtoken bridge screen and route judgment): Learned soft attention from slot queries to all positions.
   Failed because soft averaging over positions doesn't produce discrete role decisions.
   Learned a nonselective "action result" bias (affected 0.714, unaffected 0.396).

2. **lexmem** (rawtoken bridge screen and route judgment): Grouped tokens by lexical identity as candidate entity cells.
   Failed because lexical identity alone doesn't distinguish subject from object of action.

3. **lexrec** (rawtoken bridge screen and route judgment): Sequential left-to-right update over same-token positions.
   Failed because sequential order doesn't encode argument structure.
   Multi-event only worked because vanilla memorized it too (0.948).

4. **discrete** (memory route closure and pivot): Gumbel-Softmax forced discrete position selection with
   iterative slot competition over 3 refinement steps, temperature annealed 2.0→0.3.
   **Worst result**: the discrete bottleneck prevented even basic multi-event learning
   (0.495 vs vanilla 0.995). The Gumbel competition collapsed rather than discovering
   entity structure. Zero write sensitivity confirms no role-indexed operations occurred.

## Diagnostic interpretation

The fundamental bottleneck is **latent occurrence-role assignment from raw text**:
- Which token spans denote which entity across mentions
- Which event/action changes which entity's state
- Which entity the prediction site queries

This is essentially unsupervised coreference resolution + semantic role labeling + 
argument structure identification. A 5M-parameter model trained from scratch on 12K 
synthetic template sentences simply cannot discover these structures from MLM loss alone.

The hard-coordinate version succeeds because it bypasses all three assignment problems
by directly providing entity positions, event positions, and query coordinates.

## Predeclared Criterion

"Treat the next TinyMLM experiment as one bounded test of role induction: if its 
assignments do not survive unseen lexical and paraphrastic structure with strong 
permutation sensitivity, stop cycling memory variants and move to a natural 
source-attested transformation substrate or a fundamentally different route."

Result: FAIL on all five predeclared criteria. Route closed per predeclared gate.

## What this route teaches (positive residue)

1. **The binding deficit is real**: DeBERTa at 100M words learns entity identity
   (affected probe 0.825 at layer 6) but not event-result state polarity (0.5406).
   This is the same computational deficit as vanilla TinyMLM on counterbalanced data.

2. **The mechanism WORKS when roles are known**: hard-coordinate memory reaches
   0.833-0.849 held recombination, with strong paraphrase transfer (0.919) and
   decisive write-permutation destruction. The entity-keyed compositional update
   IS the right computational solution.

3. **Role assignment is the bottleneck**: all four raw-token variants failed at
   approximately the same level despite very different architectures. The common
   bottleneck is latent role assignment, not the update/read mechanism itself.

4. **The bridge to natural text requires either**: 
   (a) Pre-existing role representations (from a pretrained model large enough to
       have learned argument structure), or
   (b) A fundamentally different training signal beyond MLM that teaches role structure.

## Route portfolio after closure

### Closed routes (no further GPU)
- Compact views as architecture-general principle (related experiments)
- TinyMLM memory variants for role induction (compositional update test design)
- All routes listed in the research history "closed routes" paragraph

### Historical checkpoint measurements

These are historical measurements, not current release scores. See the [evaluation provenance](../../../evidence/evaluation_provenance.md) and [final comparison table](../../../results/training_strategy_comparison.csv) for the corrected release protocol and results.
- coherent86 alpha0.75: Overall **42.1210** (beats visible leader at 41.80)
- chck_82M: Overall **41.9425** (reproducible fallback)
- compact_view_reinvest: Overall **42.0868** (inherited tokenizer coordinate)

### Open research directions
1. **Natural source-attested transformation substrate**: bridge atlas found
   103 natural text pairs with compact-view-like geometry. If curated at scale,
   these might provide the data-efficient signal without synthetic generation.
   
2. **Different optimizer/curriculum with compact data**: LAMB + sequence/masking
   curriculum hasn't been cleanly tested with our compact-view configuration.
   
3. **Knowledge distillation or interaction track**: Use approved teacher models
   to provide soft targets rather than hard MLM labels — could implicitly teach
   role structure that the model can't discover from data alone.
   
4. **Synthesize the accumulated findings**: These studies have produced substantial
   scientific evidence about what works (compact views for DeBERTa), what fails
   (cross-architecture transfer, raw-token role induction), and why (the binding
   deficit, the role-assignment bottleneck). This is genuine contribution.

## Scientific Artifacts
- `training/scripts/discrete_role_induction.py`
- `data/discrete_role_induction/discrete_aug20/result.json`
- `data/discrete_role_induction/vanilla_aug20/result.json`
- This note: `notes/memory_route_closure_and_pivot.md`


## cross-data binding comparison (earlier analysis): binding deficit is generic

cross-data binding comparison using compositional update test design counterbalanced corpus
confirms that the entity-event-state binding failure is a **generic DeBERTa
property at this scale**, not specific to any data treatment:

| Treatment | Exposure | EEBF | Affected | Unaffected |
|-----------|----------|------|----------|------------|
| compact_reinvest | 80M | 0.510 | 0.396 | 0.625 |
| compact_reinvest | 100M | 0.523 | 0.396 | 0.651 |
| extractive_balanced | 80M | 0.516 | 0.406 | 0.625 |
| extractive_balanced | 100M | 0.531 | 0.401 | 0.662 |
| extractive_wide | 80M | 0.516 | 0.339 | 0.693 |
| extractive_wide | 100M | 0.510 | 0.333 | 0.688 |
| clean_qwen | 80M | 0.508 | 0.266 | 0.750 |
| scale1.75_adapter | 100M | 0.539 | 0.297 | 0.781 |

All treatments near chance (0.50-0.54). Compact views do NOT help or hurt binding.
The deficit is architectural/scale-bound, not data-induced. This eliminates the
last potential route connecting compact-view data mechanisms to the binding problem.

interpretation boundary is correct: "binding weakness is a generic DeBERTa
property at this scale, not connected to compact views."

## Consolidated route assessment

### Established Findings

1. **Compact faithful rewrites** produce a strong, reproducible DeBERTa masked-
   denoising gain (+2.4164 equal7, seed-spread-exceeding, multi-column)
2. **The mechanism** involves dense copied/shared-target supervision plus compressed
   semantic context; source-absent labels create a real abstractive channel but
   do not mediate the historical downstream gain
3. **The effect is DeBERTa-specific**: RoBERTa and GPT do not transfer the gain;
   DeBERTa compact gains anti-predict RoBERTa movement
4. **Entity-event-state binding** is a persistent computational deficit at this
   scale, universal across data treatments, not fixable by TinyMLM memory
5. **Practical SOTA** achieved: Overall 42.12 (beats visible leader at 41.80)

### What remains scientifically open:

- WHY the compact-view effect is DeBERTa-specific (attention geometry? disentangled
  position encoding? WWM interaction? something in the MLM-specific optimization?)
- Whether a fundamentally different objective (e.g., knowledge distillation from
  approved teacher models) could teach binding
- Whether the bridge atlas substrate (companion analysis, ~2,674 transformation-like candidates)
  can serve as a cleaner same-DeBERTa control

### Recommendation

The remaining research choice is whether:
(a) One more bounded investigation of WHY the effect is DeBERTa-specific would
    constitute a valuable scientific contribution (mechanism-level answer), or
(b) The accumulated findings are sufficient and should be synthesized, or  
(c) A genuinely different route (distillation, interaction track) is worth opening

The practical endpoint (42.12) already exceeds the study's score target.

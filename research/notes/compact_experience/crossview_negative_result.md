# crossview denoising design — Cross-View Conditional Denoising: NEGATIVE Result

## 10M matched evaluation (all same seed43022, same budget, same corpus)

| Column | Baseline | Semantic_xview | Δ xview | Anchor_copy | Δ anchor |
|--------|----------|----------------|---------|-------------|----------|
| BLiMP | 52.86 | 53.88 | +1.02 | 53.80 | +0.94 |
| Supplement | 54.68 | 53.80 | -0.88 | 53.90 | -0.78 |
| EWoK | 51.01 | 48.25 | **-2.76** | 50.01 | -1.00 |
| Entity | 16.66 | 16.43 | -0.23 | 17.08 | +0.42 |
| COMPS | 49.51 | 50.17 | +0.66 | 49.46 | -0.05 |
| GlobalPIQA | 33.21 | 32.15 | -1.06 | 30.70 | -2.52 |
| Reading | 7.40 | 7.24 | -0.16 | 7.85 | +0.45 |
| **Equal7** | **37.90** | **37.42** | **-0.48** | **37.54** | **-0.36** |

## Scientific conclusion

**Masking geometry is NOT the mechanism** driving the same-window paired-experience benefit.

Neither concentrating masks on non-anchor (paraphrastic) targets nor on shared content 
anchors improves over standard random WWM. Both treatments are worse on the aggregate. 
Semantic_xview specifically damages EWoK (-2.76) — the opposite of the predicted effect.

This eliminates the hypothesis that the clean-Qwen benefit comes from the model being 
forced to predict specific word types across views. Instead, the same-window benefit 
(from clean qwen control interpretation and next mechanism controls: +0.35 over duplication, +1.78 over separated-pair) must come 
from something else:

**The mechanism is attention-mediated representation learning from co-occurring semantic 
variants.** The model doesn't need to be forced to predict one view from the other through 
targeted masking — it naturally builds better representations by attending to both 
variants during the standard MLM objective. The benefit is in the DATA COMPOSITION 
(having both views present), not in which positions are masked.

## Implication for next route

Since the masking/objective layer is closed (MNTP, causal, targeted masking all negative),
and the 40k tokenizer layer is closed, and the architecture is constrained to 8×480 by 
the parameter budget, the remaining lever is:

1. **Optimizer** — LAMB with high LR (leader uses LAMB 0.007 vs our AdamW 1e-3)
   - Single-factor, clean test
   - Layer-wise scaling could genuinely improve how the model learns from pairs
   - Major unexplored difference from the 41.8 leader

2. **Data composition** — if data presence (not masking) is the mechanism, can we 
   construct more effective paired content or increase its fraction?

## Closed routes (cumulative)
- tail/mask allocation family (AoA negative)
- MNTP auxiliary (AoA negative)  
- causal batch substitution (no-AoA negative)
- 40k tokenizer interface (fragile/narrow)
- masking curriculum (WWM→token, AMLM-hard)
- content/style selection (C/S heuristics)
- developmental ordering
- contextual one-pair / row geometry
- agreement / SWA / clusters
- **Cross-view conditional denoising (this Step)**

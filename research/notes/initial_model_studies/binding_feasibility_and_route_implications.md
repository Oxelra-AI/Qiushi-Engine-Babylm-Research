# binding feasibility and route implications — binding feasibility assessment and route implications

## Binding-pattern scan results

Evidence: `data/binding_pattern_scan.json`

Scanner: conservative heuristic on high_entity_state windows (quartile of official corpus).
- Total unique windows: 31,325
- Windows with ≥2 entities + swappable bindings: 1,969 (6.3%)
- Windows with + downstream binding-dependent reference: 819 (2.6%)
- Estimated binding-dependent words: ~52k in quartile, ~40k in full corpus

Full-corpus estimate (9.2% of 1.1M lines have ≥2 distinct cap entities):
- At 2.6% binding-downstream rate: ~2,638 qualifying windows × ~15 words = ~40k words
- Over 100M exposure (10 epochs): ~400k binding-word exposures
- As training signal: ~132k binding-targeted gradient updates in ~2.4M total steps (<6%)

## Conclusion: binding-dependent standalone objective is infeasible on the official corpus

Entity-name prediction alone is a copy effect. The valid mechanism requires binding-dependent state prediction. But the official BabyLM corpus (children's dialogue, subtitles, narrative, simple encyclopedia) contains too little multi-entity binding material for this to work at scale — even as a mixed/auxiliary objective.

## What these experiments have systematically ruled out on the official corpus

Under plain or modified masking on the official 10M corpus:

| Route | Steps | Result |
|---|---|---|
| Causal exposure (more data) | 15-17 | Entity flat |
| Sparse routing, prefix memory | 14, 18-27 | No Entity/EWoK |
| Data ordering (curriculum, readability) | 28-31 | No Entity/EWoK |
| Whole-word masking (WWM) | 43-45 | +0.51 Entity (small) |
| Entity mention consistency | 100-108 | Failed replication |
| Counterfactual propagation | 107-114 | Signal didn't localize |
| Cross-sentence span infilling | 114-121 | Failed transfer |
| WikiAuto aligned pairs | 150-155 | Negative Entity/EWoK |
| Official-corpus structure density | 156-161 | Negative Entity/EWoK |
| Entity-name anchored masking (probe) | 164 | Copy effect |
| Binding-dependent masking (feasibility) | 165 | Material insufficient |
| Procedural state microstories | 67-71 | No event sensitivity |

The invariant: no data-selection, data-adjacency, or masking-target mechanism has moved Entity or EWoK when using only the official corpus under DeBERTa-v2.

## What remains live

### 1. Legal alternative data sources (most promising for Entity/EWoK)
The leader uses gated FineWeb simplification pairs. We cannot use that specific dataset, but we CAN use:
- Other accessible web-derived text (Common Crawl extracts, public Wikipedia)  
- Simple English Wikipedia (already partially used in official corpus as simple_wiki)
- Publicly accessible entity-rich factual text
- Rule-based generated binding-rich data (tried in Phase 5 but with weaker model and tiny budget)

Any legal text up to 10M total words counts under BabyLM rules. The key is finding text with genuine entity-state binding density that transfers to Entity/EWoK evaluation.

### 2. GPT-BERT / MNTP hybrid objective (outlined related experiments, never tested)
A masked next-token prediction objective or alternating causal/masked training that forces the model to process both directions. This attacks a different representational bottleneck: directionality and next-token coherence rather than entity binding specifically. The AntLM, GPT-BERT, and ACLM literature suggests this can improve downstream tasks broadly.

### 3. Full 100M LAMB trajectory (secondary)
Weakened by 10M evidence but not fully closed. The leader uses LAMB for the full trajectory with high LR 0.007. Our 10M early signal was weak (-Entity, -EWoK), but late-trajectory behavior might differ. Lower priority than (1) and (2).

### 4. Maximize Overall through non-Entity columns
Our protected model already beats the leader on Supplement (+3.87) and Reading (+2.20). If Entity/EWoK cannot be improved with official data alone, the SOTA path may require gaining on GlobalPIQA, COMPS, SuperGLUE, or further extending Supplement/Reading advantages. S1 shape gives +1.97 GlobalPIQA. Combining S1 shape + full 100M + other optimizations might close the 1.27 Overall gap without matching Entity.

## Recommendation for next route

Route (1) — legal alternative data — is the highest-value next action because:
- The leader's Entity/EWoK gap is almost certainly data-driven (all model-side factors tested)
- The official corpus is demonstrably insufficient for entity-binding signal
- Legal accessible alternatives exist (accessible Wikipedia, factual text, generated binding data)
- The BabyLM rules explicitly allow custom data under the 10M word budget

Route (2) — hybrid objective — is the second priority because it attacks a different mechanism dimension that has not been tested in these experiments and has literature support from GPT-BERT and AntLM.

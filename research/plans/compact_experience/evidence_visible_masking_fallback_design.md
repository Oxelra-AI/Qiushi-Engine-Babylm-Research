# Evidence-Visible Masking: Fallback Route Design

## Core Mechanism

Instead of masking tokens uniformly at 15%, **preferentially mask tokens that 
require relational/world knowledge to predict**, while keeping context (evidence) 
tokens visible. This focuses the gradient on learning entity attributes, relations, 
and compositional facts — exactly what EWoK and COMPS measure.

## How This Differs From Closed Routes

| Route | What Changes | Closed Because |
|-------|-------------|----------------|
| AMLM-hard | Loss weighting (upweight hard tokens) | No broad gain at 100M |
| Masking curriculum | Mask probability over time | Token-level ≤ WWM at scale |
| Qwen pairs | Data composition | Not sufficient vs leader |
| Ordering | When text appears | No effect on broad frontier |
| **Evidence masking** | **Which positions get masked** | **Not yet tried** |

Key distinction: AMLM-hard masks uniformly then reweights loss. Evidence-visible 
masking changes which positions are masked, keeping relational evidence visible 
so the model can use it to predict masked entity/attribute tokens.

## Token Classification (Corpus-Internal Only)

Identify mask-priority positions using only surface/syntactic heuristics:

**High-priority (mask at 25-30%)**:
- Named entity tokens (capitalized multi-word sequences not at sentence start)
- Number tokens in measurement/comparative context
- Attribute-assigning content words (adjective/adverb after copula)
- Object nouns after relational verbs ("built", "discovered", "became", "won")

**Normal priority (mask at 10-12%)**:
- Function words, determiners, prepositions
- Common verbs in standard position
- Sentence-initial tokens

**Overall constraint**: Total masked tokens per sequence ≈ same as standard 15% MLM.
This is NOT an exposure-changing intervention — it's a credit-assignment intervention.

## Implementation Approach

1. Modify `wwm_mask_batch()` to accept per-token mask priorities
2. Token priority computed once per batch from simple heuristics:
   - Capitalization pattern (entity detection)
   - Following copula/relational verb
   - Numeric context
3. Sample masked positions with priority-weighted probabilities instead of uniform
4. Keep total masked count approximately equal to uniform 15%

## Controls

- C1: Standard uniform WWM at 15% (matches current clean-Qwen baseline)
- C2: Random priority assignment (same total masks, random which positions are "high priority")
- C3: Inverse priority (mask function words more, entities less — tests directionality)

## Expected Signal

If the mechanism works:
- EWoK/COMPS improve (model forced to predict entity attributes from context)
- BLiMP may slightly decrease (less gradient on syntactic positions)
- Entity Tracking may improve (entities are prediction targets)
- Supplement unchanged (depends on overall language model quality)

## Decision: When to Activate

- Activate if cluster mechanism construction record cluster E1 does NOT beat E2/E3/E4 on the no-AoA equal7 profile
- Can use same 80M→100M continuation framework
- Implementation cost: ~2 hours (modify masking function + materialize + train)
- Evaluation cost: same as current cluster arms

## Relation to AMLM-Hard (Why This Is Different)

AMLM-hard (closed in masking curriculum training done):
- Masks uniformly, then upweights loss on tokens the model gets wrong
- Makes easy tokens easier (less gradient), hard tokens harder
- Failed because: at scale, "hard" tokens are often noise/rare words, not entity facts

Evidence-visible masking:
- Changes WHICH tokens are masked based on linguistic role
- Entity/relation tokens are masked (model must predict them)
- Context/evidence tokens are visible (model can use them)
- This is credit assignment: the model receives gradient for entity/attribute prediction
  backed by visible relational evidence, not for guessing function words from syntax

## Implementation Timeline

If clusters fail → design takes ~1 hour:
1. Implement priority scorer in token-classification function
2. Modify masking to use priority-weighted sampling
3. Materialize matched pools (same data, different masking seeds)
4. Launch continuation from chck_80M with same framework

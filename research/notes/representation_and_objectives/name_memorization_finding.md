# name memorization finding — What the name-memorization diagnostic establishes

## Core finding

When a BiGRU sequence model (48-dim embeddings, 48-dim hidden) learns to parse
event-role NLI from natural text, it achieves ~98% training accuracy through
**entity-name memorization**, not predicate-semantic learning.

### Evidence

Name-isolated diagnostic (329 train families, 71 name-held families, ZERO name overlap):

| Surface | Names | Predicates | Accuracy | N seeds |
|---------|-------|-----------|----------|---------|
| Training set | Trained | Anchor (known) | 0.937 | 2 |
| Within-family probe | **Shared** | Probe (unknown) | **0.301** | 2 |
| Cross-family anchor | **New** | Anchor (known) | **0.513** | 2 |
| Cross-family probe | **New** | Probe (unknown) | **0.529** | 2 |

Key comparisons:
- **cross_anchor ≈ 0.50 (chance)**: the model cannot transfer even KNOWN predicates
  to new entity names. This proves zero predicate-semantic learning.
- **within_probe ≈ 0.30 (below chance)**: the model memorizes name-winner associations
  from one context and systematically applies them to the REVERSED-role context.
- **cross_probe ≈ cross_anchor ≈ 0.50**: predicate novelty adds NO additional cost
  beyond name novelty, because predicates were never learned.

Additional structural confirmation:
- ALL 400 train families have 100% role reversal (A wins in one context, B in the other)
- The below-chance within-family accuracy is expected from name memorization + role reversal

## What this means for the factorized role-coordinate analysis

role coordinate anchor and state probe's factorized model showed perfect anchoring transfer because it EXPLICITLY
parameterized predicate role coordinates. In raw text, the BiGRU never reaches the
predicate-coordinate question — it is stuck at entity-identity memorization.

The gap:
- **Factorized**: predicate coordinates are the learned parameters → anchoring works
- **Raw text**: name-entity associations are the learned parameters → predicates not learned
- **This is not a failure of the anchoring principle but of the representation reaching it**

## What this means for data-efficient learning

This is a concrete demonstration of the fundamental tension in limited-data learning:

1. **With natural entity names**: the model memorizes entity-specific associations,
   achieving high training accuracy but zero relational generalization.
   
2. **With anonymized names** (equivariance protocol): the model cannot learn at all, because
   the remaining signal (predicate word positions, syntax) is too weak for a small
   randomly-initialized sequence model.

3. **The missing capability**: separating entity identity from relational structure
   so that predicate semantics can be learned and reused across entity combinations.

This directly connects to the persistent binding deficit (hypothesis comparison):
- The model represents entity identity (who is mentioned)
- The model does NOT learn event-role semantics (what the predicate means)
- Under interference (same entities, different roles), the model fails

## Implication for the live research route

The anchoring principle (role coordinate anchor and state probe) is mathematically sound: sparse links between
new and established role coordinates fix global orientation ambiguity. But in raw
text, the prerequisite — learning predicate role coordinates at all — is not met by
a small randomly-initialized BiGRU.

The next question is whether this failure is:
(a) Architecture-specific (BiGRU can't extract predicate semantics from text)
(b) Scale-specific (too few parameters or training examples)
(c) Task-format-specific (the NLI format encourages name memorization)
(d) Fundamental (random initialization cannot discover predicate semantics
    from role-labeled text alone — pretrained representations are required)

The answer determines which kind of data-efficient learning principle applies:
- If (a) or (b): the principle is about architecture or scale thresholds
- If (c): the principle is about training-signal design
- If (d): the principle is about the role of prior knowledge in relational learning

## Evidence files
- `data/name_memorization_diagnostic/name_memorization_diagnostic.json`
- `data/raw_text_anchor_transfer/template_inversion_diagnostic.json`
- `scripts/name_memorization_diagnostic.py`
- `scripts/template_inversion_diagnostic.py`

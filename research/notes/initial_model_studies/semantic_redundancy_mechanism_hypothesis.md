# semantic redundancy mechanism hypothesis — Semantic Redundancy Mechanism Hypothesis

## The learning principle under test

**Hypothesis**: Within-context semantic redundancy converts ordinary masked language modeling into implicit contrastive representation learning for surface-invariant entity and relation features.

When an MLM window contains two surface renderings of the same semantic content (e.g., "The marble was placed in the red box" + "It was put into the crimson container"), predicting masked tokens in either version is aided by attending to the other version. This attention pressure forces internal representations to capture meaning-level features (entity identity, property, relation) that are invariant to surface form. Ordinary single-surface MLM has no such pressure: the model can learn purely surface-statistical patterns without extracting entity/relation abstractions.

## Why this should specifically improve Entity/EWoK

- **Entity Tracking** requires recognizing that different surface mentions (pronouns, definite descriptions, name variants) refer to the same entity across operations. Aligned pairs provide exactly this signal: the same entity appears in different surface forms within one learning window.
- **EWoK (world knowledge)** requires knowing properties and relations that hold regardless of how they are stated. Seeing "glass is fragile" and "the glass breaks easily when dropped" in one window forces the model to learn the property invariant, not the specific surface.
- **BLiMP/Supplement (grammar)** should be largely unaffected because grammatical acceptability is a surface-level property that single-surface MLM already teaches well.

## Experimental design: what the comparison means

| Comparison | Positive result means | Negative result means |
|---|---|---|
| aligned > shuffled (Entity, EWoK) | Correct semantic correspondence matters; the mechanism is pair-binding-specific | Text distribution or same-window co-occurrence is sufficient; no need for correct pairing |
| aligned > unpaired_mix | Same-window co-occurrence matters beyond just seeing both texts | The benefit is from the data mixture, not structural co-occurrence |
| unpaired_mix > source_only | Rewrite-style text adds value beyond original text alone | Simplification text itself is not the driver |
| rewrite_only > source_only | Simpler text is a better pretraining distribution | Complexity/style is not the primary axis |
| aligned ≈ shuffled ≈ unpaired ≈ source ≈ rewrite | The data source is irrelevant; mechanism is not in this axis | (Negative; need different approach) |

## Connection to data-efficient learning principles

If the mechanism is real and the aligned > shuffled signal is robust:

1. **Sample efficiency**: structured semantic redundancy extracts more representation learning per word than random text, because each word exposure carries both surface-prediction and invariance-learning signal.
2. **Transferability**: the principle applies to any domain where surface-invariant abstractions matter (not just language: multi-view learning, augmentation in vision, etc.).
3. **Scaling law implication**: under extreme data budgets (≤10M words), structured redundancy may be a stronger lever than additional raw text or architectural tricks.
4. **Mechanism**: MLM + correct pairing = implicit InfoNCE where the positive pair is the aligned version and negative pairs are random context. No explicit contrastive loss is needed.

## Current experimental state

- **10M five-arm corpus materialized**: 9,999,996 words per arm, 212,001 selected pairs, all hard invariants pass (identical text multisets, zero token/group deltas, no overlength, no identity shuffled adjacency).
- **Training launched**: aligned and shuffled arms on S1 12×384/baseline16k/AdamW/flat-WWM.
- **Pending**: evaluation of both arms on BLiMP, Supplement, Entity, COMPS, GlobalPIQA, Reading, full local EWoK.
- **Decisive signal**: aligned minus shuffled on Entity and EWoK, with guard rails on BLiMP/Supplement/Reading.

## If the signal is positive

Scale to 100M (10 epochs over the 10M aligned-pair corpus). If the mechanism survives scaling and produces Entity/EWoK gains competitive with the public leader, it becomes the foundation for a genuine SOTA submission and a transferable data-efficient learning principle paper.

## If the signal is zero or negative

The mechanism is falsified for this data source and scale. Next: either the accessible wiki_auto simplification pairs lack sufficient entity/relation density to activate the mechanism, or the mechanism hypothesis itself needs revision. Check whether the leader's advantage comes from a different data property (e.g., FineWeb source quality, entity density, or diversity that wiki_auto lacks).

# earlier analysis — corrected route after the ideal binding inference test

## Corrected interpretation

ideal binding dependency probe shows that the existing protected 100M WWM model is insensitive to entity→state swaps on a synthetic template set. It is an inference result on a trained model. It does **not** prove that all proposition-rich text trained with WWM could never shape binding ability; that would require pre/post training-response evidence, gradients, and learning curves.

The scientifically secure conclusion is narrower:

- Current WWM representations do not use entity→state assignment for masked state/location/property predictions on this template family.
- Therefore, before spending 10M-word training on any proposition-rich corpus mixture, we must first create a learning signal where the **same word bag** produces a different loss when the entity→state binding is wrong, then verify that this signal changes binding behavior.

## S1 100M is not a shortcut to Overall SOTA

Known seven-column comparison:

- Protected 8×480 known-seven sum: 296.895
- S1 12×384 known-seven sum: 296.525
- S1 minus protected over those seven columns: -0.370

To exceed public leader Overall 41.80, a 9-column model needs total sum > 376.20.

Given S1 known-seven sum 296.525, S1 would need:

- SuperGLUE + AoA > 79.675

Protected SuperGLUE + AoA is:

- 68.0218 + (-0.1745) = 67.8473

So S1 would need a SuperGLUE+AoA jump of about +11.83 over the protected model. There is no existing evidence for such a jump. Returning to already completed S1 100M as a practical SOTA route is therefore weak.

## Correct next object: same-word-bag binding-contrastive learning

We need a training objective whose loss cannot be minimized by local frequency alone.

### Training pair

For each event passage, construct two versions with the same words:

- correct: `Alice put the ball in the box. Bob put the cup in the basket. Later Alice got the ball from the box.`
- swapped: `Bob put the ball in the box. Alice put the cup in the basket. Later Alice got the ball from the box.`

Both contain the same entity names, objects, and locations. Only the entity→object/location assignment changes.

### Loss options

#### A. Pairwise ranking loss over a sentence/passage score

Let `S(x)` be a scalar compatibility score from the encoder, computed from a `[CLS]` vector or pooled relation-span vectors.

\[
L = L_{WWM}(x^+) + \lambda \max(0, m - S(x^+) + S(x^-)).
\]

This directly makes the correct binding lower-loss than the swapped binding under identical word bag.

#### B. Span-local binary compatibility

Mark relation-relevant tokens in both passages and train a binary head on relation-span pooled states:

\[
L = L_{WWM} + \lambda \sum_j BCE(d(h_j), z_j),
\]

where `z=1` for correct entity-state assignment and `z=0` for swapped assignment. The model must use context assignment because local tokens and unigram statistics are matched.

#### C. Masked target + contrastive context

For the same target span in both passages, compare target log-likelihood under correct vs swapped context:

\[
L = L_{WWM} + \lambda \max(0, m - \log p(t|x^+) + \log p(t|x^-)).
\]

This most directly connects to Entity/EWoK-style state prediction. It may be the best first implementation because it reuses the MLM head and the existing ideal binding dependency probe template generator.

## First decisive experiment: tiny learning-response test before any 10M run

Use the ideal binding dependency probe ideal template generator to create a compact legal synthetic training set. Every word counts only if later used for BabyLM training; this preliminary method test is not a submission corpus yet.

Train four small, matched models from the same initialization on the same template family:

1. WWM only on correct passages
2. WWM only on correct+swapped passages
3. WWM + random-pair ranking, where positive/negative labels are shuffled
4. WWM + true same-bag binding contrastive loss

All arms match word exposure, batch count, seed, model size, and tokenizer.

Evaluate before and after training on:

- held-out same-template names/locations
- held-out template families
- distractor entities and overwritten states
- the ideal binding dependency probe correct-minus-swapped target likelihood margin
- exact binding-choice accuracy: whether `log p(target | correct) > log p(target | swapped)`

The needed result is not just lower training loss. The true binding-contrastive arm must selectively increase held-out correct-minus-swapped margin and binding-choice accuracy relative to WWM and random-pair controls, especially on unseen names/locations and overwritten-state templates. If it only memorizes templates or improves entity-name copying, it is not the mechanism.

## If the learning-response test succeeds

Then build a legal 10M training route:

- replace a modest portion of official text with generated or licensed same-bag contrastive episodes, every word counted under Strict-Small;
- train S1-shape DeBERTa with WWM plus the contrastive loss;
- include an equal-word episode-only WWM control and a random-pair contrastive control;
- evaluate Entity, EWoK, COMPS, GlobalPIQA, BLiMP, Supplement, Reading, plus internal binding margins.

## If it fails

If even true same-bag contrastive training does not induce held-out binding behavior, then the bottleneck is likely the architecture/state representation rather than the data or loss alone. The next route becomes persistent entity/event memory or another state-carrying inductive bias.

## Current route decision

Do not run S1 100M again. Do not train a 15–20% mixed proposition corpus without first verifying learning response. The immediate next work is construction of a small same-bag binding-contrastive trainer and held-out binding evaluation using the ideal binding dependency probe template generator.

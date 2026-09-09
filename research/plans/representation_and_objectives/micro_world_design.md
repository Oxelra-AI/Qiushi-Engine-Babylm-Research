# hypothesis comparison — Executable Counterfactual Micro-World: Design

## Purpose

Build a fixed legal-pool diagnostic that measures **context-conditioned
alternative binding** through crossed-sign accuracy per frame. This
measurement must:

1. Separate the four hypotheses in `notes/hypothesis_comparison.md`
2. Reject the sparse20 turnover artifact (margin compression cannot
   produce crossed signs)
3. Predict full-surface EWoK behavior
4. Be scored on existing checkpoints without training

Only if this measurement validates against known endpoint rankings should
any new training experiment be proposed.

## Metric: crossed-sign interaction

For each frame (C₁, C₂, query, alternatives {a, b}):

  Δ₁ = s(C₁, query, a) − s(C₁, query, b)  
  Δ₂ = s(C₂, query, a) − s(C₂, query, b)  
  I  = Δ₁ − Δ₂

where s(C, Q, x) is the masked-LM pseudo-log-likelihood of completion x
given context C at the query position (using the fw ewok interaction reader four-cell reader).

**Crossed-sign accuracy**: fraction of frames where Δ₁ > 0 AND Δ₂ < 0.
This requires the model to prefer the correct alternative in EACH context,
not just have a positive interaction term.

**Why this rejects turnover**: Margin compression (the sparse20 artifact)
pushes both |Δ₁| and |Δ₂| toward zero. This might flip one sign but
cannot systematically produce both correct signs. A model that compresses
margins would have low crossed-sign accuracy even if I > 0 for some frames.

## Frame types and composition depths

### Type A: Direct relation binding (depth 1)

The context states a single relation; the query asks about it.

Template:
```
C₁: "[E1] [verb] [relation₁] [E2]."
C₂: "[E1] [verb] [relation₂] [E2]."
Query: "[E1] is [MASK] [E2]."
Alternatives: {relation₁, relation₂}
```

Examples from the legal pool (spatial family):
- C₁: "The cat is on the table." → answer: on
- C₂: "The cat is under the table." → answer: under

Extraction approach:
- Mine the legal 10M pool for sentences matching "ENTITY is {in/on/under/
  behind/above/below/inside/outside/near/far from} ENTITY"
- Extract (entity_pair, relation, source_sentence) triples
- Generate C₂ by substituting the relation with its antonym
- Keep entity pair and query fixed
- Validate: relation must be unambiguous from the context

### Type B: Argument-role binding (depth 1)

The same event with swapped argument roles.

Template:
```
C₁: "[E1] gave the book to [E2]."
C₂: "[E2] gave the book to [E1]."
Query: "Who received the book? [MASK]"
Alternatives: {E1, E2}
```

Extraction approach:
- Mine for transitive possession/transfer verbs: gave/sent/sold/handed/
  passed/offered/showed/taught/told
- Extract (agent, patient, verb, object) tuples
- Generate C₂ by swapping agent/patient roles
- Validate: the agent and patient must be distinct named or common entities

### Type C: Belief attribution (depth 1.5)

A mental-state verb determines the implied content.

Template:
```
C₁: "[E1] believes that the diamond is inside."
C₂: "[E1] doubts that the diamond is inside."
Query: "According to [E1], the diamond is most likely [MASK]."
Alternatives: {inside, outside}
```

Extraction approach:
- Mine for belief/report verbs: believes/thinks/knows/doubts/denies/
  suspects/fears/hopes
- The polarity of the mental-state verb determines the implied content
- Generate C₂ by replacing with an antonymic mental-state verb
- Validate: the subordinate clause must contain a clear binary choice

### Type D: State-change composition (depth 2)

Two events compose to determine a final state.

Template:
```
C₁: "[E1] opened the box. Then [E1] closed the box."
     → final state: closed
C₂: "[E1] closed the box. Then [E1] opened the box."
     → final state: open
Query: "The box is now [MASK]."
Alternatives: {open, closed}
```

Extraction approach:
- Mine for reversible state-change verbs: open/close, lock/unlock,
  add/remove, enter/leave, attach/detach, turn on/turn off
- Extract two-event sequences operating on the same object
- Generate C₂ by swapping event order (noncommuting pairs only)
- Validate: the final state must be deterministically specified by the
  last event

## Controls (required before scoring)

1. **Context-removed baseline.** Score the query with alternatives but no
   context C. Measures candidate prior. A model with a strong prior for one
   alternative will score well on half the frames by chance. Crossed-sign
   accuracy with no context should be ≈ 0%.

2. **Shuffled-context control.** Score with context from a different frame
   paired with the current query/alternatives. Should be at chance.

3. **Candidate-prior control.** For each alternative pair, compute the
   corpus frequency of each alternative in the training data. If one is
   much more frequent, the frame is biased. Report frequency-balanced and
   frequency-imbalanced subsets separately.

4. **Entity renaming.** Replace entity names with novel but tokenizer-stable
   names (e.g., "Blicket", "Dax"). Renaming should not change binding
   accuracy; if it does, the model relies on entity identity rather than
   relational structure.

5. **Token-length matching.** Both alternatives in each frame should have
   the same number of tokens (for the given tokenizer). If not, report
   length-matched and length-mismatched subsets.

## Checkpoint families in the scoring plan

These are model-family descriptions, not filesystem paths. For concrete model
identities and access instructions, use the
[checkpoint references](../../../reproducibility/checkpoint_references.md).

### Primary (span key endpoint variation):

| Label | Model family | Tokenizer | Checkpoints | Notes |
|---|---|---|---|---|
| scale1.75_77M–83M | Scale-1.75 compact-view reference | legal16k | chck_77M–chck_83M | Known peak 82M for Overall |
| scale1.75_100M | Same | legal16k | chck_100M | Terminal |
| legal40k_s43022_80/90/100 | 40K-tokenizer reference, seed 43022 | legal40k | chck_80M, chck_90M, chck_100M | Known coarse peak 90M |
| legal40k_s43122_45/80/100 | 40K-tokenizer reference, seed 43122 | legal40k | chck_45M, chck_80M, chck_100M | Known coarse peak 80M |

### Negative controls (should NOT show improved binding):

| Label | Model family | Tokenizer | Checkpoints | Notes |
|---|---|---|---|---|
| mlm_only_20M | MLM-only relation-pool baseline | legal16k | final | MLM baseline |
| coupled_aligned_20M | Coupled aligned-pair model | legal16k | final | Turnover, not binding |
| coupled_shuffled_20M | Coupled shuffled-pair control | legal16k | final | Turnover control |

### Important scoring rule

Each checkpoint is scored with its OWN tokenizer. The crossed-sign metric
is tokenizer-independent (it compares two completions under the same tokenizer).
Legal16k and legal40k models will produce different absolute pseudo-likelihoods,
but the relative ordering of alternatives within each model is comparable.

## Validation thresholds (from independent_review, adapted)

The probe must satisfy ALL of these before any training is proposed:

1. **Rank 82M near the top of scale-1.75 77–83M**: The micro-world should
   identify 82M's peak if binding contributes to Overall score.

2. **Prefer seed43022 90M over 80M and 100M**: Consistent with the known
   coarse legal40k peak.

3. **Prefer seed43122 80M over 45M and 100M**: Consistent with the known
   coarse seed43122 peak.

4. **Survive all five controls**: Context-removed, shuffled-context,
   candidate-prior, entity-renaming, and token-length controls.

5. **Add checkpoint-ranking information beyond held-text MLM loss**: The
   micro-world must correlate with Overall/EWoK score more strongly than
   with training loss alone.

6. **Not reward coupled aligned over shuffled**: The coupled variants should
   NOT show higher crossed-sign accuracy than MLM-only, confirming the
   measurement rejects the turnover artifact.

## Hypothesis-discriminating measurements

Beyond crossed-sign accuracy, the micro-world should produce:

1. **Composition-depth profile**: Accuracy at Type A/B (depth 1) vs
   Type C (depth 1.5) vs Type D (depth 2). A steep drop supports H4
   (compositional bottleneck). A flat profile supports H1/H2/H3.

2. **Cross-trajectory variance**: Compare scale-1.75 82M, legal40k s43022
   90M, and legal40k s43122 80M. If they score similarly → H1. If they
   differ → investigate data/geometry.

3. **Representation geometry (optional extension)**: Extract hidden states
   at the query position given C₁ vs C₂. Compute the cosine distance.
   If this predicts crossed-sign accuracy → H3.

4. **Training-progress curve**: Score scale-1.75 at 20M intervals
   (chck_20M, 40M, 60M, 80M, 82M, 100M). If the curve plateaus early
   → H1 (basin reached). If it's monotonically increasing → H2 (data
   coverage builds gradually).

## Proposed Construction Pipeline

1. **Extract**: Scan the legal 10M pool JSONL for sentences matching
   relation patterns. Use rule-based regex/pattern matching, NOT a
   pretrained parser (which would need compliance accounting).

2. **Transform**: For each extracted sentence, generate the contrastive
   context by rule-based substitution (relation antonym, argument swap,
   verb antonym, event reorder).

3. **Filter**: Require both alternatives to be single tokens (for the
   primary tokenizer family), unambiguous, and corpus-attested. Reject
   frames where the transformation changes token count or introduces OOV.

4. **Balance**: Equalize frame counts across types and relation subtypes.
   Balance the correct-alternative identity (half frames correct for
   alternative a, half for b).

5. **Freeze**: Lock all construction parameters, frame IDs, transformations,
   and weights BEFORE scoring any checkpoint. Official labels are never
   consulted during construction.

6. **Score**: Run the fw ewok interaction reader four-cell pseudo-likelihood reader for each
   (frame, checkpoint) pair. This is inference-only, no training.

7. **Aggregate**: Compute crossed-sign accuracy, interaction distributions,
   control baselines, and the discriminating measurements above.

## Relation to the practical endpoint

The micro-world is a DIAGNOSTIC, not a training objective. It measures
whether an existing model has context-conditioned binding. If it validates
(thresholds 1–6), it becomes the selection criterion for any future
training experiment. A training experiment would only be justified if:

- The micro-world identifies a binding deficit in chck_82M that the leader
  likely does not share (based on the leader's EWoK advantage)
- The training experiment targets a specific hypothesis (H1/H2/H3/H4)
  identified by the discriminating measurements
- The experiment preserves the broad-competence substrate

The chck_82M endpoint remains the protected, reproduced, above-frontier
practical candidate regardless of micro-world results.

## Dependencies and cost

- **Extraction**: CPU-only, minutes (regex/pattern over 64K rows)
- **Scoring**: GPU inference, ~30–60 min per checkpoint (reusing fw ewok interaction reader
  reader infrastructure), parallelizable across GPU0/GPU1
- **Total checkpoints to score**: 7 scale-1.75 + 3 legal40k s43022 +
  3 legal40k s43122 + 3 controls = 16 models
- **Total GPU time**: ~8–16 hours (parallelized to ~4–8 wall hours)
- **No training required**: all existing checkpoints

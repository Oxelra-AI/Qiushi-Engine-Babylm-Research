# hypothesis comparison — Micro-world feasibility analysis and refined design

## Raw extraction results (noisy)

| Type | Raw count | Estimated clean % | Estimated clean count |
|---|---:|---:|---:|
| A_spatial | 4,286 | ~15-20% | ~640-860 |
| B_transfer | 1,679 | ~25-35% | ~420-590 |
| C_belief | 189 | ~75-85% | ~140-160 |
| D_state_change | 972 | ~20-30% | ~195-290 |
| **Total** | **7,126** | | **~1,395-1,900** |

### Noise sources

1. **Idiomatic spatial**: "is in progress", "is in a state of free fall",
   "is in effect" — these use spatial prepositions idiomatically and would
   produce trivially false antonymic transformations ("is outside progress").
   Dominates A_spatial/in_vs_outside (2,927 of 4,286 spatial frames).

2. **Phrasal verbs**: "opened up" (territory), "gave place to" (yielded),
   "showed her tendency" — these aren't genuine transfer/state-change events.
   Contaminates B_transfer and D_state_change.

3. **Abstract uses**: "added to trust", "removed from the body" — these are
   abstract rather than concrete state changes.

4. **Missing entities**: Many spatial frames lack two clearly identified
   entities (the CONTAINER and the CONTAINED object).

### Belief type is too sparse
Only 189 total (182 positive, 7 negative). The positive-only dominance means
most frames lack a natural antonymic transformation partner.

## Revised approach: template-based micro-world

Instead of mining natural sentences (which are heavily contaminated with
idiomatic uses), construct minimal contexts from **corpus-attested vocabulary**
using fixed templates. This is structurally analogous to BLiMP and EWoK:
controlled templates that isolate a specific linguistic capability.

### Advantages over corpus extraction
1. **No idiomatic contamination**: templates only use prepositions/verbs
   in their literal relational sense
2. **Perfect balance**: every frame has exactly two alternatives, with
   the correct one deterministically specified by the context
3. **Compositional depth control**: can systematically vary from depth 1
   (single-fact) to depth 2+ (multi-step)
4. **Controls are trivial**: context removal, shuffling, and renaming
   are all mechanistic
5. **Tokenizer-independent**: alternative pairs chosen for single-token
   status in both legal16k and legal40k

### Vocabulary extraction needed
From the legal 10M pool, extract:
- **50-100 proper names**: frequent proper nouns in the corpus
- **50-100 concrete objects**: nouns appearing near spatial/transfer verbs
  (book, key, ball, box, hat, etc.)
- **20-50 locations**: nouns appearing as spatial endpoints
  (room, table, box, house, bag, etc.)
- **Common verbs**: transfer verbs (gave, sent, handed, passed, showed,
  told, sold)

All vocabulary items must be single tokens in BOTH legal16k and legal40k.

### Template set

#### Type A: Spatial binding (depth 1)
```
C₁: "[NAME] put the [OBJECT] [REL₁] the [LOCATION]."
C₂: "[NAME] put the [OBJECT] [REL₂] the [LOCATION]."
Q:  "The [OBJECT] is [MASK] the [LOCATION]."
Alternatives: {REL₁, REL₂}
```
Relation pairs: (in, on), (in, under), (on, under), (inside, outside),
(above, below), (behind, near)

Target: ~200 frames (balanced across 6 relation pairs)

#### Type B: Argument-role binding (depth 1)
```
C₁: "[NAME1] gave the [OBJECT] to [NAME2]."
C₂: "[NAME2] gave the [OBJECT] to [NAME1]."
Q:  "[MASK] received the [OBJECT]."
Alternatives: {NAME1, NAME2}
```

Target: ~200 frames

#### Type C: State-change binding (depth 1)
```
C₁: "[NAME] [VERB₁] the [OBJECT]."
C₂: "[NAME] [VERB₂] the [OBJECT]."
Q:  "The [OBJECT] is now [MASK]."
Alternatives: {STATE₁, STATE₂}
```
Verb/state pairs: (opened, closed)→(open, closed), (locked, unlocked)→
(locked, unlocked), (turned on, turned off)→(on, off)

Target: ~100 frames

#### Type D: State-change composition (depth 2)
```
C₁: "[NAME] [V₁] the [OBJ]. Then [NAME] [V₂] the [OBJ]."
C₂: "[NAME] [V₂] the [OBJ]. Then [NAME] [V₁] the [OBJ]."
Q:  "The [OBJ] is now [MASK]."
Alternatives: {STATE₁, STATE₂}
```
The answer depends on the LAST event, so swapping event order changes
the correct answer.

Target: ~100 frames

### Total: ~600 frames × 16 checkpoints × 4 scoring conditions = ~38,400 evaluations
Estimated GPU time: ~2-4 hours total (parallelizable)

### Scoring method

Reuse the fw ewok interaction reader four-cell pseudo-likelihood reader:
1. Concatenate context + query
2. Mask the query target position
3. Score both alternatives
4. Compute Δ₁, Δ₂, I per frame

Metric: **crossed-sign accuracy** (fraction with Δ₁ > 0 AND Δ₂ < 0).

### Next steps

1. **Extract corpus vocabulary**: names, objects, locations from the legal pool
   (CPU-only, fast)
2. **Filter for single-token status**: check against both legal16k and legal40k
3. **Generate frames**: fill templates with filtered vocabulary
4. **Freeze frames**: lock all frame IDs, vocabulary, and templates before
   any checkpoint scoring
5. **Score checkpoints**: parallel GPU inference on 16 existing checkpoints
6. **Aggregate and analyze**: crossed-sign accuracy, composition-depth
   gradient, cross-trajectory variance, validation thresholds

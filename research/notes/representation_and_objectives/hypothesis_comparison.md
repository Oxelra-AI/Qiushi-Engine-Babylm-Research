# hypothesis comparison — Hypothesis comparison: the binding deficit

## The problem stated precisely

Across all compliant endpoints, DeBERTa-v2 models trained on ≤10M unique words
fail when the same plausible alternatives must exchange roles under competing
contexts. This manifests as:

- **EWoK** stable conditional reversals: ~2,400–2,600 / 7,618 rows are
  stably wrong across checkpoints, concentrated in context-target interaction
  failures rather than token rarity.
- **GlobalPIQA_parallel** hard52: 0–2/52 correct for compliant endpoints;
  correct-option ranks distributed nearly uniformly 1/2/3/4 = 23/24/26/30;
  mean top-minus-correct margin 1.953 nats.

The deficit is **not explained** by any tested local objective change, masking
strategy, synthetic exchange packet, late-readout protection, source-rewrite
alignment, adapter coupling, optimizer, curriculum, or averaging. The only
intervention that strongly moved the hard rows (coupled sparse20) was proven
correspondence-free and trajectory-destructive.

## Four constraints any explanation must satisfy

1. **Local compatibility is saturated.** Same-context masked-target ranking
   is 94% for natural targets. Visible-target auxiliary reduces to lookup.
   Same-context slot assignment already works.

2. **Constructed role reversals are learnable but don't transfer.**
   Balanced exchange packets achieve 0.6987 synthetic accuracy but
   GlobalPIQA_parallel stays at ~24%, hard52 at 0–2/52. The model memorizes
   the specific packet associations without generalizing.

3. **No robust interior solution survives.** Mid-layer scans show transient
   correctness, but decoder-robust tests with adjacent-depth persistence,
   target-swap, and rotated-label nulls reduce the signal to chance.

4. **Source-rewrite correspondence adds ~nothing.** Coupled aligned vs
   shuffled: 4.47% true-correspondence share on the fixed 1,471-row subset;
   on all 7,618 EWoK rows, shuffled outperforms aligned (accuracy +0.263 vs
   −0.105, stable failures 2,237 vs 2,366).

## Hypothesis H1: Loss-landscape trapping

**Statement.** Standard MLM at 10M-word scale converges to a region where
context-independent plausibility rankings satisfy >99% of the training gradient.
The loss landscape is locally flat in directions that would encode context-
conditioned binding. Escaping requires a perturbation large enough to shift the
optimization trajectory's basin, which simultaneously disrupts the broad-
competence basin.

**Explains (1):** Local compatibility is the dominant loss-reduction mechanism;
the model is well-trained for it and the minimum is deep.

**Explains (2):** Packets (0.22% of stream) are too sparse to reshape the
trajectory. The model encodes them as additional local associations.

**Explains (3):** No persistent selection pressure to maintain binding
representations; intermediate correctness appears transiently through
feedforward computation but subsequent layers overwrite it in favor of the
locally-optimal prediction surface.

**Explains (4):** Source-rewrite pairs share propositional content and don't
create gradient in binding-relevant directions. Coupled training's trajectory
perturbation creates correspondence-free margin compression.

**Key prediction.** Micro-world crossed-sign accuracy would be **similar
across checkpoints from the same trajectory** and **similar across
trajectories** that reached comparable overall training quality, because they
all converge to the same basin. The micro-world score would plateau early
in training (well before 50M words).

## Hypothesis H2: Context-diversity starvation

**Statement.** With 10M unique words from limited-diversity genres
(child-directed speech, SimpleWiki, Gutenberg, subtitles), the model
encounters too few instances of the same alternatives in genuinely contrasting
contexts. The 10 epochs of repetition reinforce the same context-target
co-occurrences rather than building context-conditioned discrimination.

**Explains (1):** High-frequency co-occurrences are well-represented, so
local compatibility is easy.

**Explains (2):** Packets provide contrasting contexts but they're genre-alien
and semantically narrow; the model treats them as a separate distribution.

**Explains (3):** No interior solution forms because the training data don't
contain enough naturally contrastive examples to supervise it.

**Explains (4):** Source-rewrite pairs repeat the same propositional content
in different words but don't provide contextual diversity (same facts,
just compressed).

**Key prediction.** Micro-world scores would **vary across trajectories
trained on different data compositions**. Models from breadth-diverse data
would score higher than models from repeated compact-view data. The score
would correlate with the diversity of context-alternative pairings the model
encountered.

## Hypothesis H3: Representational conflation

**Statement.** Alternatives that need context-conditioned binding (on/under,
inside/outside, believes/doubts) are conflated in the model's embedding space
because they share local distributional contexts. The model lacks training
pressure to separate them in dimensions that context would resolve. Both
alternatives are high-probability completions, and the model cannot distinguish
them because their representations project similarly through the MLM head.

**Explains (1):** Local compatibility works through category membership (the
model knows "a spatial preposition goes here") even with within-category
conflation.

**Explains (2):** Packets create local separation pressure for specific pairs,
but this doesn't generalize because the separation is example-specific.

**Explains (3):** Mid-layer computations may briefly access context-specific
information, but the attention patterns at subsequent layers re-conflate
alternatives because their key-query profiles are too similar.

**Explains (4):** Source-rewrite pairs don't present competing alternatives;
they present the same content in different surface forms.

**Key prediction.** Micro-world scores would **correlate with representational
geometry**: specifically, with how well the model separates hidden states for
the two alternatives given each context. A direct measurement: for each
micro-world frame, extract the hidden state at the query position given C1 vs
C2, and measure how much the hidden state moves. If the movement is small,
binding accuracy will be low.

## Hypothesis H4: Compositional depth bottleneck

**Statement.** Context-conditioned binding requires compositional reasoning:
tracking entity states, aggregating events, and drawing inferences across
multiple steps. DeBERTa-v2 at 8×480 with standard bidirectional attention
can process individual facts but cannot compose them into a dynamic world
model at this scale. The binding deficit reflects an architectural limit on
the depth of compositional inference within a single forward pass.

**Explains (1):** Single-step predictions don't require composition.

**Explains (2):** Packets present direct fact-alternative pairings (one step),
which the model can learn. But test items requiring compositional inference
(two+ steps) are not helped.

**Explains (3):** Composition would need persistent state across layers;
each layer's self-attention is computed independently without accumulated
world-model state.

**Explains (4):** Source-rewrite pairs preserve propositional structure and
provide no compositional depth signal.

**Key prediction.** Micro-world items at **different composition depths would
show a sharp accuracy gradient**: single-fact binding (direct context →
answer) would be moderately successful; multi-step composition (context →
inference → answer) would fail sharply. This gradient would be **similar
across checkpoints from the same architecture** regardless of data or
optimizer.

## How a micro-world measurement discriminates

| Measurement | H1 prediction | H2 prediction | H3 prediction | H4 prediction |
|---|---|---|---|---|
| Cross-trajectory variance (same data, different optimizer/scale) | Low | Low | Variable (geometry-dependent) | Low (architecture-dependent) |
| Cross-data variance (different data, same architecture) | Low | High | Moderate | Low |
| Representation–score correlation | Weak | Weak | Strong | Weak |
| Composition-depth gradient | Flat | Flat | Flat | Steep |
| Plateau exposure (when does binding accuracy stabilize?) | Early (<50M) | Late or never (data-limited) | Mid (geometry forms gradually) | Early (architecture limit) |

The strongest discriminator is the **composition-depth gradient combined
with cross-trajectory variance**:

- If depth gradient is steep AND cross-trajectory variance is low → H4
  (architecture limit) dominates.
- If depth gradient is flat AND cross-data variance is high → H2
  (data diversity) dominates.
- If representation geometry predicts scores after controlling for training
  quality → H3 contributes.
- If all trajectories plateau at similar scores regardless of data/architecture
  → H1 (optimization landscape) dominates.

**The hypotheses are NOT mutually exclusive.** The binding deficit likely
results from a combination. The micro-world measurement reveals the
**relative contribution** of each factor.

## Evidence already pointing toward partial answers

- **Against pure H4**: Scale-1.75 (1.75× adapter capacity, effectively deeper
  computation) does NOT improve binding on matched-tokenizer diagnostics.
  GlobalPIQA_parallel 26.21% (scale-1.75) vs 29.13% (matched legal16k base).
  This argues against capacity alone.

- **Against pure H2**: The visible leader achieves EWoK 56.07 with essentially
  the same 10M-word budget. Unless their simplification data provides much
  more contextual diversity than our compact-view data, the data alone isn't
  the bottleneck. (But we can't rule out that their specific data treatment
  IS more contextually diverse.)

- **For H1**: Coupled sparse20's correspondence-free margin compression,
  reproduced by shuffled, is consistent with optimization-basin effects rather
  than learned binding.

- **For H3**: The fact that no tested intervention moves binding while
  maintaining broad competence suggests the model's representation space
  conflates alternatives in a way that can't be resolved without destructive
  retraining.

## Actionable conclusion

The micro-world measurement IS justified because:
1. It produces crossed-sign accuracy per frame, rejecting turnover artifacts.
2. By including frames at different composition depths, it tests H4.
3. By scoring multiple trajectories/data compositions, it tests H1 vs H2.
4. With representation extraction at the query position, it tests H3.
5. It provides a legal-pool diagnostic independent of official labels.

The micro-world should be built BEFORE any new training, and scored on
existing checkpoints only. See `research/plans/representation_and_objectives/micro_world_design.md`.

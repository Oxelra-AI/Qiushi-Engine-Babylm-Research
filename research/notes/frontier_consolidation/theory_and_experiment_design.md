# asset freeze and principles — Toward a Theory of Implicit Multi-View Consistency in Pretraining

## The core empirical finding

The existing controlled BabyLM Strict-Small comparisons use at most 10M corpus words and 100M word exposure. The following findings summarize that evidence:

**When a language model sees text X and its semantically faithful compact paraphrase C(X) in the same training context window, it develops broader and more robust downstream competence than models trained on either the original text alone or on diverse unrelated text of equal word budget.**

This is not a trivial compression finding. The evidence shows:
1. Compact views beat exact repetition (+2.13 equal7, density eval repair)
2. Compact views beat independent source breadth (+0.54 cheap7 at maturity, fw absolute progress decision discipline)
3. The alignment between source and compact view is critical — shuffled source-view pairs are worse (COMPACT_EXPERIENCE inherited evidence, two-seed validation)
4. The source+view MUST be in the same training window (joint visibility 0.999 → 0.572 destroys the benefit, mechanism preservation summary)
5. Explicit consistency objectives (R-Drop, coherence-margin) destroy rather than enhance the learning (-7.27 cheap7, dualview panel readiness and budget, 158-159)

## Why does implicit consistency work while explicit consistency fails?

### Hypothesis: Context-mediated invariance learning

When source text S and compact view C(S) appear in the same training window, the MLM objective creates masked positions within both S and C(S). For a masked token in S, the model can use evidence from C(S) (which contains the same semantic content in different words) to make its prediction. Conversely, for a masked token in C(S), the model uses evidence from S.

This creates an implicit cross-view transfer without any explicit consistency term:
- The model learns that certain semantic features are invariant under the S→C(S) transformation
- This invariance generalizes to other surface-form variations in downstream tasks
- The learning is IMPLICIT because it emerges from the standard MLM objective applied to multi-view contexts

### Why explicit objectives fail

R-Drop adds KL(p₁||p₂) + KL(p₂||p₁) between two dropout passes. Coherence-margin adds softplus(margin + NLL_coherent - NLL_disrupted). Both fail because:

1. **Gradient interference**: The explicit consistency loss creates gradients that directly oppose the MLM gradients on the shared encoder. When these gradients conflict (which they inevitably do, because the optimal representation for consistency ≠ the optimal for MLM), the model compromises on BOTH objectives, achieving neither well.

2. **Representation collapse**: Explicit consistency loss encourages the model to map S and C(S) to identical representations. But the MLM objective NEEDS different representations to predict different surface forms. The conflict drives the model toward degenerate representations.

3. **The implicit mechanism avoids this**: In implicit multi-view learning, the model is NEVER asked to make S and C(S) representations identical. It only uses the cross-view evidence when it helps MLM prediction. The model learns to extract invariant features as a BYPRODUCT of maximizing prediction accuracy, not as a separate objective.

### Formalization

Let θ be the model parameters. For a training context W = [S; C(S)] containing source S and compact view C(S):

**Standard MLM loss on multi-view context:**
L(θ) = -Σᵢ log p(wᵢ | W\{wᵢ}; θ)

where the sum is over masked positions in both S and C(S).

For a masked position wᵢ ∈ S, the model's prediction depends on:
- Local context within S (standard MLM signal)
- Cross-view evidence from C(S) (implicit consistency signal)

The gradient ∂L/∂θ naturally decomposes into:
∂L/∂θ = (local component) + (cross-view component)

The cross-view component is always aligned with the MLM objective (it helps predict masked tokens). This is fundamentally different from an explicit consistency term, which may or may not align with MLM.

**Key prediction**: The cross-view component should be larger when:
- The masked token has a clear semantic equivalent in C(S) (→ strong invariance signal)
- S and C(S) differ in surface form but agree semantically (→ useful invariance to learn)
- C(S) is compact relative to S (→ higher information density, more efficient signal)

**Testable consequence**: The learning benefit should scale with the "semantic overlap × surface form diversity" product between S and C(S), not simply with compression ratio or semantic similarity alone.

## Connection to function-preserving residual capacity

The adapter mechanism (zero-output initialization, scale 1.75) creates a separate learning trajectory from the base model. The narrow 82M peak suggests:

**Hypothesis**: The adapter allows the model to explore a wider function space during early-to-mid training (when cross-view invariance is being learned) but the expanded capacity creates destructive interference in late training (when the model starts overfitting to corpus surface patterns).

If this is correct:
- The peak should shift later with lower adapter scale (less expansion → slower interference)
- The peak should be sharper with higher adapter scale (more expansion → faster interference)
- The peak should disappear without adapters (no expanded function space to create the interference)

These predictions are directly tested by the two running experiments:
- scale1.25 (GPU1): predicts later, broader peak
- scale1.75 seed43122 (GPU0): predicts similar peak timing if the phenomenon is initialization-independent

## Transferable principles for larger LLMs

### Principle 1: Multi-view data preparation
For any pretraining corpus, generate compact paraphrases of a fraction of the text and pack source+paraphrase as adjacent text in training documents. This is a pure data-preparation technique that requires no model modification.

**Scaling hypothesis**: The benefit should increase with corpus size because larger corpora have more redundancy to exploit. A 1B-word corpus might benefit from 10-20% compact-view reinvestment.

### Principle 2: Information density control
The compact views work because they increase the semantic information per word while maintaining faithfulness. This suggests a general principle: pretraining data should have controlled information density, not just quality filtering.

**Connection to existing work**: This relates to but differs from:
- FineWeb/DCLM quality filtering (which removes bad data but doesn't densify good data)
- Data deduplication (which removes identical or near-duplicate text but doesn't create semantic variants)
- Synthetic data generation (which adds new content but doesn't leverage existing content structure)

### Principle 3: Curriculum over information density
The narrow peak phenomenon suggests that the optimal information density changes during training:
- Early training benefits from high-density views (learning broad patterns)
- Late training may benefit from reverting to natural density (preventing overfitting to compressed patterns)

This is a novel form of curriculum learning that operates on INFORMATION DENSITY rather than difficulty, sequence length, or topic.

## Experiments to establish transferability

### Experiment Series A: Seed × Scale Landscape (running now)
- scale1.75/seed43122 + scale1.25/seed43022 (dense 2M checkpoints)
- Tests whether the peak phenomenon is robust to initialization and adapter amplitude

### Experiment Series B: Architecture Transfer (proposed)
- Train GPT-2-style causal LM on the same compact-view reinvest data
- Compare trajectory to MLM-based DeBERTa
- Tests whether the multi-view benefit is architecture-independent

### Experiment Series C: Scale-up Pilot (proposed)
- Apply compact-view reinvestment to the full 100M BabyLM Strict corpus
- Test whether the principle holds at 10× data scale
- This is within BabyLM rules and directly tests scaling

### Experiment Series D: Information Density Analysis (proposed)
- Measure per-word information density (entropy, surprisal) in source vs compact views
- Correlate with the learning benefit
- Tests the theoretical prediction about "semantic overlap × surface form diversity"

## Running experiments

- GPU0: scale1.75 seed43122 dense100M → tests seed robustness of peak
- GPU1: scale1.25 seed43022 dense100M → tests scale dependence of peak
- Both have checkpoints every 2M words for fine trajectory resolution
- Batch evaluator ready: `scripts/batch_trajectory_eval.py`

# Complete mechanism synthesis: orbit binding design

## The causal chain

This synthesis incorporates the causal intervention causal activation intervention, which
provides the first direct evidence that entity selection is causally determined
by a 1D query-match marker at an intermediate layer.  Previous steps established
correlational, ablation, and objective-allocation evidence; causal intervention completes the
causal picture.

### 1. Ordinary training learns bag-level approximation (orbit binding design)

Standard causal-order, full-objective training drives the model toward bag-level
context representations.  BOUND and BAG_INDEP targets produce identical behavior:
MRR ≈ 0.52, B-swap ≈ 0, query-swap ≈ 0, across model sizes 153K–1.2M.  Increasing
answer weight to 64× or training answer-only to 1000 epochs drives context-bag mass
to 0.986 but no query-conditioned binding.

### 2. Causal access enables prospective query-conditioned marking (loss allocation preliminary result)

Query-first presentation, without changing architecture, answer position, latent rows,
or labels, enables near-perfect counterfactual binding by epoch 500: top-4 0.998,
B-swap +11.371, query-swap +11.459.  Query-first BAG_INDEP stays bag-level.

### 3. The marker is a 1D direction at layer 1 (causal intervention)

Donor-query activation intervention proves:
- **Block 1 creates a scalar query-match component** at each attribute position
  in the 64D residual stream.  This is a single direction learned by
  mean(matched) − mean(unmatched), with test accuracy 0.972–0.980.
- **The direction is necessary and sufficient for selection**: d_only replacement
  gives redirect rate 0.998–1.000; orth_only gives 0.000–0.002.
- **Layer specificity**: only L1 produces redirection; L0 and L2 are zero.
  The marker is formed at block 1 (after first attention + FFN) and consumed
  by blocks 2+ and the output head.

### 4. Full-objective continuation erases the marker (020 mechanism synthesis, 025)

Body-only full/context training collapses held top-4 from 0.953 to ≈0.315 in
25 epochs, while body-only answer-only preserves or strengthens binding (body objective ablation).
causal intervention localizes this: the collapsed model's L1 direction accuracy drops from
0.972–0.980 to 0.262–0.322 (chance).  Per-model direction learning confirms the
signal is genuinely erased, not reorganized into a different direction (own-model
L1 accuracy also at chance: 0.262–0.284).

### 5. Downstream readout remains intact (causal intervention)

Cross-model transplant: injecting the preserving model's L1 attribute-position
activations into the collapsed model recovers near-perfect redirection:

| seed | L1 full transplant | L1 d_only transplant |
|---:|---:|---:|
| 43 | **0.946** | **0.840** |
| 100 | **0.982** | **0.860** |

This proves the collapsed model's blocks 2+ and output head remain functional.
The damage is specifically to query-match signal formation at L1.

### 6. Compatible credit allocation preserves the marker (accumulated mechanism synthesis)

Static w = 1/17 (matching equal answer-only/full alternation) preserves the L1
direction perfectly: direction accuracy 0.998–1.000, d_only redirect 1.000.
causal intervention confirms this with causal evidence, not just behavioral metrics.

Alignment controls (answer credit alignment): bag-independent answer targets, context-only
training, and deterministic slot0 all fail to preserve the marker despite learning
context prediction.  The answer credit must reward the query-conditioned relation;
generic answer exposure or reduced context dose is insufficient.

### 7. The marker generalizes to unseen entities (branching experiment, 025)

The direction, learned from train-entity examples, redirects held-entity answers:
- Held-donor L1 d_only: 0.765 (s43), 0.940 (s100) — with orth ≤ 0.110
- Train-donor L1 d_only: 0.910 (s43), 0.885 (s100) — with orth ≤ 0.055

This means the marker is entity-general: it marks "this attribute belongs to the
queried entity" regardless of which entity is involved.

## The principle

**Finite experience can install a reusable relational selector as a low-dimensional
marker at intermediate layers.  Standard broader-objective training selectively erases
this marker through gradient interference while preserving downstream readout.
Compatible credit allocation — maintaining enough relative weight on the relational
objective — preserves the marker during broader learning.**

The marker is:
- **Formed** at block 1 through query-conditioned attention
- **Used** by blocks 2+ for entity-specific answer selection
- **Generalizable** to entities not seen during training
- **Fragile** under context-prediction gradients that dominate the mixed objective
- **Preserved** when the objective maintains relation-aligned answer credit

The matched coefficient for K entities in the query-first orbit-binding substrate:

$$w^* = \\frac{1}{3K + 5}$$

This equals the nominal credit allocation of equal answer-only/full alternation.
Whether this coefficient scales correctly with K (producing marker preservation at
w = 1/(3K+5) for different K) is the next quantitative prediction to test.

## Relation to the reach-asymmetry findings

The corrected reach-asymmetry frame:
- Exact-recurrence liability saturates at half dose (HM ≈ R)
- Wrong correspondence (SHUF) installs broad discounting
- Practiced-relation readout crosses register (Wikipedia delta(N-T))
- Unrelated-source vulnerability is near-register only

These map onto the synthetic mechanism:
- **Recurrence liability** ↔ context-prediction gradients that erase the marker
- **Wrong correspondence** ↔ bag-independent answer targets that collapse binding
- **Practiced-relation readout** ↔ the L1 query-match direction that generalizes
- **Dose saturation** ↔ the coefficient w* at which context credit saturates
  the destructive gradient

The natural-language bridge should test: does form-robust lexical transfer in a
controlled NLM setting depend on the same credit-allocation mechanism, where
replacing relation-aligned training with generic or misaligned supervision
at matched coefficient destroys transfer while natural or relation-matched
supervision preserves it?

## What remains

1. **K-scaling**: test w*=1/(3K+5) at K=2,3,6,8
2. **Natural/semi-natural bridge**: connect to the natural-language relation-learning findings
3. **Model capacity scaling**: test whether the mechanism holds at d=128, d=256
4. **Coefficient optimization**: is w*=1/(3K+5) optimal or just sufficient?
5. **Theory**: derive the marker stability boundary from gradient geometry

## Files

| step | central file | what it establishes |
|---|---|---|
| 014 | notes/orbit_binding_result.md | bag-level attractor |
| 015 | notes/query_first_binding_result.md | query-first enables binding |
| 017 | notes/branching_result.md | context-only destroys, blocked q→ctx also |
| 018 | notes/budget_matched_result_interpretation.md | budget-matched correction |
| 019 | notes/019b_embedding_role_decomposition_analysis.md | embedding role rejected |
| 020 | notes/020_mechanism_synthesis.md | body is the destructive locus |
| 021 | notes/021b_credit_matched_analysis.md | w=1/17 static ≈ interleaved |
| 022 | notes/effective_credit_synthesis.md | alignment required, not generic exposure |
| 023 | notes/alignment_vs_dose_analysis.md | dose alone insufficient |
| 024 | notes/credit_alignment_marker_update.md | activation probe (correlational) |
| **025** | **notes/causal_intervention_analysis.md** | **causal d_only/orth dissociation, transplant, generalization** |
| 025b | data/causal_intervention/per_model_directions.json | signal erasure confirmed |

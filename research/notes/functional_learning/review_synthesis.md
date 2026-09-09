# Review Synthesis: Inherited Evidence and Current Route

## Controlled coordinate transport mechanism

### What is established (related experiments, representation_and_objectives)

In a controlled binary candidate-event harness with 4 relation types (h0, h1, h2, h3),
nonce names, and character-level encoding:

1. **Finite primitive equality evidence** can create a context-invariant identity coordinate.
   - Separate token/query character embeddings: initial matching 0.0
   - Full-alphabet character-pair supervision: matching → 1.0 for all held/fresh names
   - Partial alphabet coverage → partial held-name generalization (earlier analysis)

2. **Sparse absolute anchors** fix the gauge sign on directly supervised relations.
   - h0/h2 have 32 state training rows each; bridge_sign ±1 flips their d_e sign
   - 3 seeds × shared_trunk: all 192/192 direct anchor rows flip correctly

3. **Connected comparison evidence** transports anchor sign to unsupervised relations.
   - Full graph, shared_trunk: h1/h3 row-paired sign reversal = 192/192, held-held closure = 1.0
   - Mixed held-seen orientation reverses with bridge_sign; unchanged coordinates stable

4. **Shared computational representation** is necessary for transport.
   - Untied model: local state/comparison fit, but h1/h3 graph transfer partial/non-causal
   - shared_trunk (shared GRU + separate scalar heads): full transport
   - Scalar head identity is sufficient but not necessary (earlier analysis)

5. **Post-hoc frozen-affine calibration**: fitting z = a·d + b from h0/h2 direct rows
   calibrates h1/h3 to 1.0 in comparison-only models where absolute sign was wrong
   but relative structure was correct in the earlier calibration analysis.

### What is NOT established

- earlier analysis `no_cmp` removes ALL 192 comparison rows, simultaneously eliminating:
  graph connectivity, h1/h3 training exposure, and 192 training rows.
  It cannot separate graph topology from exposure/row-count effects.

- The untied control in the learned-equality setting failed to fit comparison
  (train_cmp = 0.552, stuck at p ≈ 0.5 symmetric point), so it is not a clean
  "local fit without transport" contrast for learned equality specifically.

- The harness still supplies: candidate discovery, binary ownership, same-owner
  comparison semantics, changed/unchanged routing, separate static scorer.

- No natural language binding, parser-free coreference, learned semantic role
  induction, or BabyLM-scale evidence.

### Evidence files (representation_and_objectives)
- `notes/causal_gauge_transport_result.md`
- `notes/corrected_multiseed_gauge_affine_and_next_raw_binding.md`
- `notes/posalign_repair_and_equality_emergence.md`
- `notes/comparison_channel_result.md`

---

## Fixed-budget substitution object

### What is established (frontier_consolidation)

The measured object is a conditional substitution difference:

$$\Delta_T(A,D; B, \mathcal{C}, \tau, s) = Y_T(B - D + A; \mathcal{C}, \tau, s) - Y_T(B; \mathcal{C}, \tau, s)$$

where admission A, displacement D, base corpus B, learner coordinate C, training
horizon τ, and seed s jointly determine the outcome.

1. **Distinct view/breadth beats exact recurrence at late budget**:
   DeBERTa V−C_max = +0.3853, B−C_max = +0.3350, R−C_max = +0.0343

2. **Register substitution is not seed-stable**:
   seed43022 (child_removed − adult_removed) mean exEntity4 = −0.5525
   seed43122 (child_removed − adult_removed) mean exEntity4 = +0.58875
   → The opportunity term is real but basin-conditional

3. **In-corpus adult-prose has positive late movement** vs clean at 70/80/100M:
   exEntity4: +0.180, +0.615, +0.325 (late mean +0.470)
   Own admitted-block loss drop 60→100M: 0.273919

4. **RoBERTa private fitting**: own-arm view loss drops (0.205920 at 80→100M)
   but selected view−clean exEntity5 = −0.6873
   → Material being fitted ≠ target-usable competence

5. **Entity aggregate limitations**: zero-op vs nonzero-op rows can have opposite signs;
   Entity total does not independently prove state tracking (Kim-Schuster analysis).

### What is NOT established

- A decomposition "distinct support × residual work × conversion − opportunity cost"
  remains a conceptual account, not a fitted or independently measured equation

- clean-prior terminal loss rate failed to order breadth above repeat

- No shared-vs-private conversion measurement has been made

### Evidence files (frontier_consolidation)
- `notes/fixed_budget_learning_principle_after_review.md`
- relation_learning: `notes/002_frontier_consolidation_seed43122_incorpus_readout.md`

---

## Literature position

| Topic | What's established | Contribution examined here |
|---|---|---|
| Repeated data diminishing returns | Scaling Data-Constrained LMs: few epochs ≈ fresh; more repetitions saturate | The fixed-budget substitution study adds substitution/displacement + cross-coordinate reversal |
| Model-centric curriculum | Irreducible Curriculum, Influence-driven: perplexity/MMLU not aligned; ordering direction unstable | "Coordinate readiness" here ≠ easy-to-hard |
| Sparsity × repetition | When Data Is Scarce: sparsity changes saturation | The coordinate-transport study's shared vs untied contrast is structural, not parameter-count sparsity |
| Variable binding in Transformers | 2025 Symbolic Programs paper: binding mechanism identified | The coordinate-transport study examines a causal *training path* mechanism for forming/orienting coordinates from finite evidence, not just identifying binding circuitry post-hoc |
| Entity tracking | Kim-Schuster: entity measurement caveats | The fixed-budget substitution study's operation-conditioned analysis |
| CLUTRR / compositional generalization | Graph-based systematic generalization | The coordinate-transport study's graph is internal comparison structure, not input format |

---

## Current bounded principle

> In a controlled setting, finite experience becomes reusable when:
> (i) identity evidence covers primitive types enough to form a context-invariant equality coordinate,
> (ii) sparse absolute anchors fix the gauge on a subset of relations,
> (iii) connected relational constraints propagate relative coordinates through a comparison graph, and
> (iv) a shared computational representation transports the anchored coordinate to unanchored decisions.
> At a macro level, fixed-budget experience value depends on distinct admitted support,
> residual work still learnable, coordinate conversion to target-usable directions,
> and displaced opportunity value.

The coordinate-transport and fixed-budget substitution accounts are connected by the
concept of *target-usable coordinate*: the coordinate-transport study shows how such
a coordinate forms from finite evidence; the fixed-budget substitution study shows
that the value of admitted material depends on whether it enters such a coordinate
rather than a private or recurrence-saturated path.

---

## Key unresolved scientific objects

1. **Graph connectivity vs exposure confound**: earlier analysis no_cmp conflates three removals.
   Need exposure-matched topology intervention.

2. **Representation formation vs graph-mediated orientation vs readout calibration**:
   These three mechanisms contribute jointly. Current evidence doesn't separate them.
   The next topology experiment must distinguish all three.

3. **Shared-vs-private conversion in real BabyLM checkpoints**: Can a gradient-alignment
   or loss-matrix measurement distinguish target-usable from privately fitted work?

4. **Generality boundary**: All coordinate-transport evidence here is in a controlled synthetic harness.
   No evidence yet for natural language, soft routing, or BabyLM scale.

---

## Planned complementary measurements

- A topology/gradient-detach design on the coordinate-transport harness to separate the three mechanisms.
- A fixed-example/fixed-mask cross-arm loss and gradient-alignment matrix on
  existing DeBERTa/RoBERTa checkpoints


---

## Subsequent seed-by-data findings and corrected reading

### Corrected seed×data decomposition
The initial per-row correlations are now treated as correlation observations, not as measured private-learning fractions or causal conversion mechanisms. The missing DeBERTa `D_C_43122` cell was completed and a balanced 2×2 seed×data design (`VIEW/CLEAN × seed43022/43122`) was recomputed. On heldout 60→100M, average same-data cross-seed partial `r=0.381`, average same-seed cross-data partial `r=0.369`, and the independent VIEW-minus-CLEAN data-effect vector across seeds is weak (`raw r=0.0247`, residualized `r=0.0567`). The earlier shared-arm correlation near `0.49` is consistent with its algebraic/shared-arm null, not evidence that the same rows are both data-responsive and seed-sensitive.

### Revised implications for the data-efficient learning principle
1. Low loss-change correlation should not be converted into a percentage of private learning; it measures instability of this row-level loss-change vector under the chosen instrument.
2. A stable benchmark VIEW-CLEAN effect, if it replicates across seeds, would more likely point to representation-level conversion than to a simple reproducible heldout-row MLM-loss-improvement direction.
3. If benchmark effects do not replicate, the fixed-budget substitution ledger must be rebuilt around seed-stable sub-benchmark facts rather than single-seed aggregate movements.
4. RoBERTa cannot yet be called data-inert from the reported correlations because the available `n=3` entries appear to be row/mask replicates, not independent model-seed replicates.

### Open relation-learning questions
- Does seed43122 reproduce the seed43022 VIEW-CLEAN benchmark movement on Supplement, Entity, COMPS, Reading, BLiMP, and EWoK despite VIEW being worse on heldout MLM loss?
- If benchmark movement replicates while row-level MLM-loss data-effect vectors do not, which hidden-state/representation coordinate changes carry the competence movement?

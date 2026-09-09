# Accumulated mechanism synthesis: orbit binding design

## Scientific arc

orbit binding design trace a controlled investigation from bag-level failure, through
query-first causal access as an enabling condition, through held-symbol transfer
loss during full-objective continuation, to the discovery that compatible learning
trajectories exist.  The final picture is a general principle about objective
scheduling and reusable computation, not another catastrophic forgetting observation.

## Core chain of evidence

### 1. Ordinary training learns bags, not bindings (orbit binding design)
Standard causal Transformers (153K–1.2M parameters) trained on entity-specific
permutation-orbit binding converge to bag-level context reading.  Binding-swap and
query-swap remain at zero.  Answer-only weighting up to 64× also fails.  Binding
affects only 1/16 of per-token losses (~0.087 nats), so standard next-token credit
never reaches the selector.

### 2. Query-first causal order enables binding (loss allocation preliminary result)
Moving the query before context—same architecture, same answer position, same latent
rows—enables near-perfect counterfactual binding by epoch 500: answer NLL 0.008,
top-4 0.998, B-swap +11.37, Q-swap +11.46, both-correct 0.999, query-novel
selectivity +0.993.  BAG_INDEP remains bag-level.  Blocking query→context attention
immediately reduces top-4 to chance, supporting prospective query-conditioned context
marking as the mechanism.

### 3. Full-objective continuation recovers familiar but erodes held binding (query first binding compact summary)
From bound query-first checkpoints, full next-token training initially disrupts then
recovers trained-entity binding (3/3 seeds).  But held-symbol transfer degrades
severely: seed100 drops from held_top4=0.953 to 0.238 while trained_top4 reaches
1.000.  Budget-matched controls show fresh training also reaches binding in 2/3
seeds; the preparation advantage is seed-dependent trajectory shaping, not uniquely
retained binding knowledge.

### 4. Held-input drift is not the cause (Step 019b)
The CLM ties input/output embeddings.  Step019c verified that held entity rows
absent from inputs and targets still receive tied non-target softmax gradients.
But Step019b showed that freezing/restoring held rows does not rescue held transfer,
and inserting final held rows into the preparation network preserves its behavior.
Held-input drift exists but is not the load-bearing mechanism.

### 5. Non-answer body pressure is sufficient for collapse (body objective ablation)
Component ablations with fixed input and output tables show:
- `body_only_full`: collapses held binding (h4 0.953→0.301)
- `body_only_context_only`: collapses (0.953→0.253)
- `body_only_answer_only`: preserves and strengthens (0.953→0.953, hB +7.77→+11.45)
- `output_only_full`: preserves (0.953→0.949)
- `input_only_full`: preserves (0.953→0.957)

The destructive update acts on the contextual body through non-answer losses.

### 6. Compatible trajectories exist (accumulated mechanism synthesis) — THE KEY RESULT

Five training schedules (3 seeds × 500 continuation epochs each) all achieve similar
final context CE (~1.23 nats), but differ dramatically in held binding:

| schedule | final held_top4 | final held_B | final ctx_ce |
|---|---:|---:|---:|
| direct_full | 0.418 | +3.11 | 1.233 |
| calibrate_out50 | 0.423 | +3.31 | 1.233 |
| gradual_ctx100 | **0.587** | **+5.70** | 1.229 |
| slow_lr25 | 0.486 | +4.13 | 1.231 |
| **interleaved** | **0.746** | **+7.77** | 1.316 |

**Interleaved answer/full (every other epoch)**: preserves held binding dramatically.
Seed 43: 0.820→0.855, actually strengthened.  Seed 100: 0.953→0.812.
The binding marker is actively maintained by answer-only epochs while context features
accumulate on full epochs.  The slight context CE increase (1.32 vs 1.23) reflects half
as many context-gradient epochs—a tiny cost for dramatically better transfer.

**Gradual context ramp (0.01→1.0 over 100 epochs)**: partially preserves binding.
Seed 43: 0.820→0.742.  Seed 100: 0.953→0.578.  The body finds a more compatible
representation when context pressure is introduced gradually, but without ongoing
answer reinforcement, some binding degrades under sustained full pressure.

**Readout calibration**: fails because out.weight alone cannot learn context prediction
(context CE barely drops from ~34 to ~25 after 50 readout-only epochs).

**Slow lr**: delays but doesn't prevent collapse.  Once full lr engages at be=25,
the shock proceeds.

## The general principle

**The conflict between local prediction and reusable computation is not inherent—it
arises from standard training dynamics.**

The body of a small Transformer can simultaneously support query-conditioned binding
(reusable, counterfactual, transferable) and next-token context prediction (local,
statistical).  The architecture has sufficient capacity.  Standard practice of abruptly
switching to full objectives creates a destructive optimization trajectory that
overwrites the binding marker with local prediction features.

Two alternative learning regimes prevent this:
1. **Active reinforcement**: periodic objective episodes that reward the reusable
   computation maintain it through body gradients that align with the binding marker.
2. **Gradual introduction**: slowly increasing competing objective weight gives the
   body time to find compatible representations.

The key mechanism is credit alignment: reusable computation survives only when
the objective delivers credit to the internal state variables that carry it.  Under
standard full next-token training, the answer position's credit is diluted to 1/16
of the total loss.  Context positions receive 15/16 of the credit and are rewarded
for local prediction features that overwrite the query-conditioned context marker.
Interleaved training restores credit alignment: on answer-only epochs, 100% of
credit goes to the binding-relevant computation.

## Relation to BabyLM and data-efficient learning

This principle applies beyond the synthetic task:

1. **BabyLM pretraining**: under limited data, some training phases may form reusable
   linguistic computations.  If later phases prioritize local prediction (frequent
   patterns, common constructions), they may specialize the network at the expense of
   broader transfer.  Managed objective scheduling could help.

2. **Curriculum effects**: the inherited controlled-binding "curriculum" finding (preparation helps
   more seeds reach binding) was partly a budget artifact (budget matched design note).  The accumulated mechanism synthesis
   result replaces it with a sharper statement: the value of preparation is not just
   initialization but ongoing credit alignment.

3. **Replay and rehearsal**: the interleaved arm connects directly to replay methods
   in continual learning, but with a specific mechanistic explanation: replay works
   not by preventing forgetting of declarative knowledge, but by maintaining credit
   to the internal computation that carries reusable relations.

4. **Architecture implications**: since the conflict is dynamical, not representational,
   architecture changes (modular networks, etc.) are not strictly necessary for retaining
   reusable computation—managed training can achieve the same effect.  But architecture
   that naturally separates reusable and local-prediction pathways could make the
   compatible trajectory more accessible.

## Open questions for strengthening

1. **Interleaving frequency**: does 1:1 ans/full work because it's frequent enough, or
   would 1:4 or 1:8 also work?  What's the minimum reinforcement frequency?
2. **Constant reduced context weight**: does ctx_weight=0.5 permanently (not ramped)
   preserve binding as well as interleaving?
3. **Size scaling**: does the compatible trajectory become easier or harder in larger
   models?  If larger models have more capacity, the compatible basin may be larger.
4. **Natural language bridge**: can the interleaved principle be tested in the BabyLM
   setting?  E.g., mixing binding-relevant training (entity prediction) with full LM.
5. **Activation-level marker**: what does the binding marker look like at the hidden-state
   level?  Is it a separable subspace?  Does the compatible trajectory maintain it in a
   different subspace from the context prediction features?

## Canonical files

- Orbit binding: `data/orbit_binding/results.json`
- Query-first binding: `data/query_first_binding/results.json`
- Branching: `data/branching/results.json`
- Budget-matched: `data/budget_matched_full_objective/results.json`
- Embedding-role: `data/revision_019b_embedding_role_decomposition/results.json`
- Component ablation: `data/component_ablation_after_switch/results.json`
- Body-objective: `data/body_objective_ablation/results.json`
- **Compatible trajectory**: `data/calibration_trajectory/results.json`
- **accumulated mechanism synthesis figure**: `figures/calibration_trajectory.png`
- Main notes: `020_mechanism_synthesis.md`, `calibration_trajectory_analysis.md`

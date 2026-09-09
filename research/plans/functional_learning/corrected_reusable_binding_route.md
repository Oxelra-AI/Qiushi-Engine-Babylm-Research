# rebinding and interpretation correction corrected route: measure and build reusable binding before BabyLM-sign reproduction

## Current bottleneck

source trigger experiment design produced cue-sensitive source-conditioned interactions, but rebinding and interpretation correction showed that those interactions do not yet establish reusable entity--attribute computation.

- `UNPAIRED_SRC` enforced `b != a`, so it is anti-identity rather than independent source-target exposure.
- `IDENT_BLOCKED` removed only a direct attention pattern during training and left possible multi-layer paths.
- Held-out rewrite NLL was best around epoch 50 and then deteriorated while training loss fell.
- Same-token-bag rebinding was weak near epoch 50 and at final checkpoints.
- A final-checkpoint positive-control panel using trained query entities also lacked strong rebinding.

The next controlled contribution should establish when finite examples create a reusable selector-and-transform computation, how long it is retained, and what interventions preserve or destroy it.

## Scientific target

Replace late source-conditioned NLL signs with a direct reusable-computation quantity.

For a context `C` and same-token-bag swapped context `C'`, where the queried entity's attribute changes from `a` to `b`, define the binding swap margin:

\[
B_{swap}=\frac{1}{2}\left[(z_{RWT(a)}-z_{RWT(b)})(C,q) + (z_{RWT(b)}-z_{RWT(a)})(C',q)\right].
\]

A model has reusable entity-specific binding only when `B_swap` is consistently positive across permutation orbits, with good absolute correct-target NLL, stable family mass, and a causal intervention showing dependence on the query-matched event.

## Minimal corrected controlled experiment

### Data construction: permutation orbits

Use contexts with `K=4` entity--attribute facts, episode-unique attributes, and one query entity. For a fixed entity set, attribute set, query, and operation, generate a full or cyclic set of attribute permutations across entities.

Properties:

- Same entity tokens, same attribute tokens, same query, same output family.
- Only the entity--attribute assignment changes.
- Correct answer must follow the query entity's current attribute.
- Entity, attribute, position, query role, and target frequencies are counterbalanced.

Split evaluation into:

1. `seen_query_new_config`: training entities in query role, held-out attribute permutation.
2. `new_edge`: held-out entity--attribute edges with matched marginals.
3. `new_query_role`: entity appeared as context during training but not as query.
4. `held_entity`: held-out entity tokens, used only after positive controls work.

### Arms

All arms should share sequence counts, token counts, target-family rates, optimizer updates, and evaluation probes.

- `BOUND`: target is `RWT(attribute of query entity)`.
- `BAG_INDEPENDENT`: target is sampled from a context attribute slot independent of query; allow equality with the query attribute at the correct `1/K` rate and counterbalance across orbit.
- `MARGINAL_INDEPENDENT`: target sampled from the global attribute marginal independent of context and query.

For auxiliary identity/relation practice after the base binding substrate works:

- `BOUND + COPY_QUERY`: add `COPY(attribute of query entity)` rows.
- `BOUND + COPY_BAG_INDEP`: add query-independent copy of a context attribute.
- `BOUND + COPY_MARGINAL`: add marginal copy tokens.

Only if `COPY_QUERY` lowers the independent-orbit threshold for rewrite binding while the independent copy arms do not can we say identity supervision amortizes a shared selector.

### Measurements across training

Record at frequent checkpoints, not only final epoch:

- correct-target NLL and top-k within RWT;
- RWT family mass and within-family NLL;
- `B_swap`, its mean, variance, and fraction positive;
- query-swap test: same context, different query entity;
- nonquery-swap control: swap two decoy attributes while the query fact remains fixed;
- absent-query/entity control;
- event-order invariance;
- entity-familiarity strata (trained query, context-only entity, held entity only after positive controls succeed).

Fit simple logit probes on balanced evaluations:

\[
z_k=\alpha_k+\beta_{bind}\,1[k=a_q]+\beta_{bag}\,1[k\in A(C)]+\beta_{pos}\,1[k=a_{recent}]+\beta_{prior}\,f_k.
\]

A reusable binding result needs `β_bind` and `B_swap` to dominate bag/position/prior terms.

### Causal intervention

For ordinary Transformer checkpoints:

- single queried-event SRC corruption: replace only the query entity's attribute with an attribute absent from the bag and measure whether logits move to the replacement;
- matched decoy corruption: replace a nonquery attribute with an absent attribute and measure small effect on query target;
- event influence matrix: perturb each fact and measure each RWT logit shift;
- activation/source-slot transplant, if feasible, to test whether query-matched event information mediates output.

If this remains difficult, build a matched slot-reference architecture as a positive-control substrate: encode each fact independently, query attends once to slots, operation head maps selected attribute to output. Compare slot model against the generic Transformer under the same token budget. If slot succeeds with far fewer permutation orbits, that itself is a valuable principle: finite data are used efficiently when architecture provides a selector bottleneck aligned with the reusable variable.

## First implementation step

The proposed small CPU/GPU implementation `orbit_binding.py` requires:

- model options: `transformer` and optionally `slot_model`;
- `--smoke` mode and full mode with seeds `[42,43,100]`;
- data orbit generator with `K=4`, `N_ENT≈10`, `N_ATTR≈12`;
- arms `BOUND`, `BAG_INDEPENDENT`, `MARGINAL_INDEPENDENT`;
- checkpoint metrics at epochs or fixed update counts;
- saved JSON under `data/orbit_binding/` and figure under `figures/`.

Start with a small grid where full run is under ~10 minutes. The purpose is not to maximize final score; it is to find whether and when the substrate forms a measurable binding algorithm, and whether independent permutation coverage beats repetition under the same token/update budget.

## Natural-stream comparison

The neutral-anchor relation-learning analysis suggests a natural-stream comparison: local same-window pairing spends neutral-context target fit while buying relation-specific source-conditioned computations; exact recurrence harms nonidentical targets, whereas restatement helps. A causal next-token CLEAN/REPEAT/REPEAT_SPLIT comparison on the same streams with T/U/N readout would test whether the locality effect is MLM-specific. This was a proposed complementary test, not a substitute for the controlled reusable-binding route.

# Next plan after query first binding compact summary: branch bound query-first checkpoints

## Current state

Step015b: original-order answer-only, weighted-full, and extended answer-only training all learn context-bag membership without query-conditioned selection.

query first binding compact summary: query-first answer-only training with the same 153K Transformer and orbit task reliably reaches near-perfect counterfactual binding on trained entity symbols by 500 epochs. Query-first full training sometimes transitions late; query-first weighted-full with answer weight 16 does not bind by 500 epochs. Query-first BAG_INDEP answer-only remains bag-level.

The result shows that the architecture can implement binding in at least one causal format, so the original-order failure is not simply missing parameter capacity. The unresolved mechanism is why query-first answer-only makes the selector reachable, and whether the learned computation persists when the objective changes.

## Decisive next experiment

Train query-first answer-only models until each seed crosses a stable binding threshold, save model and optimizer states, then branch from the same checkpoint onto identical subsequent streams.

### Acquisition threshold

For each seed, train `query_first / answer_only / BOUND` until all of the following hold at a checkpoint:

- tie-safe top among four context attributes >= 0.95;
- B-swap fraction >= 0.95 and B-swap margin strongly positive;
- Q-swap fraction >= 0.95 and Q-swap both >= 0.95;
- query-novel corruption selectivity >= 0.8.

This avoids retention experiments that begin from a weak or unbound state.

### Branches

From the same bound checkpoint, compare:

1. `continue_answer_only`: same query-first answer-only objective. This measures natural stability under continued direct selector-answer pressure.
2. `query_first_full`: switch to ordinary full next-token objective. This asks whether full sequence modeling preserves the learned selector-router when the answer loss is diluted but still present.
3. `query_first_context_only`: answer weight zero, context/structure tokens only. This asks whether binding persists without direct answer supervision.
4. `query_first_full_no_query_context_attention`: during training and evaluation, context positions are blocked from attending to the early query, while the final IS position can still attend to everything. Immediate failure under this intervention means the active computation depends on query-conditioned context states. Gradual decay means the path is used in acquisition/maintenance but the final state can execute differently.
5. `query_first_answer_only_block_query_context_attention`: same attention block under answer-only. This tests whether answer-only can learn an alternative final-position matcher when query-conditioned context marking is removed.
6. Optional `original_order_answer_only_from_bound`: render the same rows in original order and train answer-only. This tests whether a query-first-bound parameterization accelerates learning or adapts to original order better than random or bag-level starting states.

### Metrics

Use paired latent probes in both query-first and original order when meaningful:

- correct NLL, context-bag mass, within-bag NLL;
- tie-safe top among context attributes;
- query margin;
- B-swap margin/fraction;
- Q-swap margin/fraction/both;
- query-novel corruption selectivity;
- held-entity versions of the standard metrics;
- context CE for branches that include context losses.

Track curves per seed; do not collapse bimodal transitions into only a mean.

### Mechanistic interpretation

- Immediate binding collapse under evaluation-time query-context attention block: query-first success uses active prospective slot marking.
- No immediate collapse but inability to maintain under training-time block: query access is needed for maintenance/credit assignment more than execution.
- Binding survives full but not context-only: direct answer reactivation is needed for retention.
- Sparse selector replay or answer replay can later be added only after these baseline branches locate what decays.
- Original-order adaptation from a bound query-first checkpoint would show that a learned selector/router is a reusable computation that can increase the value of later finite examples.

## Relation to BabyLM

The natural restatement probe shows a natural REPEAT-side source-conditioned cost but not a stable natural VIEW-side benefit. The corrected mechanism wording is source-recognition/content pull rather than exact copying. This sets a boundary: before linking to natural streams, establish whether the controlled selector-router computation can be acquired, retained, and reactivated. Then the bridge can ask whether natural-language adjacency supplies query-first-like access, selector reactivation, or only source-recognition without full counterfactual binding.

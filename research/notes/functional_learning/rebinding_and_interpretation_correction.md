# rebinding and interpretation correction rebinding and interpretation correction

## Why this correction is necessary

source trigger experiment design produced a real and useful signal, but the causal reading in the first result note was too strong in two places.

1. **`UNPAIRED_SRC` is not source--target independence.** In source trigger experiment design the comparator used `SRC(a) -> SRC(b)` with `b != a` for every extra row. That preserves contexts and SRC-family target rates, but it teaches an anti-identity relation rather than removing dependence between source and target. Therefore `IDENT_FULL - UNPAIRED_SRC` cannot be read as pure identity-practice versus independent source exposure.

2. **`IDENT_BLOCKED` is not a complete removal of local source access.** The mask blocks direct final-segment positions `[ENT(query), cue, IS]` / zero-based positions `[14,15,16]` from attending to the source event positions of the queried entity during the identity rows; `SEP` itself is at zero-based position 13 and was not included in that blocked target set. In a multi-layer Transformer, information can still travel through unblocked positions such as separator or other context tokens across layers. Therefore `IDENT_FULL - IDENT_BLOCKED` tests removal of a direct edge pattern in the training substrate, not a proof that all local source paths are necessary.

3. **The old T/U probe scored the old answer after a binding change.** This is valid for one question: whether the presence of the true source changes the score of an old target differently under identity versus comparator training. It does not establish that the model follows the queried entity's current context binding. For reusable computation, the more direct question is: with the same token bag but a changed entity--attribute binding, does the model move probability to the newly correct rewrite token?

These corrections do not erase the source trigger experiment design data; they narrow its meaning. The cue-dependent interaction remains an observed interaction, but it is not yet a clean causal isolation of identity practice against independence or a local-source-access requirement.

## Existing source trigger experiment design learning trajectory already changes the scientific emphasis

The source trigger experiment design final-epoch interaction emerged after the held-out predictive competence had deteriorated. Recomputing the typed-cue curve from `data/source_trigger/results.json` gives the following seed-averaged trajectory for the true-source old-target probe:

| Arm | e50 RWT NLL | e100 | e150 | e200 | e250 | e300 | e300 MRR |
|---|---:|---:|---:|---:|---:|---:|---:|
| IDENT_FULL | 2.43 | 4.09 | 6.41 | 7.03 | 7.00 | 7.03 | 0.43 |
| IDENT_BLOCKED | 2.44 | 3.67 | 7.22 | 8.11 | 8.26 | 8.28 | 0.39 |
| UNPAIRED_SRC | 2.37 | 3.87 | 6.95 | 7.53 | 7.80 | 7.93 | 0.40 |

Training loss continues to fall while held-out rewrite NLL rises sharply after roughly epoch 50. The late positive typed-cue interaction (`D_ident_full` on old-target RWT NLL: about 0 at e50, +0.272 at e150, +0.283 at e300) is therefore not evidence that more training builds useful generalization. It is a difference among models that are increasingly overconfident on a weak held-out heuristic.

This shifts the bottleneck: a data-efficient learning principle must explain acquisition and retention of useful competence. It is not enough to show that late overfit models differ in a source-conditioned contrast.

## Final-model rebinding probe

I wrote `scripts/rebinding_analysis.py` and evaluated the source trigger experiment design saved final models on a paired rebinding probe. For each probe:

- Original `T`: held query entity has attribute `old`, another context entity has `new`.
- Swapped `S`: the same context token bag is preserved, but the held query entity now has `new` and the other entity has `old`.
- The model is scored on both `RWT(old)` and `RWT(new)` under both contexts.
- Reusable binding would prefer `RWT(old)` in `T` and `RWT(new)` in `S`; pair success requires both margins to be positive.

Final-model aggregate from `data/rebinding_analysis/results.json`:

| Cue | Arm | T old-vs-new logit margin | S new-vs-old logit margin | pair success | S new NLL | S top-new |
|---|---|---:|---:|---:|---:|---:|
| typed | IDENT_FULL | -0.318±0.389 | -0.206±0.426 | 0.104±0.009 | 7.53 | 0.169 |
| typed | IDENT_BLOCKED | -0.495±0.439 | +0.302±0.171 | 0.131±0.012 | 8.47 | 0.171 |
| typed | UNPAIRED_SRC | -0.223±0.141 | +0.096±0.128 | 0.158±0.027 | 8.73 | 0.173 |
| uninformative | IDENT_FULL | -0.437±0.111 | -0.013±0.129 | 0.141±0.009 | 13.62 | 0.185 |
| uninformative | IDENT_BLOCKED | -0.212±0.492 | +0.182±0.242 | 0.144±0.012 | 9.74 | 0.173 |
| uninformative | UNPAIRED_SRC | -0.391±0.090 | +0.251±0.147 | 0.129±0.015 | 14.56 | 0.160 |

The final models do not show robust entity-specific rebinding. The two correct-direction margins do not sum to a reliably positive value, and IDENT_FULL is not better than the imperfect comparator on margin-sum or pair-success measures. Pair success is reported descriptively; because the two comparisons share a token bag and are correlated, it does not have a simple `1/10` null value. IDENT_FULL can have lower absolute `S_new_nll` than UNPAIRED_SRC, but because UNPAIRED_SRC is anti-identity and all arms are in a poor late-generalization state, this does not by itself establish reusable binding.

The figure `figures/rebinding_trajectory.png` summarizes the old-target trajectory and final rebinding margins.

## Current scientific interpretation

The strongest current synthetic result is not “within-sequence heterogeneity is the missing BabyLM mechanism.” That remains a candidate condition, but rebinding and interpretation correction shows a more basic problem: before adding another ingredient, we need to identify whether the synthetic models acquire a reusable entity--attribute correspondence that can be retained and rebound under changed context bindings.

A more faithful current statement is:

- Identity/source practice changes learned behavior in a way that is cue-sensitive and source-conditioned, but the source trigger experiment design comparator and mask do not cleanly isolate identity versus independence or all local source access.
- The late source-conditioned interaction appears after held-out NLL has deteriorated, so its contribution to data-efficient learning is ambiguous unless linked to earlier or more durable competence.
- Final saved models do not robustly rebind same-bag context changes to the newly correct rewrite token.
- The next valuable question is not whether we can force a BabyLM-like sign in this substrate, but what computation is being acquired and lost: entity-specific binding, context-token familiarity, source-family confidence, or a brittle heuristic.

## Typed-cue rebinding trajectory rerun

The trajectory rerun completed successfully. `scripts/rebinding_trajectory_rerun.py` repeated only the typed source trigger experiment design substrate and added the paired rebinding probe at epochs 1/25/50/75/100/150/200/250/300. It did not add within-sequence heterogeneity or another training ingredient. Results are in `data/rebinding_trajectory_typed/results.json`; the trajectory figure is `figures/rebinding_trajectory_rerun.png`.

The result is not an early acquisition followed by clean loss of a good binding mechanism. Rebinding was weak even near the absolute-NLL optimum:

| Arm | epoch | old-target NLL | original T old-vs-new margin | swapped S new-vs-old margin | margin sum | pair success |
|---|---:|---:|---:|---:|---:|---:|
| IDENT_FULL | 50 | 2.36 | +0.06 | -0.05 | +0.01 | 0.131 |
| IDENT_BLOCKED | 50 | 2.46 | +0.00 | -0.01 | -0.01 | 0.040 |
| UNPAIRED_SRC | 50 | 2.38 | +0.02 | -0.02 | 0.00 | 0.061 |
| IDENT_FULL | 100 | 4.13 | +0.18 | -0.37 | -0.19 | 0.096 |
| IDENT_FULL | 300 | 7.19 | +0.33 | -0.62 | -0.29 | 0.124 |
| IDENT_BLOCKED | 300 | 7.92 | -0.37 | +0.27 | -0.10 | 0.145 |
| UNPAIRED_SRC | 300 | 8.09 | +0.09 | -0.40 | -0.31 | 0.151 |

At epoch 50, all arms have near-zero binding margin sums: the prediction barely changes in the correct direction when the held entity's attribute is swapped. Later IDENT_FULL increasingly favors the old attribute in the original context but fails to follow the queried entity after the swap; its swapped new-vs-old margin reaches `-0.62` by epoch 300. Absolute NLL on the newly correct swapped target simultaneously rises from `2.48` at epoch 50 to `7.88` at epoch 300.

The old-target source trigger experiment design interaction also appears only after this deterioration. In the rerun, typed `D_ident_full` on old-target NLL was `-0.005` at epoch 50, `+0.239` at epoch 150, and `+0.548` at epoch 300; the analogous within-RWT values were `-0.008`, `+0.237`, and `+0.685`. Run-to-run magnitudes differ from the original source trigger experiment design because stochastic minibatch orders were regenerated, but both runs agree on the trajectory: the interaction is absent near peak absolute competence and grows during late overfitting.

IDENT_FULL's late rebinding margin is not improved relative to UNPAIRED_SRC: its swapped new-vs-old margin difference was `-0.223` at epoch 300, with a pair-success difference of `-0.027`. The comparison remains descriptive because UNPAIRED_SRC teaches anti-identity, but it supplies no evidence that identity practice improves reusable rebinding in this substrate.

The current inference is therefore stronger than the final-checkpoint analysis alone: this source trigger experiment design substrate did not learn a robust entity-specific correspondence at any measured useful budget. It learned modest ranking structure and then a progressively overconfident old-binding/context heuristic. The next construction should repair the representation-learning problem and use a corrected independent-target comparator before returning to BabyLM-sign reproduction.

## Corrected next experimental design, after the rerun

A corrected comparison should replace the two weak source trigger experiment design contrasts with cleaner controls:

1. **Independent target comparator.** Use `SRC(a) -> SRC(b)` where `b` is sampled independently from the marginal target distribution, allowing `b=a` at the correct base rate, or better use a counterbalanced target assignment independent of the source attribute rather than enforcing `b != a`. This avoids teaching anti-identity.

2. **Stronger source isolation.** Either block all paths from the queried source event to the query/output positions across layers during training/evaluation, use sequence separation that prevents same-window attention, or perform an activation/logit intervention that removes the source event contribution at evaluation. The current direct-edge mask is insufficient as a full local-access intervention.

3. **Trajectory-first measurements.** Score absolute correct-target NLL, family mass, within-family ranking, and rebinding margin throughout training, not only at the final epoch. Prefer the budget where useful held-out competence is high and before collapse unless the collapse itself is the phenomenon under study.

4. **Intervention target.** If useful rebinding appears and then collapses, test whether regularization, early stopping, split/locality removal, or cue/task structure preserves rebinding while reducing source-token or old-binding attraction. If rebinding never appears, rebuild the substrate around a stronger relation-learning pressure before asking BabyLM-bridge questions.

## Existing-final-checkpoint positive-control panel

independent_review verification raised an important scope issue: the first rebinding and interpretation correction rebinding probes used `HELD_E=[6,7]` as query entities, while training only used entities `0..5` as possible query entities. Therefore weak held-query rebinding could have been a role-transfer failure rather than a substrate-wide binding failure.

I wrote and ran `scripts/existing_checkpoint_panel.py` on the source trigger experiment design saved final models. It evaluates same-token-bag rebinding for four groups:

- `train_query_alltrain`: query entity and all context entities are training entities;
- `held_query_traindecoys`: held query with trained decoys;
- `train_query_helddecoy`: trained query with a held nonquery decoy;
- `held_query_heldpartner`: held query with the other held entity as the swap partner.

The trained-query positive-control panel still does not show strong rebinding. Selected final-checkpoint aggregates:

| Cue | Arm | Group | T old-vs-new margin | S new-vs-old margin | margin sum | pair success | S new NLL |
|---|---|---|---:|---:|---:|---:|---:|
| typed | IDENT_FULL | train_query_alltrain | +0.063 | +0.096 | +0.159 | 0.127 | 6.97 |
| typed | IDENT_BLOCKED | train_query_alltrain | +0.264 | +0.075 | +0.339 | 0.137 | 8.47 |
| typed | UNPAIRED_SRC | train_query_alltrain | +0.185 | +0.518 | +0.703 | 0.166 | 8.74 |
| typed | IDENT_FULL | held_query_traindecoys | -0.178 | +0.053 | -0.125 | 0.105 | 7.27 |
| typed | IDENT_FULL | train_query_helddecoy | -0.161 | +0.131 | -0.030 | 0.124 | 7.13 |
| typed | IDENT_FULL | held_query_heldpartner | -0.169 | +0.280 | +0.111 | 0.135 | 7.33 |
| uninformative | IDENT_FULL | train_query_alltrain | +0.053 | +0.003 | +0.056 | 0.130 | 13.56 |
| uninformative | IDENT_BLOCKED | train_query_alltrain | +0.066 | +0.239 | +0.305 | 0.155 | 9.77 |
| uninformative | UNPAIRED_SRC | train_query_alltrain | +0.050 | +0.247 | +0.297 | 0.159 | 15.03 |

The imperfect UNPAIRED_SRC comparator sometimes shows the strongest train-query margin sum, which reinforces that source trigger experiment design's comparator is not a clean source-independence baseline. More importantly, IDENT_FULL does not display a large positive trained-query rebinding signal. Thus the weakness is not merely that held entities were unseen as queries; the final source trigger experiment design substrate does not expose a robust selector-and-rewrite computation even where query entities were practiced. This does not prove no binding trace exists, because the probe is final-checkpoint and the substrate may contain weak rank information, but it blocks any strong reading of source trigger experiment design as a reusable-binding result.

## Peer body objective ablation update and consequence

The compact rewrite analysis added a neutral ordinary-text source anchor `N` to its compact rewrite readout. This changed the natural-language side in a useful way. Local same-window pairing has a broad neutral-context target-fit price relative to split exposure: `R-RS ΔN=+0.3027` and `V-VS ΔN=+0.2688` on seed43022, while `U-N` shifts are small (`-0.0282`, `+0.0581`). The price is therefore visible under a neutral source, not mainly unrelated-compact-neighbor damage. What local pairing buys depends on the relation: exact recurrence buys harmful related-source identity use (`R-RS Δ(T-N)=+0.692`, `Δ(U-T)=-0.720`), while restatement buys helpful source-conditioned content use (`V-VS Δ(T-N)=-0.596`, `Δ(U-T)=+0.654`). The CHILDES surface-held-out variation bridge was weak at all-bin nonoverlap (`V-C +0.012±0.037`, `R-C -0.020±0.068`, high bin only 17 pairs), so it should not be used as a natural replication.

This evidence increases the value of testing local pairing in next-token training on the same natural streams, but it does not remove the synthetic substrate problem. Another synthetic comparison targeting the BabyLM sign first requires a valid measurement of reusable computation. Two complementary next routes are now valuable:

1. **Controlled substrate repair:** construct a counterfactual permutation-orbit relation task where same-bag contexts force the answer to follow the queried entity's current binding, with independent-target rather than anti-identity controls, trajectory-first metrics, and stronger causal interventions.
2. **Natural-stream bridge:** given the original stream and scorer, run CLEAN/REPEAT/REPEAT_SPLIT under causal next-token training with the same T/U/N readout. This would test whether the BabyLM exact-recurrence identity cost and locality are specific to MLM masking or persist in next-token training. It should be a bridge check, not a substitute for the controlled rebinding mechanism.

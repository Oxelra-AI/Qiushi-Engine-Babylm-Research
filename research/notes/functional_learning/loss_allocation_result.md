# loss allocation preliminary result result: answer-loss allocation improves context-bag retrieval, not entity binding

## Why this experiment was necessary

orbit binding design showed that the ordinary full next-token objective produced bag-level behavior on the permutation-orbit binding task. That did **not** justify a strong landscape explanation, because the binding answer receives only one of sixteen non-PAD next-token losses. Perfect entity-specific binding improves the ideal bag predictor's average full-objective loss by only `log(4)/16 = 0.0866` nats per supervised token. loss allocation preliminary result therefore tested whether the same task and baseline Transformer acquire counterfactual binding when answer-token learning pressure is increased.

## First run and repair

The first script, `scripts/loss_allocation_binding.py`, confirmed the answer-position indexing and showed a preliminary null result: answer-only and answer-weighted training for 200 epochs lowered answer NLL to the same range as the ordinary full objective, but left direct binding scores near zero. It had two important weaknesses: different finite streams across arms and a tie-biased top-among-four metric.

The repaired experiment, `scripts/revision_015b_paired_loss_allocation.py`, fixes those issues:

- all BOUND objective arms consume the same generated contexts and minibatch order for each seed/epoch;
- the sequence layout is asserted (`IS_POS=15`, answer at position 16);
- metrics include mass on the four context attributes (`bag_mass`), within-bag NLL, tie-safe top-among-four, B-swap, Q-swap, and query-specific corruption selectivity;
- answer pressure is tested by `w16_200`, `w64_200`, `ans_only_200`, and `ans_only_1000`;
- `bag_ans_only_1000` is the answer-focused null where targets are random context attributes independent of query;
- `ans500_full500` checks whether any answer-only acquisition at epoch 500 persists under the ordinary full objective.

Data: `data/revision_015b_paired_loss_allocation/results.json`  
Figure: `figures/revision_015b_paired_loss_allocation.png`  
independent_review design verification: 

## Main repaired result

Answer allocation changes *how quickly and how strongly the model retrieves the context bag*, but it does not produce entity-specific binding.

| Arm | final correct NLL | bag mass | within-bag NLL | tie-safe top4 | query margin | B-swap | Q-swap margin | Q-swap frac | query-specific novel follow |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full_w1_200 | 1.613 | 0.925 | 1.535 | 0.249 | +0.004 | +0.016 | -0.001 | 0.499 | -0.000 |
| w16_200 | 1.542 | 0.964 | 1.505 | 0.257 | -0.009 | +0.016 | +0.000 | 0.486 | -0.003 |
| w64_200 | 1.523 | 0.967 | 1.489 | 0.256 | -0.003 | +0.016 | +0.001 | 0.503 | -0.004 |
| ans_only_200 | 1.521 | 0.972 | 1.493 | 0.250 | -0.012 | +0.011 | +0.001 | 0.499 | -0.006 |
| ans_only_1000 | 1.444 | 0.986 | 1.430 | 0.273 | +0.021 | -0.002 | +0.000 | 0.495 | +0.002 |
| bag_ans_only_1000 | 1.471 | 0.986 | 1.457 | 0.236 | -0.019 | -0.000 | +0.000 | 0.503 | +0.001 |
| ans500_full500 final | 1.452 | 0.982 | 1.433 | 0.266 | +0.006 | -0.003 | +0.001 | 0.517 | +0.000 |
| ans500_full500 phase0 | 1.487 | -- | -- | 0.251 | -- | -0.004 | -0.000 | -- | -0.000 |

Reference values: bag-uniform answer NLL is `log(4)=1.386`; random top among four is 0.25; true binding should raise top4 far above 0.25, make B/Q-swap margins positive, and make corruption follow a novel attribute at the query slot more than at a decoy slot.

The strongest answer-only arm after 1000 epochs reaches high context-bag mass (`0.986`) and lower NLL (`1.444`), but its direct binding signals remain null: tie-safe top4 `0.273`, B-swap `-0.002`, Q-swap margin `+0.000`, Q-swap fraction `0.495`, and corruption selectivity `+0.002`. The BAG_INDEP answer-only null reaches the same context-bag mass and similar NLL without query-conditioned targets.

The retention arm is also informative: after 500 answer-only epochs, there is no binding to retain (`phase0` top4 `0.251`, B-swap `-0.004`, Q-swap margin `-0.000`). The final full-objective phase improves context-token CE and keeps answer NLL near the bag level, but does not create binding.

## Interpretation

The objective-allocation alternative has now been tested more directly. Under this architecture and orbit task, scalar answer-token pressure alone is not enough to acquire reusable entity-attribute binding. The models use the target answer loss to learn which four attributes occur in the context, concentrating almost all probability mass on that bag, but they do not condition selection on the queried entity.

This is a stronger and cleaner negative than orbit binding design, but its scientific meaning is still bounded:

- It does **not** prove the Transformer cannot represent a binding solution.
- It does **not** isolate a specific local-minimum or landscape mechanism.
- It does **not** show that finite examples are inherently unable to teach binding.
- It shows that ordinary next-token training, answer reweighting up to 64×, and answer-only training to 1000 epochs all convert the same finite orbit experience into context-bag statistics rather than a query-conditioned selector.

## What this changes in the research thread

The current bottleneck is no longer simply insufficient answer-token loss. The model receives strong direct answer pressure, but the gradient still organizes around a lower-order statistic: attribute membership in the context. The missing object is a reusable selector computation: read the query entity, match it to a context entity, and route the adjacent attribute into the answer distribution.

The next work should establish a positive binding solution before explaining the failure mechanism. Useful positive-control routes are:

1. **Auxiliary selector pressure:** add a small slot head at the answer-position hidden state and train it to identify which context slot contains the query entity, while still evaluating answer prediction in the original format. If slot accuracy becomes high but answer binding remains weak, the difficulty is routing the selected attribute through the LM head; if both improve, the result shows that finite examples can form binding when the objective supplies an intermediate selector.
2. **Query-before-context format:** move the query entity before the four context triples. If binding becomes easy, the problem is partly causal credit assignment and multi-hop temporal routing, not abstract capacity.
3. **Minimal architecture intervention:** add explicit slot representations or pointer-style attention and compare sample efficiency to the baseline. This would test whether an inductive bias for entity matching converts finite experience into reusable computation.

A strong data-efficient learning principle can grow from this sequence only if it includes the acquisition condition: examples become efficient when the learner's objective and architecture create pressure for the right reusable computation, not merely when the examples contain the relation.

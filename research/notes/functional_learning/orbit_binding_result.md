# orbit binding design Result: Permutation-orbit binding reveals bag-level convergence

## Summary

A 3-layer causal Transformer (153K params) trained on entity-specific binding targets
for 200 epochs × 500 sequences per epoch converges to **bag-level context representation**
rather than entity-specific binding. Both BOUND (entity-specific targets) and BAG_INDEP
(random context-attribute targets) achieve indistinguishable performance on all metrics,
including the entity-binding tests. MARGINAL (context-independent targets) converges to
its expected uniform level, confirming the evaluation probes work correctly.

## Key finding

**The model learns which attributes appear in the context (bag membership) but not which
entity has which attribute (entity-attribute binding).**

Evidence:
- B_swap (same token bag, different binding): +0.014±0.027 for BOUND versus +0.021±0.027
  for BAG_INDEP. Both essentially zero. The model treats same-bag permutations identically.
- Query-swap (same context, different query entity): 0.003 for BOUND, 0.000 for BAG_INDEP.
  The model ignores the query entity.
- Corruption selectivity: query_novel ≈ decoy_novel for both arms (BOUND: 0.219 vs 0.213;
  BAG_INDEP: 0.223 vs 0.221). The model responds equally to changing any entity's attribute.
- Final MRR within RWT: 0.515 (BOUND) and 0.529 (BAG_INDEP), both close to the bag-uniform
  reference level of 0.521.
- Training loss is identical for BOUND and BAG_INDEP at every epoch (~1.31 at e200).

## Trajectory

Both BOUND and BAG_INDEP follow the same trajectory:
1. Early (e1-20): learn sequence grammar and basic vocabulary
2. Middle (e30-100): learn bag membership (which attributes are in context), MRR rises from
   ~0.24 (chance) toward ~0.45
3. Late (e100-200): refine bag-level prediction, MRR approaches bag-uniform (0.521)
4. Throughout: B_swap stays at zero, query-swap stays at zero

The model never enters a phase of entity-specific binding. There is no early acquisition
followed by loss; the entity-specific computation simply does not appear.

## Final metrics (3-seed average ± std)

| Arm | correct_nll | MRR | B_swap | query_swap | q_novel | d_novel |
|-----|-------------|-----|--------|------------|---------|---------|
| BOUND | 1.643±0.057 | 0.515±0.008 | +0.014±0.027 | 0.003 | 0.219 | 0.213 |
| BAG_INDEP | 1.572±0.052 | 0.529±0.013 | +0.021±0.027 | 0.000 | 0.223 | 0.221 |
| MARGINAL | 2.519±0.025 | 0.271±0.018 | 0.000±0.003 | 0.000 | 0.000 | 0.000 |
| bag-uniform ref | 1.386 | 0.521 | 0 | - | - | - |
| marginal ref | 2.485 | 0.259 | 0 | - | - | - |

Held-entity performance matches standard (BOUND: held_mrr=0.525 vs std_mrr=0.515),
confirming bag-level generalization transfers to unseen entities.

## Why this matters for the data-efficient learning principle

1. **The ordinary full next-token objective did not produce entity-specific binding.** The BOUND arm
   receives the correct entity-attribute target on every training sequence, yet converges to
   the same bag-level behavior as BAG_INDEP under the measured probes. This is a full-objective
   acquisition failure within the tested budget, not by itself a proof of unavailable capacity
   or a particular landscape mechanism.

2. **Bag-level statistics absorb most of the measurable answer improvement.** The model reduces
   answer NLL efficiently by learning which attributes appear in the context, without selecting
   the queried entity's attribute. Because the binding answer is only one of sixteen supervised
   next-token targets, loss allocation preliminary result tests whether stronger answer-token pressure changes this behavior.

3. **This explains source trigger experiment design's lack of rebinding.** source trigger experiment design's models operated in the same
   regime: they learned which source tokens appeared in context and responded to bag-level
   changes, but never acquired entity-specific binding that could survive same-bag
   permutations. source trigger experiment design's late-epoch source-conditioned interactions were differences
   among models learning slightly different bag-level heuristics, not evidence of reusable
   entity correspondence.

4. **Content-addressable lookup is the bottleneck.** The task requires: (i) read query
   entity identity from position 14, (ii) match it to one of 4 context entities at
   positions 1/4/7/10, (iii) extract the adjacent attribute. This indirect addressing
   requires the attention mechanism to be conditioned on previously read content.
   A 3-layer Transformer with 2 heads and d=64 does not learn this computation from
   100K entity-specific examples under the ordinary full next-token objective.

## What changes could enable binding

- **More capacity**: larger d, more heads, more layers may provide enough representation
  space for indirect addressing. Next test: d=128, nh=4, nl=4 (BOUND only).
- **Explicit binding pressure**: auxiliary loss on entity matching, or training data that
  creates stronger gradient toward content-based attention.
- **Architectural inductive bias**: slot-based models, copy mechanisms, or explicit
  entity-attribute separation could provide the selector bottleneck.
- **Curriculum**: start with K=2 entities (easier matching), increase to K=4.
- **Training dynamics**: learning rate schedules that prevent premature convergence to
  bag-level local minima.

## Connection to BabyLM

In natural language, "bag-of-words" statistics similarly provide a strong baseline that
small models exploit before learning compositional structure. The finding that entity
binding requires more than entity-specific supervision suggests that BabyLM data efficiency
depends not only on what the data teaches (content) but on whether the model can form and
maintain the right computational structure (architecture × optimization × data interaction).

The general principle being tested: limited data is efficient when it installs reusable
computations (binding, matching, composition) rather than just statistical summaries.
orbit binding design shows that explicit binding targets inside the ordinary full next-token objective
are not enough for this small Transformer to form measurable binding. loss allocation preliminary result asks whether
answer-loss allocation changes that result before the work turns to architectural support,
auxiliary selector pressure, or other interventions.

## Files

- Script: `scripts/orbit_binding.py`
- Data: `data/orbit_binding/results.json`
- Figure: `figures/orbit_binding.png`
- Design: `notes/orbit_binding_design.md`


## Step014b Capacity sweep result

Tested BOUND arm across 4 model sizes (3 seeds × 300 epochs each):

| Config | Params | correct_nll | MRR | B_swap | query_swap |
|--------|--------|-------------|-----|--------|------------|
| small (d=64, nh=2, nl=3) | 153K | 1.519 | 0.517 | -0.003 | 0.007 |
| wide (d=128, nh=4, nl=3) | 602K | 1.535 | 0.504 | -0.019 | 0.010 |
| deep (d=64, nh=2, nl=6) | 304K | 1.522 | 0.513 | -0.000 | 0.003 |
| large (d=128, nh=4, nl=6) | 1.2M | 1.518 | 0.539 | -0.009 | 0.000 |
| bag-uniform ref | - | 1.386 | 0.521 | 0 | - |

**No model achieves entity-specific binding.** B_swap stays at zero and query-swap
stays near zero for all configurations, including the 1.2M-parameter model (8× the
baseline). All models converge to identical bag-level performance: correct_nll ≈ 1.52,
MRR ≈ 0.51-0.54, within noise of the bag-uniform reference.

This shows that increasing size within the tested range does not rescue full-objective
binding. Even with 1.2M parameters, 6 layers, and 4 attention heads, the standard Transformer
trained with AdamW on fresh random permutation-orbit binding examples still behaves at the
bag level under B-swap and Q-swap probes. This does not isolate capacity, objective allocation,
or landscape mechanism by itself; loss allocation preliminary result addresses the objective-allocation alternative.

### Implications

1. Full-objective binding remains absent after a size increase from 153K to 1.2M parameters.
2. Standard Transformer + AdamW training in this setup reaches context-bag behavior much more
   readily than entity-level selection.
3. The next explanations to separate are objective allocation, selector-oriented auxiliary
   pressure, architectural support for entity matching, and task/data structure that makes
   bag-level prediction insufficient.
4. For BabyLM: natural language may provide pressure against bag-only prediction through richer
   contexts and downstream measurements, but that bridge still needs direct testing.

### Files

- Script: `scripts/revision_014b_capacity_sweep.py`
- Data: `data/revision_014b_capacity/results.json`
- Figure: `figures/revision_014b_capacity.png`


## loss allocation preliminary result correction to the orbit binding design interpretation

The orbit binding design measurements remain valid evidence that the ordinary full next-token objective produced bag-level behavior on the orbit-binding substrate. The stronger wording that this already establishes a bag-level local minimum or an optimization-landscape explanation is too strong.

In the implemented sequence the binding answer is one of sixteen non-PAD next-token targets. Perfect entity-specific binding improves the ideal bag predictor by only `log(4)/16 = 0.0866` nats per supervised token under the full objective. Similar aggregate training losses for BOUND and BAG_INDEP therefore do not show that entity-specific supervision is wasted or that the Transformer cannot represent the computation. They show that ordinary next-token training did not produce measurable same-bag counterfactual binding within the tested budget.

loss allocation preliminary result tests the sharper question: whether the same task and architecture acquire reusable binding when the answer token receives more learning pressure, and whether any acquired binding persists when ordinary full next-token training resumes. Until that is resolved, orbit binding design should be cited as a full-objective negative result and a motivation for studying objective allocation, not as a completed explanation of the training landscape.

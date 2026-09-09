# branching experiment Result: Branching from bound query-first checkpoints

## Summary of central findings

1. **Full-objective recovery is reliable from a bound start.** All three seeds
   recover to near-perfect binding after the full next-token objective initially
   disrupts it. In query first binding compact summary, full training from scratch was seed-bimodal (1/3
   perfect, 1/3 partial, 1/3 bag-level by 500 epochs). From a bound start,
   3/3 seeds reach near-perfect binding under the full objective.

2. **Binding collapses without answer pressure.** Context-only training
   (zero answer weight) destroys binding within 25 epochs. The selector is
   an actively reinforced computation, not a structural property of the
   parameters.

3. **Prospective marking is the primary binding mechanism.** The bound
   checkpoint loses binding immediately when evaluated with blocked
   query→context attention (blocked top4 ≈ 0.25, chance level), even though
   standard-mask evaluation shows near-perfect binding. The computation
   depends on query-conditioned context states.

4. **Transfer to original order is partial.** 1/3 seeds shows progressive
   original-order binding from a bound initialization (seed 43: top4=0.459,
   B-swap=+2.50 at branch epoch 500), while the matched-compute control shows
   no binding for any seed. Transfer is positive but unreliable.

## Acquisition phase

| Seed | Acquisition epoch | Final B-swap at checkpoint |
|------|------------------|---------------------------|
| 42   | 300              | +10.19                    |
| 43   | 500              | +7.57                     |
| 100  | 400              | +9.36                     |

All seeds reached the binding threshold (top4 ≥ 0.95, B-swap frac ≥ 0.95,
Q-swap frac ≥ 0.95, selectivity ≥ 0.8).

## Branch: continue_qfirst_ans_only (retention baseline)

All seeds maintain perfect binding throughout 500 additional epochs.
Final: NLL≈0, top4=1.000, B-swap≈13-18, selectivity=1.000.
The binding computation is unconditionally stable under continued
answer-only training.

## Branch: switch_qfirst_full (full next-token objective)

The full objective initially disrupts binding in all seeds, driving metrics
toward bag-level within the first 25 epochs. But ALL seeds recover:

| Seed | bep=1 top4 | Nadir top4 | Recovery epoch | bep=500 top4 | bep=500 B-swap |
|------|-----------|------------|----------------|-------------|---------------|
| 42   | 0.654     | 0.363 (e25)| ~75-100        | 0.998       | +15.10        |
| 43   | 0.295     | 0.232 (e200)| ~325-400      | 1.000       | +9.24         |
| 100  | 0.379     | 0.225 (e50)| ~100-275       | 1.000       | +12.96        |

**This is the most important finding.** In query first binding compact summary, query-first full from
scratch produced binding in only 1/3 seeds by 500 epochs. From a bound
checkpoint, full training produces RELIABLE binding recovery in 3/3 seeds.
The bound initialization changes the loss landscape from seed-bimodal to
reliably convergent.

This establishes a curriculum principle: acquire binding under a focused
objective, then switch to the full objective. The computation persists and
makes the full objective converge to binding rather than the bag-level
attractor.

## Branch: switch_qfirst_ctx_only (no answer pressure)

Binding collapses rapidly in all seeds:

| Seed | bep=1 top4 | bep=25 top4 | bep=500 top4 | bep=500 B-swap |
|------|-----------|------------|-------------|---------------|
| 42   | 0.605     | 0.217      | 0.275       | +0.29         |
| 43   | 0.293     | 0.242      | 0.246       | +0.06         |
| 100  | 0.344     | 0.252      | 0.260       | +0.04         |

The selector requires active answer-pressure reinforcement. Context-only
training develops context representations without maintaining the entity-
attribute routing that binding requires. NLL rises to 5-10+ (the model
eventually partially recovers context prediction but not binding).

## Branch: qfirst_block_qctx_ans (query-access dependency)

**Epoch-0 evaluation reveals the primary mechanism.** The bound checkpoint
evaluated with blocked query→context attention:

| Seed | Standard top4 | Blocked top4 | Standard B-swap | Blocked B-swap |
|------|-------------|-------------|----------------|---------------|
| 42   | 0.990       | 0.242       | +10.19         | +0.65         |
| 43   | 0.998       | 0.256       | +7.57          | −0.14         |
| 100  | 1.000       | 0.271       | +9.36          | +0.30         |

The binding computation depends on **prospective slot marking**: context
positions form query-specific representations (because they can attend to
the query), and the IS position reads these markers. When query→context
attention is blocked, the markers don't exist and binding collapses.

**Training with blocked mask.** During blocked training, standard-mask
evaluation shows partial binding (top4≈0.45-0.55) that slowly decays, while
blocked-mask evaluation stays near chance. But in 2/3 seeds, a late
re-acquisition occurs:

| Seed | Late transition | Final std top4 | Final blk top4 | Final std B-swap |
|------|----------------|---------------|----------------|-----------------|
| 42   | ~425-500       | 0.918         | 1.000          | +12.16          |
| 43   | none           | 0.451         | 0.271          | +0.63           |
| 100  | ~475-500       | 0.877         | 0.904          | +6.43           |

In seeds 42 and 100, the model discovers an alternative binding strategy that
works WITHOUT prospective marking. The blocked-mask evaluation reaches high
binding (blk_top4 ≈ 0.90-1.00), meaning IS-position-only content-addressable
lookup is achievable but takes much longer than the prospective-marking route.

**Interpretation:** Prospective marking is the efficient route to binding
(acquired by epoch 300-500 in the standard query-first format). IS-position-
only matching is an alternative that requires ~700-900 total epochs of blocked
training. The architecture can implement both strategies.

## Branch: transfer_orig_ans_only (bound → original order)

The bound checkpoint shows NO original-order binding at branch epoch 0
(orig top4 ≈ 0.24-0.25, B-swap ≈ 0.0). The prospective marking strategy
is format-specific and does not transfer.

During original-order training:

| Seed | Acq epoch | bep=100 orig top4 | bep=300 orig top4 | bep=500 orig top4 | bep=500 orig B-swap |
|------|----------|-------------------|-------------------|-------------------|---------------------|
| 42   | 300      | 0.250             | 0.229             | 0.238             | −0.002              |
| 43   | 500      | 0.328             | 0.361             | 0.459             | +2.498              |
| 100  | 400      | 0.236             | 0.256             | 0.262             | −0.014              |

Control (fresh init, original-order answer-only, matched total epochs):

| Seed | Total epochs | Final orig top4 | Final orig B-swap |
|------|-------------|----------------|-------------------|
| 42   | 800         | 0.250          | +0.004            |
| 43   | 1000        | 0.240          | +0.006            |
| 100  | 900         | 0.254          | −0.003            |

**Seed 43 shows clear progressive transfer** (top4: 0.25 → 0.46, B-swap: 0 →
+2.50 over 500 epochs; control stays at top4=0.24, B-swap≈0). Seeds 42 and 100
show no transfer. The transfer is positive but unreliable (1/3 seeds).

## Scientific interpretation

### The curriculum principle (strongest finding)

The most robust and publication-worthy result is the full-objective recovery:
**a computation that is difficult to acquire under the full next-token objective
can be first acquired under a focused objective, then reliably maintained and
strengthened when the full objective resumes.**

This means the bag-level convergence in original training (orbit binding design) is not an
inescapable attractor. It is avoidable through a curriculum that first routes
sufficient credit to the binding computation. The bound initialization changes
the landscape from one where the full objective finds bag-level (query first binding compact summary: 1-2/3
bag-level from scratch) to one where it reliably recovers binding (branching experiment: 3/3
recovery from bound start).

### Active reinforcement, not structural stability

The context-only failure shows that the binding is not a fixed property of the
model's parameters. It is a dynamic computation that requires ongoing
reinforcement through the answer objective. This means the selector is maintained
by the loss signal, not just encoded in the weights. When the loss signal
changes (context-only: no answer pressure), the computation decays.

This has implications for curriculum design: the transition from focused to
full objective must preserve enough answer pressure to maintain the selector.
Context-only phases (e.g., training on text structure without target prediction)
would destroy acquired computations.

### Prospective marking and format dependence

The blocked-mask evaluation definitively shows that query-first binding uses
prospective slot marking. The bound checkpoint has near-perfect binding under
standard evaluation but chance-level under blocked evaluation.

This explains why transfer to original order is unreliable: the prospective
marking strategy requires query access during context processing. In original
order, context states form before the query, so they cannot carry query-
specific markers. The 1/3 transfer seed (seed 43) may have partially learned
an alternative strategy, but this is not reliable.

### Relation to the general data-efficient learning principle

The orbit binding design-017 sequence establishes:
1. **The same finite data can produce either bag-level statistics or entity-
   specific binding** depending on format and objective routing (orbit binding design).
2. **A computation acquired under favorable conditions can make later full-
   sequence training converge to a higher-level solution** (branching experiment full-
   objective recovery).
3. **The computation requires active reinforcement, not just one-time
   formation** (branching experiment context-only failure).
4. **The primary mechanism is prospective marking, which is format-dependent**
   (branching experiment blocked-mask test).

Together: data-efficient learning requires that the training format and
objective route sufficient credit to the target computation. When the ordinary
format makes this difficult, a curriculum that first acquires the computation
in a favorable format and then transitions to ordinary training can make the
same experience produce a qualitatively different result. The acquired
computation is dynamic (requires reinforcement) and format-specific (depends
on causal access during context processing), but it changes the convergence
behavior of subsequent training.

## Files

- Script: `scripts/binding_branching.py`
- Data: `data/branching/results.json`
- Figure: `figures/branching.png`
- Design note: `notes/branching_design_and_interpretations.md`
- Integration note: `notes/relation_credit_integration_candidate.md`

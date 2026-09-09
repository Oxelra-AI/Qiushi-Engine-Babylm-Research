# orbit binding design Permutation-orbit entity-attribute binding experiment

## Why this experiment

rebinding and interpretation correction showed that source trigger experiment design's models lacked robust entity-attribute rebinding at any
training epoch. The late-epoch source-conditioned interactions observed in source trigger experiment design
emerged after held-out NLL had deteriorated, making them differences among
increasingly overconfident models rather than evidence of useful retained competence.
source trigger experiment design's comparators were also imprecise: UNPAIRED_SRC taught anti-identity (b≠a)
rather than independence, and IDENT_BLOCKED left multi-layer paths open.

Before asking whether identity practice helps or harms nonidentical-target prediction
(the BabyLM-bridge question), we need to establish whether the small Transformer
can learn reusable entity-attribute binding at all from finite training, and what
training signal is necessary.

## Design

### Data structure

K=4 entities per context, each assigned a unique attribute from N_ATTR=12. The
correct target is RWT(attribute of the queried entity). Different permutations of the
same entity-attribute sets produce different correct targets, forcing the model to
read the context rather than memorize entity-attribute associations.

Sequence: `BOS E0 HAS A(σ0) E1 HAS A(σ1) E2 HAS A(σ2) E3 HAS A(σ3) SEP Eq IS TGT PAD`
SL=18, IS_POS=15, VOCAB=39. Same CLM architecture as source trigger experiment design (153K params).

### Arms

All arms share entity/attribute distributions and evaluation probes:

- **BOUND**: target = RWT(query entity's current attribute). Forces entity-specific reading.
- **BAG_INDEPENDENT**: target = RWT(random context attribute). Provides bag-level signal
  (1/4 match rate with correct answer). Tests whether bag-level training produces
  entity-specific binding.
- **MARGINAL**: target = RWT(random attribute from full set of 12). Provides no
  context-specific signal. Tests whether the sequence structure alone helps.

### Evaluation metrics (trajectory-first, recorded every 10 epochs)

1. **Standard probes** (200 fresh random contexts with training entities):
   - correct_nll: −log p(correct RWT)
   - family_mass: sum p(all RWT tokens)
   - within_nll: −log(p(correct)/p(RWT family))
   - mrr: reciprocal rank of correct within RWT family

2. **Held-entity probes** (100 probes where query entity is from held set [8,9]):
   - Same metrics as standard. Tests entity generalization.

3. **B_swap** (200 paired probes, same token bag, different binding):
   For each pair (σ₁, σ₂) with the same entity/attribute tokens but different
   query-entity assignments:
   B_swap = 0.5 × [margin(correct₁ over correct₂ | σ₁) + margin(correct₂ over correct₁ | σ₂)]
   Positive B_swap = model follows the context binding. Zero = bag-level only.

4. **Query-swap** (100 probes, same context, different query entity):
   Tests whether changing the query entity changes the model's top prediction.
   both_correct = fraction where both query-entity predictions are correct.

5. **Event corruption** (100 triplets: original + query-corrupt + decoy-corrupt):
   - Replace query entity's attribute with a novel attribute not in the context.
   - Replace a random nonquery entity's attribute with the same novel attribute.
   - query_drop: decrease in p(correct) after query corruption (should be large for binding)
   - decoy_drop: decrease in p(correct) after decoy corruption (should be small)
   - query_novel: increase in p(novel RWT) after query corruption (model follows new binding?)
   - decoy_novel: increase in p(novel RWT) after decoy corruption (bag-level effect)
   - Selectivity = query_drop / decoy_drop (entity-specific vs bag-level reading)

### Expected outcomes by arm

| Metric | BOUND | BAG_INDEP | MARGINAL |
|--------|-------|-----------|----------|
| correct_nll | Low (→0 if perfect) | ~log(4)≈1.39 | ~log(12)≈2.49 |
| B_swap | Large positive | Near 0 | Near 0 |
| query_swap both | High (→1) | ~0.25 | ~0.25 |
| query_drop | Large | ~0.25 | ~0.08 |
| decoy_drop | Small | ~0 | ~0 |
| novel follow (query) | Large | ~0.25 | ~0.08 |

### Key scientific questions

1. **Can the 3-layer Transformer learn entity-specific binding?** If BOUND achieves
   strong B_swap and query-swap, yes. This establishes a positive baseline that
   source trigger experiment design's substrate lacked.

2. **When does binding appear?** The trajectory shows acquisition timing. If it
   appears early and persists, the architecture supports durable binding. If it
   appears and then deteriorates (as in source trigger experiment design), overfitting destroys binding.

3. **Does bag-level training produce entity-specific reading?** If BAG_INDEP's B_swap
   stays near zero while its MRR approaches bag-uniform level, then entity-specific
   training signal is necessary—context structure alone doesn't suffice.

4. **Does binding generalize to held entities?** If BOUND's held-entity probes are
   strong, the model learned a general entity-matching computation, not entity-specific
   memorization.

5. **How selective is the corruption response?** High selectivity (query >> decoy)
   means the model reads the specific queried event, not just the bag. Low selectivity
   means bag-level reading regardless of the query.

### What each outcome means for the research

- **BOUND succeeds, BAG_INDEP fails**: entity-specific training is necessary. The
  question then becomes whether identity/repetition practice provides this signal
  in natural language, and what architectural features support it.

- **BOUND fails**: the 3-layer Transformer cannot learn reusable binding from this
  amount of data. Either more capacity, better architecture (slot models), or
  different training pressure is needed. This would explain source trigger experiment design's failure at a
  fundamental level.

- **Both succeed**: entity-specific reading emerges from bag-level training. This
  would mean the sequence structure (entity tokens, HAS, attribute tokens) provides
  enough implicit binding signal. The principle would shift toward structural
  affordance rather than training-target alignment.

## Training parameters

- Seeds: [42, 43, 100]
- Epochs: 200
- Sequences per epoch: 500 (fresh random each epoch)
- Architecture: CLM(V=39, d=64, nh=2, nl=3)
- Optimizer: AdamW(lr=3e-4, wd=0.01)
- Batch size: 64

## Files

- Script: `scripts/orbit_binding.py`
- Data: `data/orbit_binding/results.json`
- Figure: `figures/orbit_binding.png`

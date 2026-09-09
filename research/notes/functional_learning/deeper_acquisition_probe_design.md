# deeper acquisition probe design Deeper Acquisition Probe: Design and Interpretation Guide

## What repaired relation first acquisition synthesis established (bounded)

1. **Parent baseline**: coherent86 on 120 repaired relation-first pairs has near-zero
   recipient dependence (beta≈0, joint 0/120). The model strongly follows the most
   recently appended phrase but does not condition this on which entity was updated
   when the replacement phrase is fixed across recipient variants.

2. **Answer-only acquisition**: 80 epochs of private-adapter-only training with
   uncorrupted support reached held 28/30 four-condition, 58/60 orientations,
   min4=+2.036. This establishes learnability.

3. **Corruption obstacle**: broad 15% corruption of support tokens dropped held to
   2/30; background labels added a small additional deficit.
   Critical-token protection recovered to 11/30.

## What deeper acquisition probe design tests

### Probe 1: Three-way discrimination (highest priority)

The current four-condition metric tests answer vs. foil where foil is always
the shared new value (in RETAIN) or the entity's source value (in UPDATE). It
never compares one entity's source value against the other entity's source value.

A model could pass all four conditions by learning:
- If query == update-recipient → prefer shared_new over this_source
- If query ≠ update-recipient → prefer this_source over shared_new

This is an **update-recipient/query equality gate** that suppresses the replacement
when the query doesn't match the updated entity, but it could assign probability
to BOTH source values equally. It doesn't prove entity-conditioned retrieval.

The three-way probe scores all three candidates (value_a, value_b, shared_new)
in each context:

- **Entity-conditioned retrieval** would show:
  - RETAIN query_a: P(value_a) >> P(value_b) >> P(shared_new)
  - RETAIN query_b: P(value_b) >> P(value_a) >> P(shared_new)
  - Cross-source margin > 0 consistently

- **Mere equality gating** would show:
  - RETAIN query_a: P(value_a) ≈ P(value_b), both >> P(shared_new)
  - RETAIN query_b: P(value_a) ≈ P(value_b), both >> P(shared_new)
  - Cross-source margin ≈ 0

- **Hybrid possibilities**: the model could have partial entity-level discrimination
  but not complete retrieval. Cross-source margin could be positive but small
  compared to source-over-new margin.

The key measure is **cross-source margin**: logP(correct_source) - logP(wrong_source)
in RETAIN contexts. This is distinct from the four-condition metric.

### Probe 2: Source-reassignment test

If we swap which entity has which source value in the context text—keeping entities,
update sentences, and query frames fixed—does the model's RETAIN prediction follow
the reassignment?

- Original: "A died in Lexington. B died in Turin."
  RETAIN query A → should prefer Lexington
- Swapped: "A died in Turin. B died in Lexington."
  RETAIN query A → should prefer Turin (if the model reads entity-value bindings)

This connects to the earlier bag-membership failures (orbit binding design): the permutation-
orbit task showed that ordinary training converges to bag-level behavior. Does the
80-epoch answer-only training install actual binding-following, or does it use surface
position/frequency heuristics?

**Possible outcomes**:
- High swap-following: the model reads entity-value bindings from context
- Low swap-following: the model uses positional/frequency heuristics
- Partial: some relations/pairs follow, others don't

### Probe 3: Random protection control

Step040e protected critical evidence tokens (entities, source values, replacement)
from corruption and recovered from 2/30 to 11/30. But this also reduced total
corruption from ~305K to ~241K positions.

The random control protects the same number of word groups but chosen randomly.

- **Critical >> Random**: the specific evidence tokens matter for acquisition
- **Critical ≈ Random**: generic noise reduction explains the recovery
- **Random > Critical**: unlikely, but would indicate critical protection has
  a confound (e.g., protecting common words that act as position markers)

## Interpretation framework

If the three-way probe shows strong entity-conditioned retrieval AND source-
reassignment following on held pairs, this establishes that:

1. The answer-only training installs a genuine entity-state-tracking computation
   in the private adapters, not just an update-recipient gate
2. The computation reads entity-value bindings from context text
3. The data-efficiency limitation in coherent86 is specifically about lacking
   concentrated credit on relation-relevant answer positions

If entity retrieval is weak or absent despite four-condition success, the 28/30
result overestimates the acquired computation. The model may only have learned
to suppress the replacement phrase when it detects that the query entity wasn't
mentioned in the update sentence.

## Connection to highest goal

The data-efficient learning principle candidate emerging from this work is:

**Evidence-preservation and concentrated credit**: under limited data, the model
defaults to a generic phrase-recency operation. Entity-conditioned state selection
requires (a) sparse relational evidence tokens to remain available in the input
and (b) credit to be concentrated on output positions where entity selection
determines correctness. When either condition fails—corruption removes evidence,
or background objectives dilute credit—the model satisfies local likelihood
through shared preference rather than acquiring the intended relational computation.

This is more specific than "background gradients interfere" and more actionable
than "more data helps." It predicts that:
- Selective masking/protection of relational evidence should help
- Answer-position credit concentration should help
- Generic regularization or noise reduction may be insufficient
- The benefit is not from reduced corruption per se, but from preserving
  the specific information needed for conditional selection

The deeper acquisition probe design probes test the first two predictions. The random protection
control tests the third.

## Files and tasks

- Three-way probe: `scripts/threeway_and_reassignment_probe.py`
  In progress at the time of this note
- Random protection: `scripts/random_protection_control.py`
  In progress at the time of this note
- This note: `notes/deeper_acquisition_probe_design.md`

# Coordinate transport and Entity-depth comparison

## Mechanistic synthesis across two studies

### Coordinate-transport mechanism

In the controlled relational harness:

1. **Equality pretraining** creates a shared coordinate for string identity.
2. **Sparse h0/h2 state training** orients the coordinate with absolute labels.
3. **Correct comparison constraints** transport the orientation to h1/h3 through
   the graph (topology phase1 result full_graph: all four relations at 1.0/0.0 with bridge_sign +1/-1).
4. **Wrong constraints install wrong orientation** (topology phase1 result adversarial: h1 flips to
   0.0/1.0 while h3 remains 1.0/0.0 under inverted h1-incident labels).
5. **(Pending mechanism macro convergence)** **Varied context forces abstraction** of the constraint,
   making transport robust to novel renderings.

### Entity-depth finding

In BabyLM Strict-Small DeBERTa:

1. **VIEW (varied rewrites) is a worse LM** (+0.03-0.10 nats vs CLEAN).
2. **VIEW reproducibly helps Entity tracking** (+3.21 cross-seed mean, SNR 11.9).
3. **VIEW reproducibly helps Reading** (+0.37, SNR 3.1).
4. **VIEW reproducibly hurts COMPS** (-0.35, SNR 1.8).
5. **Entity depth is a two-sided dissociation**:

| Entity ops | V-R (VIEW vs REPEAT) | Interpretation |
|---:|---:|---|
| 0 | -9.42 | REPEAT >> VIEW: direct fact retrieval needs memorization |
| 1 | +2.16 | VIEW > REPEAT: some state tracking needed |
| 2 | +3.19 | VIEW > REPEAT: state tracking through 2 operations |
| 3 | +7.54 | VIEW >> REPEAT: deep state tracking needs abstraction |
| 4 | +8.74 | VIEW >> REPEAT: deepest operation chain |
| 5 | +7.41 | VIEW >> REPEAT (smaller sample) |

## The mapping

| Coordinate-transport concept | Entity-depth analogue |
|---|---|
| Direct h0/h2 anchor state recall | Entity 0-operation retrieval |
| h1/h3 graph transport from anchors | Entity 3-4 operation state tracking |
| Correct comparison labels | Varied rewrites preserving semantic content |
| Wrong comparison labels (adversarial) | VIEW destroying comparative structure (COMPS) |
| Nonce verb abstraction across contexts | Content-invariant extraction across varied phrasing |
| REPEAT condition (same templates) | REPEAT arm (exact token repetition) |
| VARY_CONTEXT condition (different names/objects) | VIEW arm (semantic rewrites) |

## The proposed principle

**Finite experience is efficient when it supplies correct constraints on a shared
coordinate, where the constraints are abstracted across varied contexts.**

Specifically:
1. **The value of experience is competence-specific, not universal.**
   - Direct retrieval benefits from repetition (reinforces specific patterns).
   - State-tracking competences benefit from variation (forces abstraction).
   - The benefit is proportional to the depth of the state-tracking chain.

2. **The correct content of the constraint matters, not mere exposure.**
   - Wrong constraints can actively install wrong competence (adversarial result).
   - This explains why VIEW hurts COMPS: if rewrites degrade comparative structures,
     the wrong content is installed for compositional comparison.

3. **Variation must break a task-relevant shortcut to be efficient.**
   - Generic surface diversity (noise) does not force abstraction.
   - Context variation that changes the surface pattern while preserving the relation
     forces the learner to extract the abstract structure.
   - This predicts that VARY_CONTEXT > VARY_NOISE in the synthetic harness.

## What mechanism macro convergence results will establish

If VARY_CONTEXT > VARY_NOISE ≈ REPEAT on held-template h1/h3 transport:
→ The mechanism is confirmed: shortcut-breaking variation is what helps, not diversity per se.
→ This explains the BabyLM Entity depth dissociation: VIEW helps at depth because state
  tracking requires abstracted relational evidence.

If VARY_CONTEXT ≈ VARY_NOISE > REPEAT:
→ Generic diversity helps (regularization effect), not specific shortcut-breaking.
→ The mechanism is weaker but still predicts the BabyLM depth dependence.

If all three saturated at 60 epochs:
→ The harness is too simple to distinguish the mechanisms at this budget.
→ Need to reduce epochs or increase task difficulty.
→ But the BabyLM Entity depth dissociation still provides the macro evidence.

## Next priorities (regardless of mechanism macro convergence outcome)

1. If mechanism macro convergence confirms: quantify the learning dynamics (when does VARY_CONTEXT
   diverge from REPEAT?) and connect to budget efficiency.
2. If mechanism macro convergence saturated: run at reduced budgets (20, 30 epochs) or with reduced
   comparison counts to find the informative regime.
3. In either case: the adversarial result + Entity depth dissociation already
   provide a two-level mechanism. The principle should be formalized precisely
   enough to make quantitative predictions about data allocation.

## Open questions for the principle

1. How does the principle scale to larger models and datasets?
2. Does the principle explain the DeBERTa/RoBERTa difference through p2c/c2p?
3. Can the principle predict which benchmarks benefit from VIEW vs REPEAT?
4. Is there a formal relationship between "operation depth" and "graph distance"?
5. How does this connect to existing theories of compositionality and variable binding?

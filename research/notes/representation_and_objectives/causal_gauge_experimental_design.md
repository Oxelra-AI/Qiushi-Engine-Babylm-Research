# causal gauge experimental design — Causal gauge transport experimental design and evidence chain

## Evidence chain so far

### Layer 1: shared factorization result synthesis shared-coordinate positive
- Tied model: graph-transfer h1/h3 same-initial state, pair-both, mixed orientation all at 1.0
- Untied model: local fit 1.0 but graph-transfer at 0.25-0.5
- Train-only vocabulary replication preserved the effect

### Layer 2: shared factorization result synthesis heldheld-only anchor control (now verified)
- Tied without anchors: perfect hh_closure but wrong absolute orientation (direct_same=0.0, mixed_margin=-11.5)
- The shared coordinate enforces coherence but cannot determine absolute sign without anchors
- Untied without anchors: chance-level state (0.5), no coherent orientation

### Layer 3: shared gauge review and next causal controls CPU saved-output verification
- Row-paired sign analysis: tied aligned/inverted margins reverse on all 64 graph-transfer rows
- Event coordinates: held coordinates reverse between arms, seen coordinates stable
- Untied comparison coordinates identical (no state→comparison gradient path)

### Layer 4: causal gauge experimental design bridge_sign × model × factorial (running)

## causal gauge experimental design experimental design

### Primary experiment: bridge_sign × model
Fixed aligned arm, train-only vocabulary, dropout=0, paired initialization.

**bridge_sign parameter**: applied only to h0/h2 changed-state training loss.
```
if bridge_sign == -1 and q.is_direct_anchor:
    scores = scores.flip(0)  # reverse candidate ordering before CE loss
```
No change to comparison rows, unchanged/static rows, h1/h3, or eval.

**Model variants** (all from cloned initial weights at seed 29000):
- **tied**: one scorer for both comparison and state (current positive)
- **shared_trunk**: shared embedding+GRU, separate scalar heads
- **untied**: fully independent comparison and state scorers

### Factorial cells: data filters
- **Full** (current positive): all comparisons + bridge state anchors
- **Bridge-only** (--no-comparisons): bridge state anchors + common seen, NO held-held comparisons
- **Comparison-only** (--no-bridge-anchors): held-held comparisons + common seen, NO bridge state

## Predicted fingerprints

### Gauge transport (shared-coordinate hypothesis):

| Cell | Model | bridge_sign=+1 → -1 effect on h1/h3 canonical margin |
|---|---|---|
| Full | tied | h1/h3 margins reverse (all graph events flip with anchors) |
| Full | shared_trunk | NO coherent h1/h3 reversal (separate scalar heads break gauge) |
| Full | untied | NO h1/h3 response (no gradient path from anchors) |
| Bridge-only | tied | h0/h2 flip but h1/h3 DO NOT flip (no comparison graph to transport) |
| Bridge-only | untied | no effect (independent scorers) |
| Comparison-only | tied | no bridge_sign effect (no anchors to flip) |

### Alternative: generic sharing
If shared_trunk shows the same coherent reversal as tied, the effect is generic representation sharing rather than scalar gauge identity.

### Alternative: surface generalization
If bridge-only tied shows h1/h3 reversal, the model generalizes a surface rule from h0/h2 directly to h1/h3 without graph evidence. This would weaken the graph-transport claim.

## Key readouts
1. Canonical h1/h3 margins (unpermuted, from state scorer)
2. Row-paired d_e sign reversal fractions
3. Mixed held-seen comparison product signs
4. Held-held comparison product invariance
5. Seen-reference coordinate stability
6. Train fit (post-connector for anchors, canonical for everything else)

## What the result decides
- If tied-full shows reversal AND bridge-only does NOT → graph-mediated gauge transport
- If shared_trunk also shows reversal → generic sharing, not scalar gauge
- If bridge-only also shows reversal → surface generalization confound
- If comparison-only shows effect → implementation bug (anchors should be absent)
- If nothing shows reversal → the bridge_sign intervention may be absorbed differently than predicted

## Next after this experiment
1. If gauge transport confirmed: frozen-affine scalar test (d_state = a*d_cmp + b)
2. If gauge transport confirmed: proper graph-cut substrate (disconnect anchor component from h1/h3)
3. If shared_trunk matches tied: move to higher-dimensional shared representations
4. Eventually: parser-free unified updater

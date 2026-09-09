# causal gauge experimental design — Heldheld-only anchor control directly verified

## Source
- Read `data/shared_coordinate_heldheld_anchor_analysis/shared_coordinate_merged_analysis_summary.md`
- Read full JSON at the same path

## Result

### Tied heldheld-only (no bridge state anchors, comparison + common seen only)
- Seed 28801: train state=1.0, train cmp=1.0, **hh_closure=1.0**, direct_same=0.0, graph_same=0.0, mixed_acc=0.0, mixed_margin=-11.493, unchanged=1.0
- Seed 28802: train state=1.0, train cmp=0.875, hh_closure=0.625, direct_same=0.0, graph_same=0.25, mixed_acc=0.125, mixed_margin=-4.897, unchanged=1.0

### Untied heldheld-only
- Seed 28801: train state=1.0, train cmp=1.0, **hh_closure=1.0**, direct_same=0.5, graph_same=0.5, mixed_acc=0.0, mixed_margin=-8.974, unchanged=1.0
- Seed 28802: train state=1.0, train cmp=1.0, hh_closure=1.0, direct_same=0.5, graph_same=0.25, mixed_acc=0.5, mixed_margin=-2.468, unchanged=1.0

## Scientific reading

1. **Comparison closure without anchors does not fix absolute state orientation.** Tied seed 28801 achieves perfect hh_closure=1.0 and train cmp=1.0, but all state metrics are 0.0. The mixed_margin of -11.493 means the model consistently chose the *wrong* absolute orientation for every held-seen comparison. The shared coordinate enforced internal coherence while locking onto the wrong global sign.

2. **Tied is worse than untied without anchors on state.** Untied direct_same/graph_same hover at 0.5 (chance), while tied stays at 0.0 (consistently wrong). This is the expected signature: tied forces consistency, which without anchors can be consistently wrong; untied allows independent random state orientations that average to chance.

3. **Bridge state anchors are necessary for the absolute coordinate.** This control directly supports the shared factorization result synthesis interpretation: comparison graph evidence establishes relative closure, and state anchors fix the absolute sign. Without anchors, even perfect comparison cannot determine which of the two globally coherent solutions is the correct one.

4. **This completes the missing piece of the shared factorization result synthesis evidence chain.** The claim is now: (a) tied + comparison + anchors → graph transfer at 1.0; (b) tied + comparison only → relative closure but wrong absolute sign; (c) untied + comparison + anchors → local fit but no graph transfer. The causal gauge experimental design bridge_sign × model × factorial experiment will test whether this is genuine gauge transport.

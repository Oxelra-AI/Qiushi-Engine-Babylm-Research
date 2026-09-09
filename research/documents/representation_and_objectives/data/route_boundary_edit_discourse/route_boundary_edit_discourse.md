# earlier analysis route boundary from edit/discourse evidence

## What evidence says
- Edit main private true-vs-decoy held-out NLL advantage: `None`; pilot-v2 small run advantage: `None`.
- Edit main raw true-vs-decoy advantage: `2.741850732602998`; pilot-v2 raw advantage: `1.2432499827555148`.
- Edit targets in main CSV are wordlike length>=4 or capitalized only `0.3906` of items; many high-advantage targets are subword/morphological pieces.
- Discourse raw true-vs-reversed `0.018266387062130462` versus true-vs-shuffled `0.49339718942439603`; private true-vs-reversed `None`, private true-vs-shuffled `None`.

## Interpretation for ACS closed single-context hard-negative sharpening; edit/discourse evidence is useful but not yet a training route for companion analysis. The edit object contains real legal source-conditioned changed-token value, but current evidence does not yet show an explicit dense equivariant transformation in which a context change induces a known permutation or direction among the same plausible alternatives. The discourse object is abundant but its order direction is nearly absent: true and reversed neighbors behave almost the same while shuffled neighbors lose value, so this is mostly topic/entity/register state.

## Next useful work
Do not duplicate edit-state or intra-row discriminator work and do not launch GPU training from these artifacts. A valuable next construction should either (a) build a small transformation-inventory test that filters source→rewrite pairs for source-absent, full-word, semantic/operator edits and tests whether such operators are dense enough to matter, or (b) invent a different representation-forming mechanism not based on correlated paired MLM/discriminator examples. Any later training screen must have an explicit legal false-structure control and fixed natural GlobalPIQA hard52/EWoK readouts.

JSON: `experiments/archive/representation_and_objectives/data/route_boundary_edit_discourse/route_boundary_edit_discourse.json`

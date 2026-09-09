# edit route closed: Edit equivariant route — CLOSED

## Inventory results
- 12,155 compact-view pairs analyzed
- 3,507 total 1:1 word substitutions
- **1,157 SAFW** (source-absent, both full-word, both content) across 1,017 unique word pairs
- **Only 13 bidirectional pairs**, 10 of which are morphological variants (become↔becomes, cause↔causes, etc.)
- **40.6% stem leakage** — nearly half of SAFW are morphological, not semantic
- 8.9% of pairs contribute any SAFW substitution

## Why it fails the equivariant binding criterion
Simplification rewrites are inherently **unidirectional**: complex → simple.
Top pairs: allows→lets, difficult→hard, requires→needs — always same direction.
The corpus contains essentially **zero semantic bidirectional word pairs**.
No equivariant structure exists to train context-conditioned alternative binding.

## Threshold failures
- SAFW 1,157 < 2,000 (viable threshold)
- Bidirectional 13 << 50 (viable threshold), and 10/13 are morphological
- Stem leakage 0.406 > 0.30 (viable threshold)
- Contributing fraction 0.089 < 0.10 (viable threshold)

## What remains from the edit/simplification evidence
- The compact-view PAIRING itself is validated as a broad-learning improvement (replicated +1.49 mean7)
- sparse relation aux preflight-123 show that holistic source→rewrite TRANSFORMATION information
  (not per-word substitutions) carries learning signal that transfers via dual-view training
- This is a different mechanism: dual-view shared pathway, not per-word equivariant ops

## Route decision
**Edit per-word equivariant route: DEFINITIVELY CLOSED.**
- Do not search for more substitution pairs or refine the alignment
- Do not try to manufacture bidirectional pairs from the unidirectional edit stream
- The research boundary for companion analysis on this route is the inventory evidence itself

## Remaining Scientific Comparisons
The dual-view shared/private pathway remains a separate comparison. The immediate endpoint question uses the existing scale1.75 100-checkpoint ladder: the current best is 80M at 41.7716, approximately 0.028 below 41.80.

Files: `data/edit_operator_inventory/`

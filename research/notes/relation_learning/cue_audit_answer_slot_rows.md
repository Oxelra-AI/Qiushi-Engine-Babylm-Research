# cue audit answer slot rows: Cue Audit of Answer-Slot Rows

## Summary

The earlier analysis answer-slot construction has systematic local-copy cues that make 
base accuracy (~81%) trivially achievable without entity-conditioned retrieval.

## Key numbers (382 held-out rows: 191 DISTRACTOR, 191 UPDATED)

| Statistic | Count | Fraction |
|---|---|---|
| Single-token answers | 304 | 79.6% |
| Foil longer than answer | 333 | 87.2% |
| Answer verbatim in source (DISTRACTOR) | 180/191 | 94.2% |
| Answer verbatim in update (UPDATED) | 180/191 | 94.2% |
| Context favors source | 188 | 49.2% |
| DISTRACTOR: only source hits in use | 191/191 | 100% |
| UPDATED: only new hits in use | 191/191 | 100% |

## Consequence

1. Base ~81% accuracy is a local-copy ceiling, not entity-binding evidence
2. Answer-only training on these rows would practice *which sentence to copy the local n-gram from*, not entity-conditioned retrieval
3. The foil/answer token-length asymmetry violates the symmetric one-token contract
4. Cannot be paired against the uniform-WWM negative as a factorial

## Fix required

Deterministic recombination: from DISTRACTOR packets (two-entity sources), create minimal 
pairs where the same source+update context yields opposite answers depending on which 
entity is queried. Only entity identity flips the answer; local cue is identical.

The earlier analysis T/U/N instrument (template mode) already implements this: 
`"The relevant state of {entity} is "` + candidate phrase, with entity name as the 
differentiating cue. The trusted-loader rescore on the 400 held-out packets reopens 
the base characterization and becomes the before/after readout.

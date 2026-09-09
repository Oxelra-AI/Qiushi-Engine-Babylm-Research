# earlier analysis context-diversity × anchor-recurrence preflight

JSON: `experiments/archive/representation_and_objectives/data/context_anchor_factorial_preflight/context_anchor_factorial_preflight.json`

## Derangement
- pairs: 12155
- self-pairs: 0
- exact view-word length matches: 12152/12155 (0.999753)
- total abs view-word delta: 4; net delta 0

## Mean receiver-source content overlap
- own compact: 0.658277; wrong compact: 0.003124
- own repeat: 0.634234; wrong repeat: 0.002906

## Mean BPE deltas per pair
- HS-LS: 2.007240
- HD-LD: 2.007240
- HS-HD: 0.000000
- LS-LD: 0.000000

## Scientific reading
This supports a four-arm ordinary-WWM factorial if the next constructor can materialize exact 10M/100M streams with identical filler, row packing, tokenizer, initialization, optimizer and precomputed WWM manifests. The different-anchor arms preserve context-type marginals but introduce source/second-view mismatch, so only downstream interaction contrasts are interpretable.

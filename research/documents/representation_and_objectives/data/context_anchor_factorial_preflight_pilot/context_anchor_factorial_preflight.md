# earlier analysis context-diversity × anchor-recurrence preflight

JSON: `experiments/archive/representation_and_objectives/data/context_anchor_factorial_preflight_pilot/context_anchor_factorial_preflight.json`

## Derangement
- pairs: 500
- self-pairs: 0
- exact view-word length matches: 495/500 (0.990000)
- total abs view-word delta: 64; net delta 0

## Mean receiver-source content overlap
- own compact: 0.661886; wrong compact: 0.002449
- own repeat: 0.642174; wrong repeat: 0.002530

## Mean BPE deltas per pair
- HS-LS: 1.902000
- HD-LD: 1.902000
- HS-HD: 0.000000
- LS-LD: 0.000000

## Scientific reading
This supports a four-arm ordinary-WWM factorial if the next constructor can materialize exact 10M/100M streams with identical filler, row packing, tokenizer, initialization, optimizer and precomputed WWM manifests. The different-anchor arms preserve context-type marginals but introduce source/second-view mismatch, so only downstream interaction contrasts are interpretable.

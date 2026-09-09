# family specific tokenizer predictor — Sequence curriculum accounting measurement

Neutral summary of the CPU-only sequence-curriculum accounting measurement on the exact compact_view_reinvest 10M pool. No model was trained or evaluated.

- Pool: 64740 rows / 10000000 words.
- Public leader factor: 64→256 sequence length with inverse batch scaling.
- Existing local schedule behavior: tokenizes rows at 256, slices short prefixes, and still counts full row words.

## legal40k
- vocab 40000; tokens/word 1.3943.
- Prefix-visible group fractions by length: L64 0.306, L128 0.608, L256 0.987.
- For 64×3/128×4/256×3, prefix slicing misses 0.369 of word-groups; faithful chunking provides 1.609× target tokens at 1.081× optimizer steps.
- For 64×7/256×3, prefix slicing misses 0.489 of word-groups; faithful chunking provides 1.988× target tokens at 1.037× optimizer steps.

## minfreq25
- vocab 29529; tokens/word 1.4139.
- Prefix-visible group fractions by length: L64 0.302, L128 0.600, L256 0.986.
- For 64×3/128×4/256×3, prefix slicing misses 0.374 of word-groups; faithful chunking provides 1.622× target tokens at 1.092× optimizer steps.
- For 64×7/256×3, prefix slicing misses 0.493 of word-groups; faithful chunking provides 2.002× target tokens at 1.050× optimizer steps.

## Route implication

A future 64->256 route should use stage-length word-boundary chunking or streaming with inverse row-batch scaling. The existing short-prefix slicing path should not be treated as the leader-style sequence factor because it hides large fractions of word-groups while counting their words.

JSON: `experiments/archive/representation_and_objectives/data/sequence_curriculum_accounting_measurement/sequence_curriculum_accounting_measurement.json`

# earlier analysis deterministic token-matched entity-orbit corpus

Rows/words: 400 / 55824

Candidate/changed rows: 380 / 380

Candidate/assigned keys; replaced spans: 1758 / 1758; 3605

Geometry, word-count, injectivity failures: 0 / 0 / 0

100M rows/words/order misses: 0 / 0 / 0

The selector is deterministic and uses no learned external model. Every accepted row preserves the complete tokenizer word-start vector, so the historical WWM trainer samples the same mask groups and token positions under the same RNG.

Manifest: `experiments/archive/representation_and_objectives/data/deterministic_token_matched_entity_orbit/det_tokenmatched_orbit_manifest_pilot_400.json`

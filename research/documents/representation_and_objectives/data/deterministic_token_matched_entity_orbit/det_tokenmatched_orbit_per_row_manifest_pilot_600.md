# earlier analysis deterministic token-matched entity-orbit corpus

Rows/words: 600 / 84090

Candidate/changed rows: 268 / 268

Candidate/assigned keys; replaced spans: 387 / 387; 792

Geometry, word-count, injectivity failures: 0 / 0 / 0

100M rows/words/order misses: 0 / 0 / 0

The selector is deterministic and uses no learned external model. Every accepted row preserves the complete tokenizer word-start vector, so the historical WWM trainer samples the same mask groups and token positions under the same RNG.

Manifest: `experiments/archive/representation_and_objectives/data/deterministic_token_matched_entity_orbit/det_tokenmatched_orbit_per_row_manifest_pilot_600.json`

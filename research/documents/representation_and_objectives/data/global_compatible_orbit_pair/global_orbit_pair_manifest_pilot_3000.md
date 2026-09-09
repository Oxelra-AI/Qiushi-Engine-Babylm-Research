# orbit pair audit and slot reuse update global-compatible orbit pair

Input rows/words: 3000 / 422814

Candidate rows/key-events/keys: 1372 / 1993 / 1503

Global aliases selected: 1501 keys; coverage classes {'full': 1498, 'partial': 3, 'zero': 2}

Final changed rows/key-events/spans: 1369 / 1986 / 4090

Rows dropped after row-level paired geometry/injectivity checks: 1

Stable keys with multiple aliases: 0 (must be 0)

Per-row keys with multiple aliases: 285 / 1499

100M rows/words: 0 / 0

Support SHA256: `07a7f8bf26c648c22823e8b4eb47ccf90f55d5d4abea91b5cc5203320458124f`

The two arms transform exactly the same row-key occurrences. `stable_global` uses one corpus-global alias per lexical key; `per_row_global_support` uses row-specific aliases on the same support. Every transformed row preserves whitespace word count and the complete legal16k WWM word-start geometry.

Manifest: `experiments/archive/representation_and_objectives/data/global_compatible_orbit_pair/global_orbit_pair_manifest_pilot_3000.json`

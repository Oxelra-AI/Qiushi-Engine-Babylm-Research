# orbit pair audit and slot reuse update global-compatible orbit pair

Input rows/words: 64740 / 10000000

Candidate rows/key-events/keys: 17433 / 24970 / 9723

Global aliases selected: 9682 keys; coverage classes {'full': 9667, 'zero': 41, 'partial': 15}

Final changed rows/key-events/spans: 16909 / 24211 / 56112

Rows dropped after row-level paired geometry/injectivity checks: 2

Stable keys with multiple aliases: 0 (must be 0)

Per-row keys with multiple aliases: 3167 / 9674

100M rows/words: 647400 / 100000000

Support SHA256: `a31063783b1f726d980fe53b1db4e82866191c3ec3e5fcb497446f769cba3e2d`

The two arms transform exactly the same row-key occurrences. `stable_global` uses one corpus-global alias per lexical key; `per_row_global_support` uses row-specific aliases on the same support. Every transformed row preserves whitespace word count and the complete legal16k WWM word-start geometry.

Manifest: `experiments/archive/representation_and_objectives/data/global_compatible_orbit_pair/global_orbit_pair_manifest.json`

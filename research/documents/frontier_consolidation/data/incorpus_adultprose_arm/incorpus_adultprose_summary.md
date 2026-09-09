# earlier analysis in-corpus adult-prose arm

No FineWeb content. Replaces developmental/speech rows with unused official
Gutenberg/SimpleWiki text at rho≈0.0443, matching full_1x geometry.

## Key numbers
- Pool: 65313 rows, 10000000 words
- Stream: 100000000 words (10 epochs)
- Changed rows: 3005 (4.43% of pool words)
- Changed source: {'gutenberg': 272048, 'simple_wiki': 170939}
- Position overlap with full_1x: 1902/3006
- Row-index mean: incorpus 13291.2, full_1x 1504.1
- Qwen preserved: 12236/12236
- Pool SHA256: e135a8798dafe262d7d57f2ee358e7b14825c6988e8dabb7193699b04f2b35c6
- Stream SHA256: 95090e559d451dae26767ac65f8efccc64104d12e33c6031788f63369ff2e3a9

## Scientific purpose
Tests whether broad ex-Entity competence gain requires out-of-corpus FineWeb
content, or whether in-corpus adult prose (Gutenberg/SimpleWiki) suffices.

If incorpus ≈ full_1x (view): target-proximity mechanism.
If incorpus ≈ clean: distinct-content/novelty mechanism.

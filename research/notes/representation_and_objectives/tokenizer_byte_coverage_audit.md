# Tokenizer byte coverage audit

tokenizer: `experiments/archive/representation_and_objectives/data/strictsmall_tokenizer_retrain/strictsmall_compact_reinvest_16k_tokenizer/tokenizer.json` SHA `4a95a2a2ead21813a409c5ad7c2c008fbf418dcf3e0dc17bd970bd9ce25ea738` (expected match: True).
Vocab size 16384, `<unk>` id 0, missing ByteLevel alphabet entries 0.

## 10M pool `<unk>`
strings=64740, tokens=14669276, unk_tokens=0, strings_with_unk=0, unk_rate=0.

## Official scored-text `<unk>`
strings=389343, tokens=6367352, unk_tokens=0, strings_with_unk=0, unk_rate=0.

### By family
- BLiMP: strings=119750, tokens=1228503, unk_tokens=0, strings_with_unk=0, unk_rate=0
- EWoK: strings=30472, tokens=201894, unk_tokens=0, strings_with_unk=0, unk_rate=0
- Other: strings=25230, tokens=189973, unk_tokens=0, strings_with_unk=0, unk_rate=0
- Reading_AoA: strings=16934, tokens=185982, unk_tokens=0, strings_with_unk=0, unk_rate=0
- SuperGLUE: strings=186521, tokens=4388096, unk_tokens=0, strings_with_unk=0, unk_rate=0
- Supplement: strings=10436, tokens=172904, unk_tokens=0, strings_with_unk=0, unk_rate=0

## Comparison to the Inherited Tokenizer on Official Scored Text
Legal-tokenizer/old token ratio = 0.99858869; old unk=584, legal-tokenizer unk=0.

## Comparison to the Byte-Alphabet Tokenizer on Official Scored Text
Byte-alphabet/legal-tokenizer token ratio = 1.00000000; legal-tokenizer unk=0, byte-alphabet unk=0.

## Interpretation
This check decides whether the corrected-tokenizer official evaluations are clean with respect to byte-level coverage or should be treated as a flawed-tokenizer contrast pending a byte-alphabet same-pool retrain.
Full JSON: `experiments/archive/representation_and_objectives/data/tokenizer_byte_coverage_audit/tokenizer_byte_coverage_audit.json`

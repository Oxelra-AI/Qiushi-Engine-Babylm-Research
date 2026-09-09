# early prediction calibration: Fast representation map

## Tokenizer training pool statistics

| Tokenizer | Vocab | tokens/word | groups/word | tokens/group | over_seq256 | unk |
|-----------|-------|-------------|-------------|--------------|-------------|-----|
| legal_a01_16k | 16384 | 1.4669 | 1.0000 | 1.4669 | 15143 | 0 |
| legal_a02_bytealpha_16k | 16384 | 1.4669 | 1.0000 | 1.4669 | 15143 | 0 |
| legal_byte_bpe_24k | 24576 | 1.4281 | 1.0000 | 1.4281 | 13039 | 0 |
| legal_byte_bpe_32k | 32768 | 1.4066 | 1.0000 | 1.4066 | 11991 | 0 |
| legal_byte_bpe_40k | 40000 | 1.3943 | 1.0000 | 1.3943 | 11407 | 0 |
| old_inherited_16k | 16384 | 1.4789 | 1.0000 | 1.4789 | 15883 | 12 |

## Key eval family token ratios (new / legal_a01_16k)


### legal_byte_bpe_24k vs legal_a01_16k
- BLiMP: token ratio 0.9381
- Supplement: token ratio 0.9691
- EWoK: token ratio 0.9449
- EWoK_scored: token ratio 0.9449
- Entity: token ratio 0.9985
- COMPS: token ratio 0.9690
- GlobalPIQA: token ratio 0.9679
- SuperGLUE: token ratio 0.9586

### legal_byte_bpe_32k vs legal_a01_16k
- BLiMP: token ratio 0.9114
- Supplement: token ratio 0.9537
- EWoK: token ratio 0.9166
- EWoK_scored: token ratio 0.9166
- Entity: token ratio 0.9985
- COMPS: token ratio 0.9496
- GlobalPIQA: token ratio 0.9568
- SuperGLUE: token ratio 0.9360

### legal_byte_bpe_40k vs legal_a01_16k
- BLiMP: token ratio 0.8984
- Supplement: token ratio 0.9447
- EWoK: token ratio 0.9114
- EWoK_scored: token ratio 0.9114
- Entity: token ratio 0.9985
- COMPS: token ratio 0.9372
- GlobalPIQA: token ratio 0.9506
- SuperGLUE: token ratio 0.9234

### old_inherited_16k vs legal_a01_16k
- BLiMP: token ratio 1.0043
- Supplement: token ratio 1.0025
- EWoK: token ratio 1.0322
- EWoK_scored: token ratio 1.0322
- Entity: token ratio 1.0015
- COMPS: token ratio 1.0107
- GlobalPIQA: token ratio 1.0081
- SuperGLUE: token ratio 0.9959

## Artifacts
- JSON: `experiments/archive/representation_and_objectives/data/fast_representation_map/fast_representation_map.json`
- CSV: `experiments/archive/representation_and_objectives/data/fast_representation_map/tokenizer_comparison.csv`
- note: `research/notes/representation_and_objectives/fast_representation_map.md`

Elapsed: 432.7s

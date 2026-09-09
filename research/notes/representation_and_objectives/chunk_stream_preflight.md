# chunk stream preflight — Chunk-stream construction for experience utilization

CPU-only construction check. No model was trained or evaluated.

## What was verified

The word-boundary chunk stream exposes every tokenizer token from the 10M-word compact_view_reinvest corpus once per epoch, while charging each whitespace word once per epoch. It therefore repairs the prefix path's hidden-word debit without adding new linguistic data or extra epochs.

## legal40k

- Corpus: 64,740 rows, 10,000,000 words, 13,942,644 untruncated tokens (1.3943 tokens/word), unassigned offsets 0.
- L64: 252,270 chunks/epoch, 10,000,000 charged words, 13,942,644 active tokens, overlong words 1 with 1 continuation chunks; first update uses 998 chunks, 54461 active tokens, 16 microbatches, sampled 8031 masked tokens.
- L128: 140,303 chunks/epoch, 10,000,000 charged words, 13,942,644 active tokens, overlong words 0 with 0 continuation chunks; first update uses 555 chunks, 49987 active tokens, 9 microbatches, sampled 7360 masked tokens.
- L256: 76,164 chunks/epoch, 10,000,000 charged words, 13,942,644 active tokens, overlong words 0 with 0 continuation chunks; first update uses 302 chunks, 54397 active tokens, 5 microbatches, sampled 8025 masked tokens.
- U256 and U64_128_256 match on 100M charged words, 139,426,440 active tokens, and 2530 stage-reset updates: True.
- U256 chunks total 761,640; U64_128_256 chunks total 1,546,514. The latter changes context/order/packing, not word count, token count, or update count.

## minfreq25

- Corpus: 64,740 rows, 10,000,000 words, 14,138,512 untruncated tokens (1.4139 tokens/word), unassigned offsets 0.
- L64: 255,775 chunks/epoch, 10,000,000 charged words, 14,138,512 active tokens, overlong words 1 with 1 continuation chunks; first update uses 1011 chunks, 55457 active tokens, 16 microbatches, sampled 8180 masked tokens.
- L128: 141,373 chunks/epoch, 10,000,000 charged words, 14,138,512 active tokens, overlong words 0 with 0 continuation chunks; first update uses 559 chunks, 51798 active tokens, 9 microbatches, sampled 7635 masked tokens.
- L256: 77,092 chunks/epoch, 10,000,000 charged words, 14,138,512 active tokens, overlong words 0 with 0 continuation chunks; first update uses 305 chunks, 56380 active tokens, 5 microbatches, sampled 8342 masked tokens.
- U256 and U64_128_256 match on 100M charged words, 141,385,120 active tokens, and 2530 stage-reset updates: True.
- U256 chunks total 770,920; U64_128_256 chunks total 1,564,093. The latter changes context/order/packing, not word count, token count, or update count.

## Implementation implication

A future trainer should build stage-specific chunk streams, form exactly 253 optimizer updates per 10M-word stage epoch by distributing chunks across updates, apply full-effective-batch WWM before microbatch forward passes, and sum charged words from chunk metadata. The single overlong whitespace word at L64 should be split into continuation chunks with zero extra charged words rather than creating an over-length tensor.

JSON: `experiments/archive/representation_and_objectives/data/chunk_stream_preflight/chunk_stream_preflight.json`
CSV: `experiments/archive/representation_and_objectives/data/chunk_stream_preflight/chunk_stream_by_length.csv`

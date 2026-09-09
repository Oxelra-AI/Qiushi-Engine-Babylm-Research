# earlier analysis register stream effective-exposure audit

Tokenization/truncation comparison for the two fixed earlier analysis MAX-register 10M streams under the exact compliant tokenizer and seq256 training loader. The 100M stream is ten repetitions, so 100M totals are 10x the 10M totals.

## Total arm comparison

| arm | words | tokens before trunc | tokens after trunc | truncated rows | truncated tokens | tokens/word | truncated row fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| childspeech_removed | 10000000 | 14586586 | 14255764 | 13457 | 330822 | 1.458659 | 0.206039 |
| adultprose_removed | 10000000 | 14697968 | 14348535 | 14741 | 349433 | 1.469797 | 0.225698 |

## childspeech_removed minus adultprose_removed

- token_len_sum: -111382
- token_after_trunc_sum: -92771
- truncated_rows: -1284
- truncated_tokens: -18611
- mean_tokens_per_word: -0.011138
- truncated_row_fraction: -0.019659

## Reading

The childspeech-removal stream has fewer effective tokens and less truncation than the adultprose-removal stream. Since the observed seed43022 register contrast is negative for childspeech_removed - adultprose_removed, tokenization/truncation burden runs opposite to the observed sign.

JSON: `experiments/archive/frontier_consolidation/data/register_stream_effective_exposure_audit/effective_exposure_audit.json`
CSV totals: `experiments/archive/frontier_consolidation/data/register_stream_effective_exposure_audit/arm_totals.csv`
CSV by-kind differences: `experiments/archive/frontier_consolidation/data/register_stream_effective_exposure_audit/kind_differences.csv`

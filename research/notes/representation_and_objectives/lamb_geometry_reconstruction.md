# lamb config audit — LAMB×sequence-curriculum geometry reconstruction

Data: `experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl`
Tokenizer: `experiments/archive/representation_and_objectives/data/legal_representation_route_map/tokenizers/legal_byte_bpe_40k`

## Totals
- rows: 647,400; charged words: 100,000,000
- kept chunks: 1,263,068; optimizer steps: 4,935 vs fixed-256 matched reference 2,529 (ratio 1.951)
- candidate tokens kept: 139,266,696; full tokenized stream: 139,426,440; visible fraction 0.998854; candidate-token ratio vs fixed-256 reference 1.016
- mid-word chunk boundaries: 183,226 in 147,427 rows (22.7722% of rows)
- dropped <8-token tails: 159,744 tokens (0.114572% of full tokens)
- WWM groups after per-chunk restart: 100,078,654; original kept groups: 99,895,428; overcount 183,226 (0.1834% of kept original groups)
- expected masked tokens under lamb curriculum route decision exact per-chunk WWM: 20908432.4; same chunks with Bernoulli WWM would be 20890004.4 (ratio 1.0009)
- expected masked-token ratio versus fixed-256 Bernoulli reference: 1.017

## Per phase
| phase | seq_len | words | chunks | steps | words/step | tokens/step | mid-word boundaries | tail tokens | expected masked tokens/step |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 64 | 20,000,000 | 484,784 | 1,894 | 10559.7 | 14686.0 | 102,500 | 69,984 | 2210.6 |
| 1 | 128 | 30,000,000 | 410,964 | 1,606 | 18680.0 | 26021.8 | 62,331 | 36,945 | 3903.1 |
| 2 | 256 | 50,000,000 | 367,320 | 1,435 | 34843.2 | 48543.8 | 18,395 | 52,815 | 7284.4 |

## Interpretation for lamb curriculum route decision readout
- The active runs are a joint LAMB×sequence-curriculum×chunking×masking intervention, not a separated optimizer or curriculum result.
- The trainer uses fixed token slicing. A chunk can begin inside a wordpiece group and then restarts that group for WWM in the new chunk.
- Tails shorter than eight tokens are not observed by the model, while row words are charged when the first chunk is consumed.
- Because rows produce multiple chunks at shorter lengths, the optimizer update count and token batch geometry differ substantially from the matched fixed-256 AdamW baselines.
- Any lamb curriculum route decision cheap7 score should be compared to matched cheap7 vectors, not to nine-column Overall values.

Summary JSON: `experiments/archive/representation_and_objectives/data/lamb_geometry_reconstruction/lamb_geometry_summary.json`
Phase CSV: `experiments/archive/representation_and_objectives/data/lamb_geometry_reconstruction/phase_geometry.csv`

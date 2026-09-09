# roberta reader and compact marginal atlas natural compact-vs-repeat mechanism atlas

Created UTC: 2026-09-02T02:26:26Z

This is CPU/file/tokenizer analysis only. It did not touch running RoBERTa jobs. The rows here are underlying compact-pair objects, not packed changed JSONL rows.

## Inputs
- pairs: `experiments/archive/frontier_consolidation/data/factorial_view_candidate_audit/compact_candidate_pairs.jsonl` SHA `6d0ac85dec1718e5f5663d09123e2f62a23f7cceb0b491b83a34f7bfca59d32d` rows 12155
- legal spatial repair route status tokenizer: `experiments/archive/frontier_consolidation/data/compliant_tokenizer` tokenizer.json SHA `91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9`

## Main geometry
- Total source/view words in changed rows: 261803 / 161708 (ratio 0.6177)
- Mean compact source-content coverage: 67.35%
- Mean first-N repeat source-content coverage: 59.81%
- Mean compact-repeat coverage delta: 7.54%; positive-delta pairs 55.10%
- Mean compact tail-content coverage after the repeat length: 70.59%; pairs with any tail recovery 97.22%
- Compact content fraction 64.74% vs repeat 48.84%
- Pairs with any source-absent compact content word: 75.86%; source-absent fraction of compact content totals 17.11%
- Pairs with any tail-only compact content word: 96.29%; tail-only fraction of compact content totals 32.55%

## Legal-tokenizer load in changed rows
- individual pair-view encoding: compact active tokens 247878; repeat active tokens 216589; compact-minus-repeat 31289. Use roberta transfer pair scaffold ready's packed-row audit as the exact training-stream number (+24000 active/candidate tokens per 10M pass; +240000 over 100M).
- compact word groups: 161708; repeat word groups: 161708; compact-minus-repeat 0
- compact/repeat truncated rows: 0 / 0

## Source-position deciles
| decile | content positions | compact coverage | repeat coverage | delta |
|---:|---:|---:|---:|---:|
| 0 | 15080 | 68.28% | 100.00% | -31.72% |
| 1 | 12538 | 62.63% | 100.00% | -37.37% |
| 2 | 13435 | 62.06% | 100.00% | -37.94% |
| 3 | 12708 | 63.18% | 99.68% | -36.50% |
| 4 | 11953 | 63.52% | 94.58% | -31.06% |
| 5 | 13694 | 64.08% | 69.58% | -5.50% |
| 6 | 13147 | 63.90% | 31.75% | 32.15% |
| 7 | 12533 | 65.96% | 7.48% | 58.49% |
| 8 | 12777 | 70.88% | 0.11% | 70.77% |
| 9 | 15151 | 76.32% | 0.00% | 76.32% |

## Scientific use
This file names the bundled data marginal: compact changes source-position spread, content density, source-absent content, lexical overlap with repeat, and BPE/candidate-token exposure while the full pool reinvests saved words in extra source diversity. For exact training-stream BPE exposure use the roberta transfer pair scaffold ready packed-row audit; this atlas decomposes the underlying pair-level source/view relation. It should prevent over-reading the pending RoBERTa result as a single-factor verdict.

Artifacts: `mechanism_atlas.json` and `pair_atlas.csv` in this directory.

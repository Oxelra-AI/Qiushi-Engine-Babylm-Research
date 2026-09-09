# pairaware sequence accounting sequence vs fixed-seq256 accounting
CPU-only correction of the sequence-route interpretation. No model was trained and no official evaluation text was read.

Input: `experiments/archive/frontier_consolidation/data/pairaware_sequence_accounting/pairaware_sequence_accounting.json`. Pool SHA matched: `True`.

## Why this comparison matters
The large faithful/prefix ratios from sequence curriculum loop measurement-pairaware sequence accounting are real, but they are ratios against the flawed prefix-sliced `seq_len_schedule` path. The current strongest legal model is fixed seq256. Against fixed seq256, a faithful sequence stream mainly recovers the row-truncation tail and changes context/update geometry; it is not a 1.6x increase in training text seen.

## Schedule comparison against fixed seq256
| tokenizer | schedule | charged words | fixed active tokens | sequence active tokens | token ratio vs fixed | token Δ | expected masked-token Δ | fixed steps | sequence steps | step ratio vs fixed | pair full | pair overlong |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| legal16k | 64x3_128x4_256x3 | 100000000 | 142948930 | 146645190 | 1.0259 | 3696260 | 554439 | 2530 | 2858 | 1.130 | 0.938 | 0.062 |
| legal16k | 64x7_256x3 | 100000000 | 142948930 | 146645190 | 1.0259 | 3696260 | 554439 | 2530 | 2766 | 1.093 | 0.858 | 0.142 |
| minfreq50_supportfloor | 64x3_128x4_256x3 | 100000000 | 141529040 | 144836000 | 1.0234 | 3306960 | 496044 | 2530 | 2825 | 1.117 | 0.944 | 0.056 |
| minfreq50_supportfloor | 64x7_256x3 | 100000000 | 141529040 | 144836000 | 1.0234 | 3306960 | 496044 | 2530 | 2733 | 1.080 | 0.871 | 0.129 |

## Scientific reading
- The faithful pair-aware sequence stream is legally clean and implementation-feasible, but relative to the actual fixed-seq256 endpoint it adds only about 2.3-2.6% active tokens (the untruncated tail), while adding about 8-13% optimizer steps under inverse batch sizing.
- Its possible value is therefore not simple token-volume expansion; it would have to come from shorter-context learning dynamics, row-tail recovery, and preserving compact source+rewrite atoms during early short-context exposure.
- This makes sequence curriculum a plausible later route, especially because it is compatible with minfreq50 and pair-aware packing, but it is not strong enough to jump ahead of the active word-mean result or the already-hardened minfreq50 screen.

Full JSON: `experiments/archive/frontier_consolidation/data/sequence_vs_fixed256_accounting/sequence_vs_fixed256_accounting.json`
CSV: `experiments/archive/frontier_consolidation/data/sequence_vs_fixed256_accounting/sequence_vs_fixed256_accounting.csv`

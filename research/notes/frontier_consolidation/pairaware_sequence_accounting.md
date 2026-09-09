# pairaware sequence accounting pair-aware sequence accounting
CPU-only combination of full-pool tokenizer-group chunk accounting with compact-block pair-atomic chunk accounting. No model was trained and no official evaluation text was read.

Input JSON: `experiments/archive/frontier_consolidation/data/tokenizer_group_chunk_measurement/tokenizer_group_chunk_measurement.json`. Pool SHA matched: `True`.

## Per-length full-pool accounting
Pair-aware chunks differ from greedy chunks only in the compact changed block. Active tokens are unchanged; step changes come from atom-preserving boundaries.

| tokenizer | L | active-token ratio vs prefix | greedy chunks | pair-aware chunks | chunk Δ | greedy steps | pair-aware steps | step Δ | step ratio vs prefix | pair full prefix/greedy/atomic | overlong atomic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| legal16k | 64 | 3.539 | 264588 | 266599 | 2011 | 259 | 261 | 2 | 1.032 | 0.184/0.387/0.797 | 0.203 |
| legal16k | 128 | 1.772 | 144471 | 145032 | 561 | 283 | 284 | 1 | 1.123 | 0.496/0.756/0.998 | 0.002 |
| legal16k | 256 | 1.026 | 79882 | 79882 | 0 | 313 | 313 | 0 | 1.237 | 0.995/0.996/1.000 | 0.000 |
| minfreq50_supportfloor | 64 | 3.496 | 261722 | 263665 | 1943 | 256 | 258 | 2 | 1.020 | 0.192/0.402/0.816 | 0.184 |
| minfreq50_supportfloor | 128 | 1.750 | 143340 | 143785 | 445 | 280 | 281 | 1 | 1.111 | 0.513/0.759/0.999 | 0.001 |
| minfreq50_supportfloor | 256 | 1.023 | 78857 | 78857 | 0 | 309 | 309 | 0 | 1.221 | 0.998/0.998/1.000 | 0.000 |

## Ten-epoch schedules
| tokenizer | schedule | charged words | active-token ratio vs prefix | prefix steps | greedy steps | pair-aware steps | step Δ vs greedy | step ratio vs prefix | pair full prefix/greedy/atomic | overlong atomic |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| legal16k | 64x3_128x4_256x3 | 100000000 | 1.658 | 2530 | 2848 | 2858 | 10 | 1.130 | 0.552/0.717/0.938 | 0.062 |
| legal16k | 64x7_256x3 | 100000000 | 2.040 | 2530 | 2752 | 2766 | 14 | 1.093 | 0.427/0.569/0.858 | 0.142 |
| minfreq50_supportfloor | 64x3_128x4_256x3 | 100000000 | 1.646 | 2530 | 2815 | 2825 | 10 | 1.117 | 0.562/0.723/0.944 | 0.056 |
| minfreq50_supportfloor | 64x7_256x3 | 100000000 | 2.027 | 2530 | 2719 | 2733 | 14 | 1.080 | 0.434/0.580/0.871 | 0.129 |

## Scientific reading
- Pair-aware group chunking is affordable: on the 64x3/128x4/256x3 schedule it adds only a few optimizer steps over greedy group chunks while preserving about 0.94 of compact source+rewrite pairs as full same-window atoms across the ten-epoch mix.
- The 64x7/256x3 schedule exposes more active tokens but sacrifices more pair atoms at L64; it is a sharper sequence-first intervention and less obviously compatible with the validated compact-view mechanism.
- This strengthens the construction case for a future faithful pair-aware chunk-stream trainer, but it remains subordinate to the live word-mean decision and the prepared support-floor screen.

Full JSON: `experiments/archive/frontier_consolidation/data/pairaware_sequence_accounting/pairaware_sequence_accounting.json`
CSV: `experiments/archive/frontier_consolidation/data/pairaware_sequence_accounting/pairaware_by_length.csv`, `experiments/archive/frontier_consolidation/data/pairaware_sequence_accounting/pairaware_schedules.csv`
